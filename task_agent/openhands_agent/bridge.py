#!/usr/bin/env python3
"""Execution bridge from versioned HarnessFix candidates to Better Harness."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import yaml

from failure_analysis.openhands_io import normalize_rollout


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = Path(__file__).resolve().parent / "original"
SUPPORTED_TASKS = {
    "woocommerce_stock_alert_s2l",
    "machine_operating_s2l",
    "refactorbench",
    "webarena",
}


def _bundle_payload(candidate_dir: Path) -> str:
    """Return a deterministic base64 zip of a complete candidate bundle."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidate_dir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(candidate_dir)
            if "__pycache__" in relative.parts or path.suffix == ".pyc" or ".git" in relative.parts:
                continue
            info = zipfile.ZipInfo(str(relative).replace("\\", "/"), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pack_candidate(candidate_dir: Path, output_path: Path) -> Path:
    """Compile a multi-file candidate into Better Harness's one-file contract.

    Better Harness mounts only ``agent_file`` into its Docker OpenHands server.
    The launcher unpacks the versioned bundle in each process and re-exports the
    required builder plus optional workspace-script/hook entry points.
    """
    candidate_dir = candidate_dir.resolve()
    if not (candidate_dir / "agent.py").is_file():
        raise FileNotFoundError(candidate_dir / "agent.py")
    payload = _bundle_payload(candidate_dir)
    launcher = f'''from __future__ import annotations

import atexit
import base64
import importlib.util
import io
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

_BUNDLE_PAYLOAD = {payload!r}
_BUNDLE_ROOT = Path(tempfile.mkdtemp(prefix="harnessfix_openhands_"))
atexit.register(shutil.rmtree, _BUNDLE_ROOT, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_BUNDLE_PAYLOAD))) as _archive:
    _archive.extractall(_BUNDLE_ROOT)
sys.path.insert(0, str(_BUNDLE_ROOT))
_SPEC = importlib.util.spec_from_file_location("_harnessfix_candidate_impl", _BUNDLE_ROOT / "agent.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Unable to load packed HarnessFix OpenHands candidate")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
build_agent = _MODULE.build_agent
if hasattr(_MODULE, "get_workspace_scripts"):
    get_workspace_scripts = _MODULE.get_workspace_scripts
if hasattr(_MODULE, "get_hook_config"):
    get_hook_config = _MODULE.get_hook_config
'''
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(launcher, encoding="utf-8")
    return output_path


def materialize_candidate(
    *,
    better_root: Path,
    task_id: str,
    prompt_name: str,
    output_dir: Path,
) -> Path:
    if task_id not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task_id {task_id!r}; expected one of {sorted(SUPPORTED_TASKS)}")
    better_root = better_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate directory: {output_dir}")
    prompt_path = better_root / "tasks" / task_id / "prompts" / f"{prompt_name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)
    shutil.copytree(TEMPLATE_ROOT, output_dir)
    (output_dir / "prompts" / "system.md").write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "task_id": task_id,
            "prompt_name": prompt_name,
            "source_prompt": str(prompt_path.resolve()),
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return output_dir


def _load_run_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {path}")
    return config


def _effective_rollout_dir(better_root: Path, config: dict[str, Any], agent_file: Path) -> Path:
    rollout = f"{config.get('rollout_version', 'v0')}_{agent_file.stem}"
    return (
        better_root
        / "results"
        / str(config["task_id"])
        / f"{config['model_name']}_{config.get('prompt_name', 'default')}"
        / "rollouts"
        / rollout
    )


def run_better_harness(
    *,
    better_root: Path,
    base_config: Path,
    candidate_dir: Path,
    rollout_version: str,
    normalized_output: Path,
    success_threshold: float,
    dry_run: bool = False,
) -> dict[str, Any]:
    better_root = better_root.resolve()
    candidate_dir = candidate_dir.resolve()
    if not (candidate_dir / "agent.py").exists():
        raise FileNotFoundError(candidate_dir / "agent.py")
    config_path = base_config if base_config.is_absolute() else better_root / base_config
    config = _load_run_config(config_path.resolve())
    task_id = str(config["task_id"])
    if task_id not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task_id in run config: {task_id}")
    candidate_config_path = candidate_dir / "config.json"
    candidate_config = json.loads(candidate_config_path.read_text(encoding="utf-8"))
    if candidate_config.get("task_id") != task_id:
        raise ValueError(
            f"Candidate task_id {candidate_config.get('task_id')!r} does not match run config {task_id!r}"
        )

    generated_dir = REPO_ROOT / "results" / "openhands_bridge" / "generated_configs"
    safe_candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_dir.name).strip("_") or "candidate"
    candidate_digest = hashlib.sha256(_bundle_payload(candidate_dir).encode("ascii")).hexdigest()[:12]
    packed_dir = REPO_ROOT / "results" / "openhands_bridge" / "packed_candidates"
    packed_agent = packed_dir / f"{task_id}_{rollout_version}_{safe_candidate}_{candidate_digest}.py"
    config["agent_file"] = str(packed_agent)
    config["rollout_version"] = rollout_version
    generated_path = generated_dir / f"{task_id}_{rollout_version}_{safe_candidate}_{candidate_digest}.yaml"
    collect_cmd = ["uv", "run", "python", "-m", "src.collect", "--config", str(generated_path)]
    evaluate_cmd = ["uv", "run", "python", "-m", "src.evaluate", "--config", str(generated_path)]
    rollout_dir = _effective_rollout_dir(better_root, config, packed_agent)

    if dry_run:
        return {
            "config": config,
            "collect_command": collect_cmd,
            "evaluate_command": evaluate_cmd,
            "packed_agent": str(packed_agent),
            "rollout_dir": str(rollout_dir),
            "normalized_output": str(normalized_output.resolve()),
        }

    pack_candidate(candidate_dir, packed_agent)
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    subprocess.run(collect_cmd, cwd=better_root, check=True)
    subprocess.run(evaluate_cmd, cwd=better_root, check=True)
    eval_results = rollout_dir / "eval_results.yaml"
    results_path, traces_root = normalize_rollout(
        eval_results_path=eval_results,
        better_root=better_root,
        output_dir=normalized_output,
        task_id=task_id,
        success_threshold=success_threshold,
    )
    return {
        "generated_config": str(generated_path),
        "rollout_dir": str(rollout_dir),
        "results": str(results_path),
        "traces": str(traces_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge HarnessFix OpenHands candidates to Better Harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize", help="Create task-specific H0 from a Better prompt")
    materialize.add_argument("--better-root", type=Path, required=True)
    materialize.add_argument("--task-id", choices=sorted(SUPPORTED_TASKS), required=True)
    materialize.add_argument("--prompt-name", default="default")
    materialize.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run collect/evaluate and normalize artifacts")
    run.add_argument("--better-root", type=Path, required=True)
    run.add_argument("--base-config", type=Path, required=True)
    run.add_argument("--candidate-dir", type=Path, required=True)
    run.add_argument("--rollout-version", required=True)
    run.add_argument("--normalized-output", type=Path, required=True)
    run.add_argument("--success-threshold", type=float, default=1.0)
    run.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "materialize":
        output = materialize_candidate(
            better_root=args.better_root,
            task_id=args.task_id,
            prompt_name=args.prompt_name,
            output_dir=args.output_dir,
        )
        print(output)
        return
    result = run_better_harness(
        better_root=args.better_root,
        base_config=args.base_config,
        candidate_dir=args.candidate_dir,
        rollout_version=args.rollout_version,
        normalized_output=args.normalized_output,
        success_threshold=args.success_threshold,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

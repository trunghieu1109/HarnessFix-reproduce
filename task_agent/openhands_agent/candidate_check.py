"""Check a packed candidate's import/build contract without running a task."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

from failure_analysis.secret_redaction import redact_secrets
from task_agent.openhands_agent.bridge import pack_candidate


def _run_check_process(command: list[str], *, cwd: Path, env: dict, timeout: int) -> subprocess.CompletedProcess:
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # uv spawns Python; killing only uv leaves the candidate worker running.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def check_candidate(*, better_root: Path, candidate_dir: Path, output_path: Path,
                    timeout: int = 60, model_settings: dict | None = None) -> dict:
    """Use Better's Python/SDK and the same launcher as collect, without an LLM call.

    Only failures inside candidate import/build become repair feedback. Missing SDK,
    failed uv startup, or a missing worker report remain orchestration errors.
    """
    with tempfile.TemporaryDirectory(prefix="harnessfix_candidate_check_") as directory:
        root = Path(directory)
        packed = pack_candidate(candidate_dir, root / "packed_agent.py")
        status_path = root / "status.json"
        workspace = root / "workspace"
        workspace.mkdir()
        command = ["uv", "run", "--offline", "python", "-B", str(Path(__file__).resolve()),
                   "--better-root", str(better_root.resolve()), "--packed-agent", str(packed),
                   "--workspace", str(workspace), "--status-file", str(status_path)]
        if model_settings:
            generation_fields = ("model", "temperature", "reasoning_effort", "max_input_tokens",
                                 "max_output_tokens", "extra_body")
            command += ["--model-settings", json.dumps({key: model_settings[key]
                                                       for key in generation_fields if key in model_settings})]
        timed_out = False
        python_path = os.pathsep.join(filter(None, (
            str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", ""),
        )))
        try:
            result = _run_check_process(command, cwd=better_root, timeout=timeout,
                                        env=dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True",
                                                 OPENHANDS_SUPPRESS_BANNER="1", PYTHONPATH=python_path))
        except subprocess.TimeoutExpired:
            timed_out = True
            result = None
        if not status_path.is_file():
            detail = "worker startup timed out" if timed_out else (result.stderr or result.stdout)[-4000:]
            raise RuntimeError("Candidate check infrastructure failed: " + redact_secrets(detail))
        report = json.loads(status_path.read_text())
        if timed_out:
            report.update(passed=False, status="failed", exception_type="TimeoutError",
                          message=f"Candidate {report['stage']} exceeded {timeout} seconds.", traceback="")
        elif report.get("status") != "complete" or result.returncode != int(not report["passed"]):
            raise RuntimeError("Candidate check worker did not complete its report")
        report = redact_secrets(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report


def _write_status(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report))
    temporary.replace(path)


def _check_packed_candidate(packed: Path, workspace: Path, status_path: Path,
                            model_settings: dict | None = None) -> dict:
    # SDK/dependency failures are outside the candidate exception boundary.
    from openhands.sdk import Agent, LLM

    llm_settings = {"model": "openai/harnessfix-preflight", "max_input_tokens": 120000,
                    "max_output_tokens": 8192} | (model_settings or {})
    llm = LLM(**(llm_settings | {"api_key": "EMPTY", "api_base": "http://127.0.0.1:1/v1",
                                 "log_completions": False}))
    report = {"schema_version": "harnessfix.candidate_check.v1", "passed": False,
              "status": "running", "stage": "import", "llm_calls": 0, "llm_calls_blocked": 0}
    _write_status(status_path, report)

    def block_model_request(*args, **kwargs):
        report["llm_calls_blocked"] += 1
        raise RuntimeError("Candidate initialization must not issue LLM requests during the check")

    try:
        with patch.object(LLM, "completion", block_model_request), patch.object(LLM, "responses", block_model_request):
            spec = importlib.util.spec_from_file_location("_harnessfix_preflight", packed)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot import candidate launcher")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            report["stage"] = "build_agent"
            _write_status(status_path, report)
            agent = module.build_agent(base_dir=str(workspace), llm=llm)
            if not isinstance(agent, Agent):
                raise TypeError("build_agent must return an openhands.sdk.Agent")
            report["passed"] = True
    except Exception as exc:
        report.update(exception_type=type(exc).__name__, message=str(exc),
                      traceback=traceback.format_exc())
    report["status"] = "complete"
    _write_status(status_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--better-root", type=Path, required=True)
    parser.add_argument("--packed-agent", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--model-settings", type=json.loads, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(args.better_root.resolve()))
    report = _check_packed_candidate(args.packed_agent, args.workspace, args.status_file, args.model_settings)
    raise SystemExit(int(not report["passed"]))


if __name__ == "__main__":
    main()

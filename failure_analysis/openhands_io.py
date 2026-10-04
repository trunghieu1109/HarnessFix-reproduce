#!/usr/bin/env python3
"""Normalize Better Harness/OpenHands artifacts for the HarnessFix pipeline.

The adapter deliberately does not add observations or inspect task state.  It
indexes the trace and evaluator artifacts already produced by Better Harness,
then emits the small ``results.json`` contract consumed by HarnessFix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml


WORKSPACE_RE = re.compile(r"^example(?P<example>\d+)_rollout(?P<rollout>\d+)$")


def _load_structured(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def _resolve_workspace(workspace: str, better_root: Path) -> Path:
    path = Path(workspace)
    if not path.is_absolute():
        path = better_root / path
    return path.resolve()


def _workspace_identity(workspace: Path) -> tuple[int, int]:
    match = WORKSPACE_RE.fullmatch(workspace.name)
    if not match:
        raise ValueError(
            f"Unsupported Better Harness workspace name {workspace.name!r}; "
            "expected example<N>_rollout<M>"
        )
    return int(match.group("example")), int(match.group("rollout"))


def _single_trace(log_dir: Path, pattern: str) -> Path | None:
    matches = sorted(log_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one {pattern} in {log_dir}, found {len(matches)}")
    return matches[0].resolve()


def _trace_error(trace_path: Path | None) -> str | None:
    if trace_path is None:
        return "filtered trace is missing"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    error = trace.get("error") if isinstance(trace, dict) else None
    return str(error) if error else None


def _stable_instance_id(task_id: str, example_index: int, rollout_id: int) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")
    return f"{safe_task}__example{example_index}__rollout{rollout_id}"


def _stable_task_instance_id(task_id: str, example_index: int) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")
    return f"{safe_task}__example{example_index}"


def normalize_rollout(
    *,
    eval_results_path: Path,
    better_root: Path,
    output_dir: Path,
    task_id: str,
    success_threshold: float = 1.0,
) -> tuple[Path, Path]:
    """Write per-rollout manifests and a HarnessFix-compatible result JSON."""
    eval_results_path = eval_results_path.resolve()
    better_root = better_root.resolve()
    output_dir = output_dir.resolve()
    raw_results = _load_structured(eval_results_path)
    if not isinstance(raw_results, list):
        raise ValueError(f"Expected a list in {eval_results_path}")

    manifests_root = output_dir / "traces"
    normalized_records: dict[str, dict[str, Any]] = {}
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []
    unsupported_ids: list[str] = []
    numeric_scores: list[float] = []
    rollout_groups: dict[str, list[str]] = {}

    for eval_result in raw_results:
        if not isinstance(eval_result, dict) or not eval_result.get("workspace_dir"):
            raise ValueError("Each Better Harness eval result must contain workspace_dir")
        workspace = _resolve_workspace(str(eval_result["workspace_dir"]), better_root)
        example_index, rollout_id = _workspace_identity(workspace)
        instance_id = _stable_instance_id(task_id, example_index, rollout_id)
        task_instance_id = _stable_task_instance_id(task_id, example_index)
        rollout_groups.setdefault(task_instance_id, []).append(instance_id)
        log_dir = workspace.parent / f"{workspace.name}_logs"
        trace_path = _single_trace(log_dir, "trace_*.json")
        raw_trace_path = _single_trace(log_dir, "raw_trace_*.json")
        trace_error = _trace_error(trace_path)
        score_value = eval_result.get("score")
        evaluator_status = "scored"
        resolved = False

        if score_value is None:
            evaluator_status = "unsupported"
            unsupported_ids.append(instance_id)
            error_ids.append(instance_id)
        else:
            score = float(score_value)
            numeric_scores.append(score)
            resolved = score >= success_threshold
            if resolved:
                resolved_ids.append(instance_id)
            elif trace_error:
                evaluator_status = "agent_error"
                error_ids.append(instance_id)
            else:
                unresolved_ids.append(instance_id)

        manifest = {
            "schema_version": "harnessfix.openhands_manifest.v1",
            "instance_id": instance_id,
            "task_instance_id": task_instance_id,
            "task_id": task_id,
            "example_index": example_index,
            "rollout_id": rollout_id,
            "workspace_dir": str(workspace),
            "log_dir": str(log_dir.resolve()),
            "trace_path": str(trace_path) if trace_path else None,
            "raw_trace_path": str(raw_trace_path) if raw_trace_path else None,
            "eval_results_path": str(eval_results_path),
            "eval_result": eval_result,
            "score": score_value,
            "success_threshold": success_threshold,
            "resolved": resolved,
            "evaluator_status": evaluator_status,
            "trace_error": trace_error,
        }
        manifest_dir = manifests_root / instance_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        normalized_records[instance_id] = {
            "instance_id": instance_id,
            "task_instance_id": task_instance_id,
            "example_index": example_index,
            "rollout_id": rollout_id,
            "score": score_value,
            "resolved": resolved,
            "evaluator_status": evaluator_status,
            "feedback": eval_result.get("feedback"),
            "workspace_dir": str(workspace),
            "manifest_path": str(manifest_path),
        }

    normalized = {
        "schema_version": "harnessfix.eval_results.v1",
        "mode": "openhands",
        "task_id": task_id,
        "success_threshold": success_threshold,
        "source_eval_results": str(eval_results_path),
        "all_ids": list(normalized_records),
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "empty_patch_ids": [],
        "error_ids": error_ids,
        "unsupported_ids": unsupported_ids,
        "score_mean": fmean(numeric_scores) if numeric_scores else None,
        "rollout_groups": rollout_groups,
        "records": normalized_records,
    }
    results_path = output_dir / "results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index_path = output_dir / "manifest_index.json"
    index_path.write_text(
        json.dumps(
            {instance_id: record["manifest_path"] for instance_id, record in normalized_records.items()},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return results_path, manifests_root


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "harnessfix.openhands_manifest.v1":
        raise ValueError(f"Unsupported OpenHands manifest schema in {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Better Harness artifacts for HarnessFix")
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--better-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    args = parser.parse_args()
    results_path, traces_root = normalize_rollout(
        eval_results_path=args.eval_results,
        better_root=args.better_root,
        output_dir=args.output_dir,
        task_id=args.task_id,
        success_threshold=args.success_threshold,
    )
    print(json.dumps({"results": str(results_path), "traces": str(traces_root)}, indent=2))


if __name__ == "__main__":
    main()

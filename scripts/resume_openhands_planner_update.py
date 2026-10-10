"""Resume an interrupted aggregate stage after updating only the planner code."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

from scripts.resume_frozen_openhands import PIPELINE_RELATIVE, verify_completed_stages


REPO_ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_RELATIVE = "failure_analysis/aggregate_results.py"


def prepare_resume(repo_root: Path, run_dir: Path) -> tuple[ModuleType, object, dict]:
    snapshot = json.loads((run_dir / "experiment.json").read_text())
    changed = [name for name, expected in snapshot["pipeline_source_hashes"].items()
               if hashlib.sha256((repo_root / name).read_bytes()).hexdigest() != expected]
    if changed != [AGGREGATE_RELATIVE]:
        raise ValueError(f"Planner-only resume requires exactly {AGGREGATE_RELATIVE} to change; found {changed}")

    pending = [path for path in (run_dir / "iterations").glob("v*")
               if re.fullmatch(r"v\d+", path.name)
               and not (run_dir / "stages" / f"aggregate_{path.name}.json").exists()
               and (path / "plan.md.raw.txt").exists()
               and any((run_dir / "stages").glob(f"*_train_h*_{path.name}_analysis.json"))]
    if len(pending) != 1:
        raise ValueError("Expected exactly one interrupted aggregate stage with completed train analysis")

    pipeline_path = (repo_root / PIPELINE_RELATIVE).resolve()
    module = ModuleType("task_agent.openhands_agent.planner_resume_pipeline")
    module.__file__ = str(pipeline_path)
    sys.modules[module.__name__] = module
    exec(compile(pipeline_path.read_bytes(), str(pipeline_path), "exec"), module.__dict__)
    path_digest = module._path_digest
    aggregate_path = (repo_root / AGGREGATE_RELATIVE).resolve()
    original_digest = snapshot["pipeline_source_hashes"][AGGREGATE_RELATIVE]
    adopted_digest = path_digest(aggregate_path)

    def resume_source_digest(path: Path) -> str:
        if path.resolve() == aggregate_path:
            if path_digest(path) != adopted_digest:
                raise ValueError("Aggregate planner source changed again after resume verification")
            # Existing stages retain their original fingerprint. The explicit
            # planner adoption is recorded separately before continuing work.
            return original_digest
        return path_digest(path)

    module._path_digest = resume_source_digest
    experiment = module.OpenHandsPipeline(snapshot["config"], run_dir=run_dir)
    completed = verify_completed_stages(module, run_dir)
    for marker in (run_dir / "stages").glob("*.json"):
        match = re.fullmatch(r"(train|val|test)_h(\d+)", marker.stem)
        if match:
            experiment._validate_run(match[1], int(match[2]))
    record = {
        "run_dir": str(run_dir),
        "original_experiment_fingerprint": snapshot["fingerprint"],
        "pending_stage": f"aggregate_{pending[0].name}",
        "completed_stages_verified": completed,
        "source_updates": {
            AGGREGATE_RELATIVE: {"original_sha256": original_digest, "resumed_sha256": adopted_digest},
        },
    }
    return module, experiment, record


def record_resume(repo_root: Path, run_dir: Path, record: dict) -> Path:
    source = repo_root / AGGREGATE_RELATIVE
    expected = record["source_updates"][AGGREGATE_RELATIVE]["resumed_sha256"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Aggregate planner source changed before recording resume")
    timestamp = datetime.now(timezone.utc)
    destination = run_dir / "runtime" / "planner_resumes" / timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True)
    shutil.copyfile(source, destination / "aggregate_results.py")
    iteration = record["pending_stage"].removeprefix("aggregate_")
    for name in ("plan.md.raw.txt", "plan.md.aggregate_agent.traj.json"):
        artifact = run_dir / "iterations" / iteration / name
        if artifact.exists():
            shutil.copyfile(artifact, destination / name)
    (destination / "resume.json").write_text(json.dumps(
        record | {"timestamp": timestamp.isoformat()}, indent=2,
    ) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    _, experiment, record = prepare_resume(REPO_ROOT, run_dir)
    print(f"[resume] Verified {record['completed_stages_verified']} completed stages", flush=True)
    print(f"[resume] Continue at {record['pending_stage']} with the updated planner; other inputs unchanged", flush=True)
    if args.dry_run:
        print("[resume] Dry run complete; no files written and no model or benchmark calls", flush=True)
        return
    archive = record_resume(REPO_ROOT, run_dir, record)
    print(f"[resume] Planner update and original failure archived in {archive}", flush=True)
    experiment.run()


if __name__ == "__main__":
    main()

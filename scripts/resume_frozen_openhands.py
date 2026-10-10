#!/usr/bin/env python3
"""Resume an experiment with its verified original pipeline source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_RELATIVE = "task_agent/openhands_agent/pipeline.py"


def load_frozen_pipeline(repo_root: Path, run_dir: Path) -> tuple[ModuleType, dict]:
    snapshot = json.loads((run_dir / "experiment.json").read_text())
    frozen_path = run_dir / "runtime/pipeline_original.py"
    expected = snapshot["pipeline_source_hashes"][PIPELINE_RELATIVE]
    source = frozen_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("Frozen pipeline source does not match the experiment's saved SHA256")

    module = ModuleType("task_agent.openhands_agent.frozen_pipeline")
    # Keep the original repository paths while executing the archived source.
    original_path = (repo_root / PIPELINE_RELATIVE).resolve()
    module.__file__ = str(original_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(frozen_path), "exec"), module.__dict__)
    path_digest = module._path_digest

    def running_source_digest(path: Path) -> str:
        # Fingerprint the code actually executing for this one module. All
        # other code, models, inputs and cached artifacts keep normal checks.
        if path.resolve() == original_path:
            return path_digest(frozen_path)
        return path_digest(path)

    module._path_digest = running_source_digest
    return module, snapshot


def verify_completed_stages(module: ModuleType, run_dir: Path) -> int:
    stages = sorted((run_dir / "stages").glob("*.json"))
    for marker in stages:
        outputs = json.loads(marker.read_text())["outputs"]
        for value, expected in outputs.items():
            path = Path(value)
            if not path.exists() or module._path_digest(path) != expected:
                raise ValueError(f"Completed stage artifact changed: {marker.stem}: {path}")
    return len(stages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    run_dir = args.run_dir.resolve()
    module, snapshot = load_frozen_pipeline(REPO_ROOT, run_dir)
    # The original constructor verifies the full experiment fingerprint,
    # including unchanged model settings, data and all other source files.
    experiment = module.OpenHandsPipeline(snapshot["config"], run_dir=run_dir)
    completed = verify_completed_stages(module, run_dir)
    print(f"[recovery] Original pipeline SHA256: {snapshot['pipeline_source_hashes'][PIPELINE_RELATIVE]}", flush=True)
    print(f"[recovery] Verified {completed} completed stages; experiment.json preserved", flush=True)
    experiment.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage-oriented HarnessFix prototype for Better Harness/OpenHands candidates.

This driver keeps the existing HarnessFix stages explicit.  It does not add a
new task-state observer or candidate acceptance gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from task_agent.openhands_agent.model_config_bridge import load_model, selected_model_kwargs


REPO_ROOT = Path(__file__).resolve().parent
FAILURE_ANALYSIS_DIR = REPO_ROOT / "failure_analysis"
MODIFY_CONFIG = REPO_ROOT / "enhancement_implementation" / "config_openhands.yaml"


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _modify_candidate(
    *,
    base_dir: Path,
    target_dir: Path,
    plan_path: Path,
    spec_path: Path,
    model_name: str,
    redo_feedback: Path | None,
) -> None:
    base_dir = base_dir.resolve()
    target_dir = target_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite candidate directory: {target_dir}")
    shutil.copytree(base_dir, target_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    config = yaml.safe_load(MODIFY_CONFIG.read_text(encoding="utf-8"))

    sys.path.insert(0, str(REPO_ROOT / "task_agent" / "mini-swe-agent" / "src"))
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

    agent_config = dict(config.get("agent", {}))
    for key in ("step_limit", "cost_limit"):
        if key in config:
            agent_config.setdefault(key, config[key])
    model_config = config.get("model", {})
    environment_config = config.get("environment", {})
    model = LitellmTextbasedModel(
        model_name=model_name,
        observation_template=model_config.get("observation_template", ""),
        format_error_template=model_config.get("format_error_template", ""),
        action_regex=model_config.get("action_regex", ""),
        model_kwargs=model_config.get("model_kwargs", {}) | selected_model_kwargs(),
        cost_tracking="ignore_errors",
    )
    result_dir = REPO_ROOT / "enhancement_implementation" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"modify_openhands_{target_dir.name}.traj.json"
    env_vars = {
        **environment_config.get("env", {}),
        "TARGET_DIR": str(target_dir),
        "ORIGINAL_DIR": str(base_dir),
        "PLAN_PATH": str(plan_path.resolve()),
        "PLAN_JSON_PATH": str(spec_path.resolve()),
        "REDO_FEEDBACK_PATH": str(redo_feedback.resolve()) if redo_feedback else "",
    }
    agent = DefaultAgent(
        model,
        LocalEnvironment(env=env_vars),
        output_path=output_path,
        **agent_config,
    )
    result = agent.run(
        target_dir=str(target_dir),
        original_dir=str(base_dir),
        plan_path=str(plan_path.resolve()),
        plan_json_path=str(spec_path.resolve()),
    )
    print(json.dumps({"exit_status": result.get("exit_status"), "trajectory": str(output_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="HarnessFix stages for Better Harness/OpenHands")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--traces-dir", type=Path, required=True)
    analyze.add_argument("--eval-results", type=Path, required=True)
    analyze.add_argument("--agent-source-dir", type=Path, required=True)
    analyze.add_argument("--output-file", type=Path, required=True)
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--better-root", type=Path, required=True)
    analyze.add_argument("--workers", type=int, default=1)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--results-file", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--spec-output", type=Path, required=True)
    aggregate.add_argument("--model", required=True)
    aggregate.add_argument("--better-root", type=Path, required=True)
    aggregate.add_argument("--val-analyses", type=Path)
    aggregate.add_argument("--prev-plan", type=Path)
    aggregate.add_argument("--prev-iteration-report", type=Path)
    aggregate.add_argument("--force", action="store_true")

    modify = subparsers.add_parser("modify")
    modify.add_argument("--base-dir", type=Path, required=True)
    modify.add_argument("--target-dir", type=Path, required=True)
    modify.add_argument("--plan", type=Path, required=True)
    modify.add_argument("--spec", type=Path, required=True)
    modify.add_argument("--model", required=True)
    modify.add_argument("--better-root", type=Path, required=True)
    modify.add_argument("--redo-feedback", type=Path)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--base-dir", type=Path, required=True)
    audit.add_argument("--candidate-dir", type=Path, required=True)
    audit.add_argument("--spec", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--baseline-traces", type=Path, required=True)
    gate.add_argument("--baseline-eval", type=Path, required=True)
    gate.add_argument("--current-traces", type=Path, required=True)
    gate.add_argument("--current-eval", type=Path, required=True)
    gate.add_argument("--ids-file", type=Path, required=True)
    gate.add_argument("--plan-spec", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.stage in {"analyze", "aggregate", "modify"}:
        os.environ["BETTER_HARNESS_ROOT"] = str(args.better_root.resolve())
        os.environ["HARNESSFIX_MODEL_ALIAS"] = args.model
        args.model = load_model(args.model)["model"]
    if args.stage == "analyze":
        _run([
            sys.executable,
            str(FAILURE_ANALYSIS_DIR / "run_analysis.py"),
            "--mode", "openhands",
            "--traces-dir", str(args.traces_dir),
            "--eval-results", str(args.eval_results),
            "--agent-source-dir", str(args.agent_source_dir),
            "--output-file", str(args.output_file),
            "--model", args.model,
            "--workers", str(args.workers),
        ])
    elif args.stage == "aggregate":
        command = [
            sys.executable,
            str(FAILURE_ANALYSIS_DIR / "aggregate_results.py"),
            "--mode", "openhands",
            "--results-file", str(args.results_file),
            "--output", str(args.output),
            "--spec-output", str(args.spec_output),
            "--model", args.model,
        ]
        if args.force:
            command.append("--force")
        if args.val_analyses:
            command.extend(["--val-analyses", str(args.val_analyses)])
        if args.prev_plan:
            command.extend(["--prev-plan", str(args.prev_plan)])
        if args.prev_iteration_report:
            command.extend(["--prev-iteration-report", str(args.prev_iteration_report)])
        _run(command)
    elif args.stage == "modify":
        _modify_candidate(
            base_dir=args.base_dir,
            target_dir=args.target_dir,
            plan_path=args.plan,
            spec_path=args.spec,
            model_name=args.model,
            redo_feedback=args.redo_feedback,
        )
    elif args.stage == "audit":
        _run([
            sys.executable,
            str(FAILURE_ANALYSIS_DIR / "plan_diff_audit.py"),
            "--mode", "openhands",
            "--original-dir", str(args.base_dir),
            "--candidate-dir", str(args.candidate_dir),
            "--spec", str(args.spec),
            "--output", str(args.output),
        ])
    else:
        _run([
            sys.executable,
            str(FAILURE_ANALYSIS_DIR / "check_val_gate.py"),
            "--mode", "openhands",
            "--baseline-traces", str(args.baseline_traces),
            "--baseline-eval", str(args.baseline_eval),
            "--current-traces", str(args.current_traces),
            "--current-eval", str(args.current_eval),
            "--val-ids", str(args.ids_file),
            "--plan-spec", str(args.plan_spec),
            "--output", str(args.output),
        ])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HarnessFix OpenHands closed loop and individual analysis/repair stages."""

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
from task_agent.openhands_agent.repair_references import (
    REFERENCE_DEFAULTS, REFERENCE_ENV, stage_references, technical_reference_prompt, validate_references,
)


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
    trajectory_output: Path | None = None,
) -> None:
    base_dir = base_dir.resolve()
    target_dir = target_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite candidate directory: {target_dir}")
    shutil.copytree(base_dir, target_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    config = yaml.safe_load(MODIFY_CONFIG.read_text(encoding="utf-8"))

    sys.path.insert(0, str(REPO_ROOT / "task_agent" / "mini-swe-agent" / "src"))
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
    from failure_analysis.prompt_safety import PromptSafeAgent, PromptSafeLocalEnvironment

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
    output_path = trajectory_output or result_dir / f"modify_openhands_{target_dir.name}.traj.json"
    env_vars = {
        **environment_config.get("env", {}),
        "TARGET_DIR": str(target_dir),
        "ORIGINAL_DIR": str(base_dir),
        "PLAN_PATH": str(plan_path.resolve()),
        "PLAN_JSON_PATH": str(spec_path.resolve()),
        "REDO_FEEDBACK_PATH": str(redo_feedback.resolve()) if redo_feedback else "",
    }
    agent = PromptSafeAgent(
        model,
        PromptSafeLocalEnvironment(env=env_vars, observation_max_chars=environment_config.get("observation_max_chars", 16000)),
        output_path=output_path,
        **agent_config,
    )
    result = agent.run(
        target_dir=str(target_dir),
        original_dir=str(base_dir),
        plan_path=str(plan_path.resolve()),
        plan_json_path=str(spec_path.resolve()),
        technical_references=technical_reference_prompt(),
    )
    print(json.dumps({"exit_status": result.get("exit_status"), "trajectory": str(output_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="HarnessFix closed loop and stages for Better Harness/OpenHands")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    run = subparsers.add_parser("run", help="Run the closed loop from one experiment YAML; resume with --run-dir")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--dry-run", action="store_true")

    test = subparsers.add_parser("test", help="Evaluate the selected candidate after a completed repair loop")
    test.add_argument("--run-dir", type=Path, required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--traces-dir", type=Path, required=True)
    analyze.add_argument("--eval-results", type=Path, required=True)
    analyze.add_argument("--agent-source-dir", type=Path, required=True)
    analyze.add_argument("--output-file", type=Path, required=True)
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--better-root", type=Path, required=True)
    analyze.add_argument("--workers", type=int, default=1)
    analyze.add_argument("--step-limit", type=int, help="Override model-call limit per diagnosis")
    analyze.add_argument("--instance-ids-file", type=Path)

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
    aggregate.add_argument("--memory-root", type=Path)

    modify = subparsers.add_parser("modify")
    modify.add_argument("--base-dir", type=Path, required=True)
    modify.add_argument("--target-dir", type=Path, required=True)
    modify.add_argument("--plan", type=Path, required=True)
    modify.add_argument("--spec", type=Path, required=True)
    modify.add_argument("--model", required=True)
    modify.add_argument("--better-root", type=Path, required=True)
    modify.add_argument("--redo-feedback", type=Path)
    modify.add_argument("--trajectory-output", type=Path)

    for stage_parser in (analyze, aggregate, modify):
        stage_parser.add_argument("--reference-dir", type=Path, help="Use a verified SDK/trace reference snapshot")
        stage_parser.add_argument("--no-technical-references", action="store_true", help="Disable supplemental SDK/trace documentation")

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
    if args.stage in {"run", "test"}:
        from task_agent.openhands_agent.pipeline import run_from_config, test_selected

        if args.stage == "run":
            run_from_config(args.config, run_dir=args.run_dir, dry_run=args.dry_run)
        else:
            test_selected(args.run_dir.resolve())
        return
    if args.stage in {"analyze", "aggregate", "modify"}:
        os.environ["BETTER_HARNESS_ROOT"] = str(args.better_root.resolve())
        if args.reference_dir:
            if args.no_technical_references:
                parser.error("--reference-dir cannot be combined with --no-technical-references; use the snapshot's options")
            reference_dir = args.reference_dir.resolve()
            validate_references(reference_dir)
        else:
            options = dict.fromkeys(REFERENCE_DEFAULTS, False) if args.no_technical_references else REFERENCE_DEFAULTS
            destination = {"analyze": "output_file", "aggregate": "output", "modify": "target_dir"}[args.stage]
            output = getattr(args, destination).resolve()
            reference_dir = stage_references(args.better_root.resolve(), output.parent / f"{output.name}.repair_references", options)
        os.environ[REFERENCE_ENV] = str(reference_dir)
        os.environ["HARNESSFIX_MODEL_ALIAS"] = args.model
        args.model = load_model(args.model)["model"]
    if args.stage == "analyze":
        command = [
            sys.executable,
            str(FAILURE_ANALYSIS_DIR / "run_analysis.py"),
            "--mode", "openhands",
            "--traces-dir", str(args.traces_dir),
            "--eval-results", str(args.eval_results),
            "--agent-source-dir", str(args.agent_source_dir),
            "--output-file", str(args.output_file),
            "--model", args.model,
            "--workers", str(args.workers),
        ]
        if args.instance_ids_file:
            command.extend(["--instance-ids-file", str(args.instance_ids_file)])
        if args.step_limit is not None:
            command.extend(["--step-limit", str(args.step_limit)])
        _run(command)
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
        if args.memory_root:
            command.extend(["--memory-root", str(args.memory_root)])
        _run(command)
    elif args.stage == "modify":
        _modify_candidate(
            base_dir=args.base_dir,
            target_dir=args.target_dir,
            plan_path=args.plan,
            spec_path=args.spec,
            model_name=args.model,
            redo_feedback=args.redo_feedback,
            trajectory_output=args.trajectory_output,
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

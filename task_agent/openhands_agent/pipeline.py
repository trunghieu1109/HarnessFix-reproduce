"""Closed-loop OpenHands runner using the same promotion policy as SWE."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml
from dotenv import dotenv_values

from failure_analysis.analysis_records import analysis_record_succeeded
from failure_analysis.harness_memory import build_memory_entry, store_memory_entry
from failure_analysis.loop_policy import compare_runs, decide_promotion
from failure_analysis.openhands_io import _stable_instance_id
from failure_analysis.secret_redaction import redact_secrets
from task_agent.openhands_agent.bridge import SUPPORTED_TASKS, materialize_candidate, run_better_harness
from task_agent.openhands_agent.candidate_check import check_candidate
from task_agent.openhands_agent.repair_references import reference_hashes, reference_options, stage_references


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "run_pipeline_openhands.py"
SOURCE_FILES = (
    "run_pipeline_openhands.py", "task_agent/openhands_agent/pipeline.py",
    "task_agent/openhands_agent/bridge.py", "task_agent/openhands_agent/runtime_logging.py",
    "task_agent/openhands_agent/model_config_bridge.py", "failure_analysis/loop_policy.py",
    "failure_analysis/run_analysis.py", "failure_analysis/aggregate_results.py",
    "failure_analysis/analysis_config_openhands.yaml", "enhancement_implementation/config_openhands.yaml",
    "failure_analysis/openhands_io.py", "failure_analysis/openhands_trace.py",
    "failure_analysis/validation_metrics.py", "failure_analysis/htir.py",
    "failure_analysis/plan_diff_audit.py", "failure_analysis/consolidation.py",
    "failure_analysis/operator_registry.py", "failure_analysis/prompt_safety.py",
    "failure_analysis/secret_redaction.py",
    "task_agent/openhands_agent/repair_references.py",
    "task_agent/openhands_agent/candidate_check.py",
)
POLICY_DEFAULTS = {
    "max_iterations": 3, "max_promotion_failures": 2, "analysis_workers": 1,
    "analysis_step_limit": None,
    "min_improvement": 1, "min_target_metrics": 1,
    "max_error_rate_delta": 0.15, "max_invalid_rate_delta": 0.15,
    "max_cost_ratio": 1.25, "run_test": False,
    "max_candidate_retries": 2, "candidate_check_timeout": 60,
}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _path_digest(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return _digest({str(child.relative_to(path)): _path_digest(child)
                    for child in sorted(path.rglob("*")) if child.is_file()
                    and "__pycache__" not in child.parts and child.suffix != ".pyc"})


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _invoke(command: list[str], accepted_codes: tuple[int, ...] = (0,)) -> None:
    print("$ " + shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode not in accepted_codes:
        raise subprocess.CalledProcessError(result.returncode, command)


class OpenHandsPipeline:
    def __init__(self, config: dict, *, run_dir: Path | None = None):
        settings = dict(dotenv_values(REPO_ROOT / ".env")) | dict(os.environ)
        config = dict(config)
        better_value = str(config["better_root"])
        if better_value.startswith("${") and better_value.endswith("}"):
            better_value = settings[better_value[2:-1]]
        self.better = Path(better_value).expanduser().resolve()
        config["better_root"] = str(self.better)
        config["repair_references"] = reference_options(config.get("repair_references"))
        self.config = config
        self.task = config["task_id"]
        self.id_field = config.get("example_id_field", "id")
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {self.task}")
        unknown = set(config["pipeline"]) - set(POLICY_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown pipeline options: {sorted(unknown)}")
        self.policy = POLICY_DEFAULTS | config["pipeline"]
        if set(config["models"]) != {"task", "analysis"}:
            raise ValueError("Define exactly task and analysis model aliases")
        if set(config["execution"]) != {"n_responses", "agent_batch_size", "eval_batch_size", "success_threshold"}:
            raise ValueError("Execution needs n_responses, agent_batch_size, eval_batch_size and success_threshold")
        for key in ("max_iterations", "max_promotion_failures", "analysis_workers", "candidate_check_timeout"):
            if not isinstance(self.policy[key], int) or self.policy[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        analysis_step_limit = self.policy["analysis_step_limit"]
        if analysis_step_limit is not None and (type(analysis_step_limit) is not int or analysis_step_limit < 1):
            raise ValueError("analysis_step_limit must be a positive integer")
        if type(self.policy["max_candidate_retries"]) is not int or self.policy["max_candidate_retries"] < 0:
            raise ValueError("max_candidate_retries must be a nonnegative integer")
        if type(self.policy["candidate_check_timeout"]) is not int:
            raise ValueError("candidate_check_timeout must be a positive integer")
        for key in ("agent_batch_size", "eval_batch_size"):
            if not isinstance(config["execution"][key], int) or config["execution"][key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        responses = config["execution"]["n_responses"]
        if not isinstance(responses, dict) or set(responses) != {"train", "val", "test"}:
            raise ValueError("n_responses must define exactly train, val and test counts")
        for split, count in responses.items():
            if type(count) is not int or count < 1:
                raise ValueError(f"n_responses.{split} must be a positive integer")
        if self.policy["min_improvement"] < 1:
            raise ValueError("min_improvement must be >= 1, as in the SWE promotion policy")
        self.responses = dict(responses)
        output_root = Path(config["output_root"])
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        label = f"{self.task}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{os.getpid()}"
        self.root = (run_dir or output_root / label).resolve()
        self.run_id = self.root.name
        self.reference_dir = self.root / "runtime" / "repair_references"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_id):
            raise ValueError("Run directory name must contain only letters, digits, _, . or -")
        self.dataset = self._better_path(config["dataset"])
        self.base_config_path = self._better_path(config["base_run_config"])
        data = json.loads(self.dataset.read_text())
        self.rows: dict[str, list[dict]] = {}
        selected_positions: set[int] = set()
        selected_ids: set[str] = set()
        if set(config["splits"]) != {"train", "val", "test"}:
            raise ValueError("Define exactly train, val and test splits")
        for split, bounds in config["splits"].items():
            start, stop = bounds["start"], bounds["stop"]
            if not (isinstance(start, int) and isinstance(stop, int) and 0 <= start < stop <= len(data)):
                raise ValueError(f"Invalid dataset range for {split}: {bounds}")
            positions = set(range(start, stop))
            if positions & selected_positions:
                raise ValueError("Train, validation and test positions must be disjoint")
            rows = data[start:stop]
            ids = [str(row[self.id_field]) for row in rows]
            if len(set(ids)) != len(ids) or selected_ids & set(ids):
                raise ValueError("Selected dataset IDs must be unique across all splits")
            for index, source_id in enumerate(ids):
                _stable_instance_id(self.task, index, 0, source_id)
            selected_positions |= positions
            selected_ids.update(ids)
            self.rows[split] = rows
        if self.task == "webarena":
            selected = [row for rows in self.rows.values() for row in rows]
            if any(row["eval"]["eval_types"] != ["string_match"] for row in selected):
                raise ValueError("The current WebArena evaluator supports only string_match tasks")
            if any("fuzzy_match" in row["eval"]["reference_answers"] for row in selected) and not config.get("eval_model"):
                raise ValueError("WebArena fuzzy_match tasks require an explicit eval_model")
        self.base_config = yaml.safe_load(self.base_config_path.read_text())
        if self.base_config["task_id"] != self.task:
            raise ValueError("Base run config task_id does not match pipeline task_id")
        prompt = self.better / "tasks" / self.task / "prompts" / f"{config['prompt_name']}.md"
        models = yaml.safe_load((self.better / "configs/models.yaml").read_text())["models"]
        aliases = {entry["name"]: entry for entry in models}
        model_aliases = dict(config["models"])
        if config.get("eval_model"):
            model_aliases["evaluation"] = config["eval_model"]
        model_fields = ("model", "api_base", "temperature", "max_input_tokens", "max_output_tokens",
                        "reasoning_effort", "extra_body", "input_cost_per_token", "output_cost_per_token")
        self.model_settings = redact_secrets({
            role: {key: self._resolve_model_value(entry[key], settings) for key in model_fields if key in entry}
            for role, alias in model_aliases.items() for entry in [aliases[alias]]
        })
        self.snapshot = {
            "config": config, "policy": self.policy, "promotion_policy": "swe_shared",
            "model_settings": self.model_settings,
            "split_hashes": {split: _digest(rows) for split, rows in self.rows.items()},
            "split_ids": {split: [str(row[self.id_field]) for row in rows] for split, rows in self.rows.items()},
            "base_run_config_hash": _path_digest(self.base_config_path),
            "task_prompt_hash": _path_digest(prompt),
            "candidate_template_hash": _path_digest(REPO_ROOT / "task_agent/openhands_agent/original"),
            "better_runtime_source_hash": _path_digest(self.better / "src"),
            "repair_reference_hashes": reference_hashes(self.better, config["repair_references"]),
            "pipeline_source_hashes": {
                name: _path_digest(REPO_ROOT / name) for name in SOURCE_FILES
            },
        }
        self.fingerprint = _digest(self.snapshot)
        snapshot_path = self.root / "experiment.json"
        if snapshot_path.exists():
            previous = json.loads(snapshot_path.read_text())
            if previous["fingerprint"] != self.fingerprint:
                raise ValueError("Config, data, model settings or pipeline changed. Use a new --run-dir.")
        elif self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Run directory is not an OpenHands pipeline experiment: {self.root}")

    def _resolve_model_value(self, value: object, settings: dict) -> object:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            better_settings = settings | dict(dotenv_values(self.better / ".env"))
            return better_settings[value[2:-1]]
        return value

    def _better_path(self, value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.better / path).resolve()

    def _candidate(self, version: int) -> Path:
        return self.root / "candidates" / f"h{version}"

    def _run_root(self, split: str, version: int) -> Path:
        return self.root / "runs" / f"{split}_h{version}"

    def _ids(self, split: str) -> list[str]:
        return [_stable_instance_id(self.task, index, rollout, str(row[self.id_field]))
                for index, row in enumerate(self.rows[split]) for rollout in range(self.responses[split])]

    def _stage(self, name: str, outputs: list[Path], action: Callable[[], None]) -> None:
        # Children reload Better's model registry. Detect edits between stages,
        # so a long experiment cannot silently compare different parameters.
        OpenHandsPipeline(self.config, run_dir=self.root)
        marker = self.root / "stages" / f"{name}.json"
        if marker.exists():
            cached = json.loads(marker.read_text())
            current = {str(path): _path_digest(path) for path in outputs if path.exists()}
            if current != cached["outputs"]:
                raise ValueError(f"Completed stage artifacts changed: {name}")
            print(f"[pipeline] Resume {name}", flush=True)
            return
        print(f"[pipeline] {name}", flush=True)
        action()
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Stage {name} did not produce: {missing}")
        _write_json(marker, {"outputs": {str(path): _path_digest(path) for path in outputs}})

    def _prepare(self) -> None:
        os.environ["BETTER_HARNESS_ROOT"] = str(self.better)
        stage_references(self.better, self.reference_dir, self.config["repair_references"])
        if not (self.root / "experiment.json").exists():
            _write_json(self.root / "experiment.json", self.snapshot | {"fingerprint": self.fingerprint})
            commits = {}
            for label, directory in (("harnessfix", REPO_ROOT), ("better_harness", self.better),
                                     ("openhands_sdk", self.better / "software-agent-sdk")):
                result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, check=False)
                commits[label] = result.stdout.strip() if result.returncode == 0 else None
            _write_json(self.root / "provenance.json", {"commits": commits,
                        "task_images": {key: self.base_config[key] for key in ("server_image", "codex_server_image")
                                        if key in self.base_config}})
        for split, rows in self.rows.items():
            data_path = self.root / "data" / f"{split}.json"
            config_path = self.root / "configs" / f"run_{split}.yaml"
            base = dict(self.base_config)
            for key in ("agent_file", "eval_lm", "rollout_version", "start_index"):
                base.pop(key, None)
            if self.config.get("eval_model"):
                base["eval_lm"] = self.config["eval_model"]
            base.update({key: value for key, value in self.config["execution"].items()
                         if key not in ("success_threshold", "n_responses")})
            base.update(model_name=self.config["models"]["task"], prompt_name=self.config["prompt_name"],
                        data_path=str(data_path), max_examples=len(rows), n_responses=self.responses[split], resume=True)
            for path, content in (
                (data_path, json.dumps(rows, indent=2) + "\n"),
                (config_path, yaml.safe_dump(base, sort_keys=False)),
                (self.root / "data" / f"{split}_ids.txt", "\n".join(self._ids(split)) + "\n"),
            ):
                if path.exists():
                    if path.read_text() != content:
                        raise ValueError(f"Experiment input changed: {path}")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)

    def _archive_partial_candidate(self, target: Path) -> None:
        if target.exists():
            archive = self.root / "incomplete_candidates" / f"{target.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            target.rename(archive)
            print(f"[pipeline] Archived interrupted candidate: {archive}", flush=True)

    def _validate_candidate(self, version: int) -> None:
        name = "materialize_h0" if version == 0 else f"modify_v{version}"
        marker = json.loads((self.root / "stages" / f"{name}.json").read_text())
        if any(not Path(path).exists() or _path_digest(Path(path)) != digest
               for path, digest in marker["outputs"].items()):
            raise ValueError(f"Candidate artifacts changed after completion: h{version}")

    def _validate_run(self, split: str, version: int) -> dict:
        root = self._run_root(split, version)
        result = json.loads((root / "results.json").read_text())
        expected = set(self._ids(split))
        if len(result["all_ids"]) != len(expected) or set(result["all_ids"]) != expected:
            raise ValueError(f"Missing, duplicate or unexpected evaluated rollouts: {split}/h{version}")
        if result.get("unsupported_ids"):
            raise ValueError(f"Evaluation is incomplete: {split}/h{version}")
        categories = [set(result[key]) for key in ("resolved_ids", "unresolved_ids", "error_ids")]
        if set.union(*categories) != expected or sum(map(len, categories)) != len(expected):
            raise ValueError(f"Evaluation outcome categories are incomplete or overlap: {split}/h{version}")
        for instance_id in expected:
            manifest = json.loads((root / "traces" / instance_id / "manifest.json").read_text())
            paths = [manifest.get(key) for key in ("trace_path", "raw_trace_path")]
            if not any(path and Path(path).is_file() for path in paths):
                raise FileNotFoundError(f"Missing trace for {instance_id}")
        checksums_path = root / "trace_checksums.json"
        if checksums_path.exists():
            for path, digest in json.loads(checksums_path.read_text()).items():
                if not Path(path).is_file() or _path_digest(Path(path)) != digest:
                    raise ValueError(f"Recorded trace changed: {path}")
        return result

    def _execute(self, split: str, version: int) -> dict:
        self._validate_candidate(version)
        if split == "test":
            selection = json.loads((self.root / "selection.json").read_text())
            if not selection["closed_loop_complete"] or version != selection["selected_version"]:
                raise ValueError("Test is available only for the final selected candidate")
        root = self._run_root(split, version)

        def action() -> None:
            run_better_harness(
                better_root=self.better, base_config=self.root / "configs" / f"run_{split}.yaml",
                candidate_dir=self._candidate(version), rollout_version=f"{self.run_id}_h{version}_{split}",
                normalized_output=root, success_threshold=self.config["execution"]["success_threshold"],
                example_ids=[str(row[self.id_field]) for row in self.rows[split]],
            )
            self._validate_run(split, version)
            evidence = set()
            for manifest_path in (root / "traces").glob("*/manifest.json"):
                manifest = json.loads(manifest_path.read_text())
                evidence.update(Path(manifest[key]) for key in ("trace_path", "raw_trace_path") if manifest.get(key))
            _write_json(root / "trace_checksums.json", {str(path): _path_digest(path) for path in sorted(evidence)})

        self._stage(f"{split}_h{version}", [root / "results.json", root / "traces", root / "manifest_index.json",
                    root / "trace_checksums.json"], action)
        return self._validate_run(split, version)

    def _analysis(self, split: str, version: int, expected: list[str], *, iteration: int) -> Path:
        name = f"{self.run_id}_{split}_h{version}_v{iteration}_analysis"
        output = self.root / "analysis" / f"{name}.jsonl"
        ids_path = output.with_suffix(".ids.txt")
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_content = "\n".join(expected) + "\n"
        if ids_path.exists() and ids_path.read_text() != ids_content:
            raise ValueError(f"Analysis instance selection changed: {ids_path}")
        if not ids_path.exists():
            ids_path.write_text(ids_content)

        def action() -> None:
            command = [
                sys.executable, "-B", str(DRIVER), "analyze", "--better-root", str(self.better),
                "--reference-dir", str(self.reference_dir),
                "--model", self.config["models"]["analysis"], "--workers", str(self.policy["analysis_workers"]),
                "--traces-dir", str(self._run_root(split, version) / "traces"),
                "--eval-results", str(self._run_root(split, version) / "results.json"),
                "--agent-source-dir", str(self._candidate(version)), "--output-file", str(output),
                "--instance-ids-file", str(ids_path),
            ]
            if self.policy["analysis_step_limit"] is not None:
                command += ["--step-limit", str(self.policy["analysis_step_limit"])]
            _invoke(command)
            records = {record["instance_id"]: record for record in
                       (json.loads(line) for line in output.read_text().splitlines() if line.strip())}
            complete = {key for key, record in records.items() if analysis_record_succeeded(record)}
            if complete != set(expected):
                raise ValueError(f"Analysis incomplete or includes unexpected samples: {name}")

        self._stage(name, [output, ids_path], action)
        return output

    def _comparison(self, split: str, base: int, candidate: int, spec: Path) -> dict:
        return compare_runs(
            mode="openhands", split=split, ids_file=self.root / "data" / f"{split}_ids.txt",
            baseline_eval=self._run_root(split, base) / "results.json",
            current_eval=self._run_root(split, candidate) / "results.json",
            baseline_traces=self._run_root(split, base) / "traces",
            current_traces=self._run_root(split, candidate) / "traces",
            spec_path=spec, max_cost_ratio=self.policy["max_cost_ratio"],
        ) | {"reported_only": split == "train"}

    def _prepare_candidate(self, version: int, base: int, plan: Path, spec: Path) -> None:
        """Retry generation with identical inputs on import/build failure."""
        directory = plan.parent
        history = []
        target = self._candidate(version)
        for attempt in range(self.policy["max_candidate_retries"] + 1):
            attempt_dir = directory / "attempts" / str(attempt)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            archived_candidate = attempt_dir / f"h{version}"
            trajectory = attempt_dir / "modify.traj.json"

            def modify() -> None:
                self._archive_partial_candidate(target)
                self._archive_partial_candidate(archived_candidate)
                command = [sys.executable, "-B", str(DRIVER), "modify", "--better-root", str(self.better),
                           "--reference-dir", str(self.reference_dir), "--model", self.config["models"]["analysis"],
                           "--base-dir", str(self._candidate(base)), "--target-dir", str(target),
                           "--plan", str(plan), "--spec", str(spec), "--trajectory-output", str(trajectory)]
                _invoke(command)
                target.rename(archived_candidate)

            self._stage(f"modify_v{version}_attempt{attempt}", [archived_candidate, trajectory], modify)
            self._validate_candidate(base)
            audit_path = attempt_dir / "audit.json"
            self._stage(f"audit_v{version}_attempt{attempt}", [audit_path], lambda: _invoke([
                sys.executable, "-B", str(REPO_ROOT / "failure_analysis/plan_diff_audit.py"), "--mode", "openhands",
                "--original-dir", str(self._candidate(base)), "--candidate-dir", str(archived_candidate),
                "--spec", str(spec), "--output", str(audit_path),
            ], accepted_codes=(0, 1)))
            audit = json.loads(audit_path.read_text())
            entry = {"attempt": attempt, "candidate": str(archived_candidate.relative_to(self.root)),
                     "audit": str(audit_path.relative_to(self.root)), "audit_passed": audit["passed"]}
            if not audit["passed"]:
                history.append(entry)
                break  # Audit rejection retains the existing outer-loop policy.
            check_path = attempt_dir / "candidate_check.json"
            self._stage(f"check_v{version}_attempt{attempt}", [check_path], lambda: check_candidate(
                better_root=self.better, candidate_dir=archived_candidate, output_path=check_path,
                timeout=self.policy["candidate_check_timeout"], model_settings=self.model_settings["task"],
            ))
            check = json.loads(check_path.read_text())
            entry.update(check=check, check_path=str(check_path.relative_to(self.root)))
            history.append(entry)
            if check["passed"]:
                break
            if attempt < self.policy["max_candidate_retries"]:
                print(f"[pipeline] h{version}: candidate initialization failed; retrying generation "
                      f"with the same context ({attempt + 1}/{self.policy['max_candidate_retries']})", flush=True)
        self._archive_partial_candidate(self._candidate(version))
        shutil.copytree(archived_candidate, self._candidate(version))
        shutil.copyfile(trajectory, directory / "modify.traj.json")
        _write_json(directory / "candidate_checks.json", {
            "attempts": history, "final_attempt": attempt, "audit_path": str(audit_path.relative_to(self.root)),
            "passed": bool(audit["passed"] and history[-1].get("check", {}).get("passed")),
        })

    @staticmethod
    def _candidate_diff(before_root: Path, after_root: Path, changed_files: list[str]) -> dict:
        diffs = {}
        for relative in changed_files:
            before, after = before_root / relative, after_root / relative
            lines = list(difflib.unified_diff(
                before.read_text().splitlines() if before.is_file() else [],
                after.read_text().splitlines() if after.is_file() else [],
                fromfile=f"{before_root.name}/{relative}", tofile=f"{after_root.name}/{relative}", lineterm=""))
            diffs[relative] = {"diff_line_count": len(lines), "diff": "\n".join(lines), "truncated": False}
        return diffs

    def _iteration(self, version: int, base: int, train: dict, previous: Path | None,
                   val_analyses: Path | None) -> tuple[dict, Path, Path | None]:
        directory = self.root / "iterations" / f"v{version}"
        directory.mkdir(parents=True, exist_ok=True)
        plan, spec = directory / "plan.md", directory / "plan.json"
        analysis = self._analysis("train", base, sorted(set(train["all_ids"]) - set(train["resolved_ids"])), iteration=version)
        command = [sys.executable, "-B", str(DRIVER), "aggregate", "--better-root", str(self.better),
                   "--reference-dir", str(self.reference_dir),
                   "--model", self.config["models"]["analysis"], "--results-file", str(analysis),
                   "--output", str(plan), "--spec-output", str(spec), "--memory-root", str(self.root / "memory")]
        if previous:
            command += ["--prev-plan", str(previous.parent / "plan.md"), "--prev-iteration-report", str(previous)]
        if val_analyses:
            command += ["--val-analyses", str(val_analyses)]
        self._stage(f"aggregate_v{version}", [plan, spec], lambda: _invoke(command))

        checks_path = directory / "candidate_checks.json"
        self._stage(f"modify_v{version}", [self._candidate(version), directory / "modify.traj.json", checks_path],
                    lambda: self._prepare_candidate(version, base, plan, spec))
        self._validate_candidate(base)
        checks = json.loads(checks_path.read_text())
        audit_path = directory / "audit.json"
        self._stage(f"audit_v{version}", [audit_path], lambda: _write_json(
            audit_path, json.loads((self.root / checks["audit_path"]).read_text())))
        audit = json.loads(audit_path.read_text())
        train_compare, val_compare = {}, {}
        if audit["passed"] and checks["passed"]:
            self._execute("train", version)
            self._execute("val", version)
            train_compare = self._comparison("train", base, version, spec)
            val_compare = self._comparison("val", base, version, spec)
            promotion = decide_promotion(audit, train_compare, val_compare, **{
                key: self.policy[key] for key in ("min_improvement", "min_target_metrics",
                                                 "max_error_rate_delta", "max_invalid_rate_delta")
            })
        else:
            promotion = {"passed": False, "promoted": False, "decision": "do_not_promote",
                         "failure_reasons": ["audit_failed" if not audit["passed"] else "candidate_initialization_failed"],
                         "regressed_ids": [], "improved_ids": []}
        report_path = directory / "iteration_report.json"

        def report() -> None:
            diffs = self._candidate_diff(self._candidate(base), self._candidate(version), audit["changed_files"])
            guidance = ["Candidate was promoted. Preserve its validated gains." if promotion["promoted"]
                        else "Candidate was rejected. Avoid repeating its harmful edits."]
            if val_compare.get("regressed_ids"):
                guidance.append("Analyze and repair validation regressions: " + ", ".join(val_compare["regressed_ids"]))
            if promotion["failure_reasons"]:
                guidance.append("Promotion failed because: " + ", ".join(promotion["failure_reasons"]))
            _write_json(directory / "train_compare.json", train_compare)
            _write_json(directory / "val_compare.json", val_compare)
            _write_json(directory / "promotion.json", promotion)
            _write_json(report_path, redact_secrets({
                "version": version, "base_version": base, "candidate_dir": str(self._candidate(version)),
                "base_dir": str(self._candidate(base)), "plan_path": str(plan), "plan_spec_path": str(spec),
                "train_analysis_path": str(analysis), "plan_summary": json.loads(spec.read_text()),
                "audit": audit, "candidate_checks": checks,
                "changed_files": audit["changed_files"], "diff_summary": diffs,
                "train_compare": train_compare, "val_compare": val_compare, "promotion": promotion,
                "next_iteration_guidance": guidance,
            }))

        self._stage(f"report_v{version}", [report_path, directory / "train_compare.json",
                    directory / "val_compare.json", directory / "promotion.json"], report)

        def memory() -> None:
            spec_data = json.loads(spec.read_text())
            entry = build_memory_entry(
                mode="openhands", version=version, outcome="accepted" if promotion["promoted"] else "rejected",
                plan_path=plan, spec=spec_data, summary=spec_data.get("plan_metadata", {}).get("summary", plan.stem),
                changed_files=audit["changed_files"], audit=audit, gate=promotion,
            ) | {"run_id": self.run_id, "task_id": self.task}
            store_memory_entry(self.root / "memory", redact_secrets(entry))

        self._stage(f"memory_v{version}", [], memory)
        regressions = val_compare.get("regressed_ids", [])
        next_val_analyses = self._analysis("val", version, regressions, iteration=version) if regressions else None
        print(f"[pipeline] v{version}: {promotion['decision']}; {promotion['failure_reasons']}", flush=True)
        return promotion, report_path, next_val_analyses

    def run(self, *, dry_run: bool = False) -> dict:
        if dry_run:
            preview = {
                "run_dir": str(self.root), "promotion_policy": "swe_shared", "models": self.model_settings,
                "splits": {split: {"samples": len(rows), "n_responses": self.responses[split],
                                   "rollouts_per_candidate": len(rows) * self.responses[split],
                                   "range": self.config["splits"][split]} for split, rows in self.rows.items()},
                "execution": self.config["execution"], "pipeline": self.policy,
                "stages": ["materialize H0", "baseline val", "current-base train", "failed train analysis",
                           "aggregate with previous report/regressions/memory", "modify", "audit",
                           "candidate import/build check + bounded generation retry with unchanged context",
                           "candidate train + val", "train/val comparison", "promotion", "iteration report + memory",
                           "val regression analysis", "repeat", "select best on val", "held-out test after selection"],
            }
            print(json.dumps(preview, indent=2))
            return preview
        self._prepare()
        selection_path = self.root / "selection.json"
        if selection_path.exists():
            self._stage("selection", [selection_path], lambda: None)
            summary = json.loads(selection_path.read_text())
            if self.policy["run_test"]:
                return self.test()
            return summary

        def materialize() -> None:
            self._archive_partial_candidate(self._candidate(0))
            materialize_candidate(better_root=self.better, task_id=self.task,
                                  prompt_name=self.config["prompt_name"], output_dir=self._candidate(0))

        self._stage("materialize_h0", [self._candidate(0)], materialize)
        baseline = self._execute("val", 0)
        base, failures = 0, 0
        promoted: list[int] = []
        reports: list[str] = []
        previous, val_analyses = None, None
        stop_reason = "max_iterations"
        for version in range(1, self.policy["max_iterations"] + 1):
            train = self._execute("train", base)
            if set(train["all_ids"]) <= set(train["resolved_ids"]):
                stop_reason = "all_train_resolved"
                break
            promotion, previous, val_analyses = self._iteration(version, base, train, previous, val_analyses)
            reports.append(str(previous))
            if promotion["promoted"]:
                base = version
                promoted.append(version)
                failures = 0
            else:
                failures += 1
            if failures >= self.policy["max_promotion_failures"]:
                stop_reason = "max_promotion_failures"
                break
        selected = max([0] + promoted, key=lambda version: len(self._validate_run("val", version)["resolved_ids"]))
        summary = {
            "closed_loop_complete": True, "run_dir": str(self.root), "promotion_policy": "swe_shared",
            "stop_reason": stop_reason, "iterations_completed": len(reports), "promoted_versions": promoted,
            "selected_version": selected, "selected_candidate": str(self._candidate(selected)),
            "baseline_val_resolved": len(baseline["resolved_ids"]),
            "selected_val_resolved": len(self._validate_run("val", selected)["resolved_ids"]),
            "val_rollouts": len(self._ids("val")), "iteration_reports": reports,
            "test_command": shlex.join([sys.executable, "-B", str(DRIVER), "test", "--run-dir", str(self.root)]),
        }
        self._stage("selection", [selection_path], lambda: _write_json(selection_path, summary))
        _write_json(self.root / "summary.json", summary)
        if self.policy["run_test"]:
            summary = self.test()
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    def test(self) -> dict:
        summary_path = self.root / "summary.json"
        selection_path = self.root / "selection.json"
        if not (self.root / "stages/selection.json").is_file():
            raise ValueError("Repair loop must finish before test")
        self._stage("selection", [selection_path], lambda: None)
        summary = json.loads(selection_path.read_text())
        if not summary["closed_loop_complete"]:
            raise ValueError("Repair loop must finish before test")
        self._prepare()
        version = summary["selected_version"]
        self._validate_candidate(version)
        result = self._execute("test", version)
        summary["test"] = {
            "candidate_version": version, "samples": len(self.rows["test"]), "rollouts": len(result["all_ids"]),
            "resolved_rollouts": len(result["resolved_ids"]),
            "resolved_rate": len(result["resolved_ids"]) / len(result["all_ids"]),
            "score_mean": result["score_mean"], "results": str(self._run_root("test", version) / "results.json"),
        }
        _write_json(summary_path, summary)
        return summary


def run_from_config(config_path: Path, *, run_dir: Path | None = None, dry_run: bool = False) -> dict:
    config = yaml.safe_load(config_path.read_text())
    return OpenHandsPipeline(config, run_dir=run_dir).run(dry_run=dry_run)


def test_selected(run_dir: Path) -> dict:
    snapshot = json.loads((run_dir / "experiment.json").read_text())
    result = OpenHandsPipeline(snapshot["config"], run_dir=run_dir).test()
    print(json.dumps(result, indent=2))
    return result

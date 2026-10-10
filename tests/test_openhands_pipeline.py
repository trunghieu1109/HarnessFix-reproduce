from __future__ import annotations

import copy
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

import run_pipeline_openhands as openhands_driver
from failure_analysis.loop_policy import decide_promotion
from failure_analysis.openhands_io import normalize_rollout
from scripts.resume_frozen_openhands import (
    PIPELINE_RELATIVE,
    load_frozen_pipeline,
    verify_completed_stages,
)
from scripts.resume_openhands_planner_update import AGGREGATE_RELATIVE, prepare_resume, record_resume
from task_agent.openhands_agent import pipeline


class OpenHandsPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "harnessfix"
        self.better = self.root / "better"
        self.repo.mkdir()
        self.task = "woocommerce_stock_alert_s2l"
        template = self.repo / "task_agent/openhands_agent/original"
        (template / "prompts").mkdir(parents=True)
        (template / "agent.py").write_text("def build_agent(base_dir, llm):\n    return llm\n")
        (template / "config.json").write_text(json.dumps({"task_id": self.task}))
        (template / "prompts/system.md").write_text("Base task instructions.\n")
        for name in pipeline.SOURCE_FILES:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        (self.better / "src").mkdir(parents=True)
        (self.better / "src/collect.py").write_text("fixture\n")
        docs = self.better / "docs"
        (docs / "sdk_reference_details").mkdir(parents=True)
        for name in ("sdk_reference.md", "read_trajectory.md", "sdk_reference_details/02_agent.md"):
            (docs / name).write_text(f"Reference fixture: {name}\n")
        (docs / "adaptation.md").write_text("Strategy guide must not enter HarnessFix references.\n")
        data_dir = self.better / "data"
        data_dir.mkdir(parents=True)
        self.data = [{"id": f"source_{i}", "seed": i, "prompt": f"Task {i}"} for i in range(6)]
        (data_dir / f"{self.task}.json").write_text(json.dumps(self.data))
        task_dir = self.better / "tasks" / self.task
        (task_dir / "prompts").mkdir(parents=True)
        (task_dir / "prompts/default.md").write_text("Base task instructions.\n")
        (task_dir / "run.yaml").write_text(yaml.safe_dump({
            "task_id": self.task, "agent_file": "old.py", "eval_lm": "unused", "max_time": 600,
            "use_docker": True, "server_image": "stock:latest",
        }))
        (self.better / "configs").mkdir()
        (self.better / "configs/models.yaml").write_text(yaml.safe_dump({"models": [
            {"name": "qwen", "model": "openai/Qwen/test", "temperature": 0.2,
             "max_output_tokens": 8192, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            {"name": "gemini", "model": "openai/gemini", "reasoning_effort": "low", "temperature": 0.2},
        ]}))
        self.config = {
            "better_root": str(self.better), "task_id": self.task, "prompt_name": "default",
            "dataset": f"data/{self.task}.json", "base_run_config": f"tasks/{self.task}/run.yaml",
            "output_root": "artifacts", "models": {"task": "qwen", "analysis": "gemini"},
            "splits": {"test": {"start": 0, "stop": 2}, "train": {"start": 2, "stop": 4}, "val": {"start": 4, "stop": 6}},
            "execution": {"n_responses": {"train": 3, "val": 3, "test": 3},
                          "agent_batch_size": 6, "eval_batch_size": 6, "success_threshold": 1.0},
            "pipeline": {"max_iterations": 2, "run_test": True},
        }
        self.run_dir = self.repo / "artifacts" / "experiment"
        self.calls: list[tuple[str, int]] = []
        self.commands: list[list[str]] = []
        self.rejected_audits: set[int] = set()
        self.analysis_fallback = False
        self.missing_eval = False
        self.unsupported_eval = False
        self.scores = {("val", 0): {0}, ("val", 1): {1, 2, 3}, ("val", 2): {1, 2, 3, 4},
                       ("train", 0): {0}, ("train", 1): {0, 1}, ("train", 2): {0, 1, 2}}
        patches = (
            patch.multiple(pipeline, REPO_ROOT=self.repo, DRIVER=self.repo / "run_pipeline_openhands.py"),
            patch.object(pipeline, "materialize_candidate", side_effect=self.materialize),
            patch.object(pipeline, "run_better_harness", side_effect=self.collect_evaluate),
            patch.object(pipeline, "_invoke", side_effect=self.invoke),
            patch.object(pipeline, "check_candidate", side_effect=self.check_candidate),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def experiment(self, config: dict | None = None) -> pipeline.OpenHandsPipeline:
        return pipeline.OpenHandsPipeline(config or self.config, run_dir=self.run_dir)

    def webarena_config(self) -> dict:
        rows = [{"task_id": index, "prompt": f"Web task {index}",
                 "eval": {"eval_types": ["string_match"], "reference_answers": {"fuzzy_match": ["answer"]}}}
                for index in range(6)]
        (self.better / "data/webarena.json").write_text(json.dumps(rows))
        task_dir = self.better / "tasks/webarena"
        shutil.copytree(self.better / "tasks" / self.task, task_dir)
        base = yaml.safe_load((task_dir / "run.yaml").read_text())
        base["task_id"] = "webarena"
        (task_dir / "run.yaml").write_text(yaml.safe_dump(base))
        return self.config | {"task_id": "webarena", "dataset": "data/webarena.json",
                              "base_run_config": "tasks/webarena/run.yaml",
                              "example_id_field": "task_id", "eval_model": "gemini"}

    def materialize(self, **kwargs) -> None:
        shutil.copytree(self.repo / "task_agent/openhands_agent/original", kwargs["output_dir"])

    def collect_evaluate(self, **kwargs) -> None:
        version = int(kwargs["candidate_dir"].name[1:])
        config = yaml.safe_load(kwargs["base_config"].read_text())
        split = kwargs["base_config"].stem.removeprefix("run_")
        self.calls.append((split, version))
        self.assertEqual(config["n_responses"], self.config["execution"]["n_responses"][split])
        self.assertEqual(config["agent_batch_size"], 6)
        self.assertEqual(config["eval_batch_size"], 6)
        self.assertNotIn("agent_file", config)
        self.assertNotIn("eval_lm", config)
        rows = json.loads(Path(config["data_path"]).read_text())
        self.assertEqual(kwargs["example_ids"], [row["id"] for row in rows])
        if split == "test":
            selection = json.loads((self.run_dir / "selection.json").read_text())
            self.assertTrue(selection["closed_loop_complete"])
            self.assertEqual(selection["selected_version"], version)
        eval_rows = []
        for index, row in enumerate(rows):
            self.assertEqual(row["seed"], int(row["id"].split("_")[1]))
            for rollout in range(config["n_responses"]):
                workspace = self.better / "runs" / f"{split}_h{version}" / f"example{index}_rollout{rollout}"
                workspace.mkdir(parents=True, exist_ok=True)
                log_dir = workspace.parent / f"{workspace.name}_logs"
                log_dir.mkdir(exist_ok=True)
                events = [{"kind": "ActionEvent", "tool_name": "finish", "action": {"kind": "FinishAction"}}]
                (log_dir / "trace_fixture.json").write_text(json.dumps({"events": events, "metrics": {"accumulated_cost": 0.1}}))
                (log_dir / "raw_trace_fixture.json").write_text(json.dumps(events))
                score = float(index * config["n_responses"] + rollout in
                              self.scores.get((split, version), set(range(len(rows) * config["n_responses"]))))
                if self.unsupported_eval and split == "train":
                    score = None
                eval_rows.append({"workspace_dir": str(workspace), "score": score, "feedback": "Fixture evaluation."})
        if self.missing_eval and split == "train":
            eval_rows.pop()
        eval_path = self.better / f"eval_{split}_h{version}.yaml"
        eval_path.write_text(yaml.safe_dump(eval_rows))
        normalize_rollout(eval_results_path=eval_path, better_root=self.better, output_dir=kwargs["normalized_output"],
                          task_id=self.task, example_ids=kwargs["example_ids"])

    def invoke(self, command: list[str], accepted_codes=(0,)) -> None:
        self.commands.append(command)
        def arg(key: str) -> Path:
            return Path(command[command.index(key) + 1])
        if "analyze" in command:
            ids = arg("--instance-ids-file").read_text().split()
            output = arg("--output-file")
            output.parent.mkdir(parents=True, exist_ok=True)
            records = [{"instance_id": value, "failure_category": "unresolved",
                        "agent_source_dir": str(arg("--agent-source-dir")), "_analysis_fallback": self.analysis_fallback} for value in ids]
            output.write_text("".join(json.dumps(record) + "\n" for record in records))
        elif "aggregate" in command:
            arg("--output").write_text("General repair plan.\n")
            arg("--spec-output").write_text(json.dumps({"fixes": [{"target_metrics": ["resolved_rate"],
                                                                       "target_files": ["prompts/system.md"]}]}))
        elif "modify" in command:
            shutil.copytree(arg("--base-dir"), arg("--target-dir"))
            prompt = arg("--target-dir") / "prompts/system.md"
            prompt.write_text(prompt.read_text() + f"Fix {arg('--target-dir').name}.\n")
            arg("--trajectory-output").write_text("{}")
        else:
            version = int(arg("--candidate-dir").name[1:])
            arg("--output").write_text(json.dumps({"passed": version not in self.rejected_audits,
                                                       "changed_files": ["prompts/system.md"], "violations": []}))

    def check_candidate(self, **kwargs) -> dict:
        report = {"passed": True, "status": "complete", "stage": "build_agent", "llm_calls": 0}
        kwargs["output_path"].write_text(json.dumps(report))
        return report

    def run_quietly(self, experiment=None) -> dict:
        with redirect_stdout(io.StringIO()):
            return (experiment or self.experiment()).run()

    def test_analysis_step_limit_reaches_train_and_regression_analysis(self) -> None:
        config = copy.deepcopy(self.config)
        config["pipeline"]["analysis_step_limit"] = 10
        self.run_quietly(self.experiment(config))
        analyses = [command for command in self.commands if "analyze" in command]
        self.assertTrue(any("/train_h" in " ".join(command) for command in analyses))
        self.assertTrue(any("/val_h" in " ".join(command) for command in analyses))
        for command in analyses:
            self.assertEqual(command[command.index("--step-limit") + 1], "10")
        snapshot = json.loads((self.run_dir / "experiment.json").read_text())
        self.assertEqual(snapshot["policy"]["analysis_step_limit"], 10)

    def test_analysis_step_limit_rejects_invalid_values(self) -> None:
        for value in (0, -1, True, 1.5, "10"):
            config = copy.deepcopy(self.config)
            config["pipeline"]["analysis_step_limit"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "analysis_step_limit"):
                self.experiment(config)

    def test_analysis_driver_forwards_step_limit_to_runner(self) -> None:
        argv = [
            "run_pipeline_openhands.py", "analyze", "--better-root", str(self.better),
            "--traces-dir", str(self.root / "traces"), "--eval-results", str(self.root / "results.json"),
            "--agent-source-dir", str(self.root / "h0"), "--output-file", str(self.root / "analysis.jsonl"),
            "--model", "gemini", "--step-limit", "10",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ),
            patch.object(openhands_driver, "load_model", return_value={"model": "offline"}),
            patch.object(openhands_driver, "stage_references", return_value=self.root / "references"),
            patch.object(openhands_driver, "_run") as run,
        ):
            openhands_driver.main()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]).name, "run_analysis.py")
        self.assertEqual(command[command.index("--step-limit") + 1], "10")

    def test_promoted_base_regression_feedback_and_test_selection(self) -> None:
        summary = self.run_quietly()
        self.assertEqual(summary["selected_version"], 2)
        self.assertEqual(summary["promoted_versions"], [1, 2])
        self.assertEqual(summary["test"]["rollouts"], 6)
        self.assertEqual(self.calls, [("val", 0), ("train", 0), ("train", 1), ("val", 1),
                                      ("train", 2), ("val", 2), ("test", 2)])
        modifies = [command for command in self.commands if "modify" in command]
        self.assertEqual(Path(modifies[1][modifies[1].index("--base-dir") + 1]).name, "h1")
        aggregates = [command for command in self.commands if "aggregate" in command]
        for flag in ("--prev-plan", "--prev-iteration-report", "--val-analyses"):
            self.assertIn(flag, aggregates[1])
        val_analysis = [command for command in self.commands if "analyze" in command and "/val_h" in " ".join(command)]
        self.assertEqual(len(val_analysis), 1)
        self.assertEqual(Path(val_analysis[0][val_analysis[0].index("--agent-source-dir") + 1]).name, "h1")
        self.assertFalse(any("test_h" in " ".join(command) for command in self.commands))
        memory = (self.run_dir / "memory/accepted_repairs.jsonl").read_text().splitlines()
        self.assertEqual(len(memory), 2)
        for command in aggregates:
            self.assertEqual(Path(command[command.index("--memory-root") + 1]), self.run_dir / "memory")
        for command in self.commands:
            if any(stage in command for stage in ("analyze", "aggregate", "modify")):
                self.assertEqual(Path(command[command.index("--reference-dir") + 1]),
                                 self.run_dir / "runtime/repair_references")
        references = self.run_dir / "runtime/repair_references"
        self.assertEqual((references / "sdk_reference.md").read_bytes(),
                         (self.better / "docs/sdk_reference.md").read_bytes())
        self.assertFalse((references / "adaptation.md").exists())
        self.assertFalse((self.run_dir / "candidates/h0/skills").exists())
        report = json.loads((self.run_dir / "iterations/v2/iteration_report.json").read_text())
        self.assertEqual(report["base_version"], 1)
        self.assertTrue(report["train_compare"]["reported_only"])

    def test_technical_references_cannot_change_on_resume_but_strategy_guide_is_excluded(self) -> None:
        self.run_quietly()
        (self.better / "docs/adaptation.md").write_text("A different strategy guide")
        self.experiment()  # This document is not an experiment input.
        document = self.better / "docs/sdk_reference.md"
        original = document.read_text()
        document.write_text(original + "Changed interface docs.\n")
        with self.assertRaisesRegex(ValueError, "pipeline changed"):
            self.experiment()
        document.write_text(original)
        (self.run_dir / "runtime/repair_references/read_trajectory.md").write_text("Changed schema docs")
        with self.assertRaisesRegex(ValueError, "Technical reference changed"):
            self.run_quietly()

    def test_reference_options_preserve_the_task_schedule_and_selection_policy(self) -> None:
        config = copy.deepcopy(self.config)
        config["repair_references"] = {"sdk_reference": False}
        summary = self.run_quietly(self.experiment(config))
        self.assertEqual(summary["selected_version"], 2)
        references = self.run_dir / "runtime/repair_references"
        self.assertFalse((references / "sdk_reference.md").exists())
        self.assertTrue((references / "read_trajectory.md").exists())

    def test_audit_failures_never_execute_candidate_and_stop_like_swe(self) -> None:
        self.rejected_audits = {1, 2}
        summary = self.run_quietly()
        self.assertEqual(self.calls, [("val", 0), ("train", 0), ("test", 0)])
        self.assertEqual(summary["stop_reason"], "max_promotion_failures")
        self.assertEqual(summary["selected_version"], 0)
        modifies = [command for command in self.commands if "modify" in command]
        self.assertTrue(all(Path(command[command.index("--base-dir") + 1]).name == "h0" for command in modifies))

    def test_rejected_validation_candidate_is_not_used_as_next_base(self) -> None:
        self.scores[("val", 1)] = {0}
        summary = self.run_quietly()
        self.assertEqual(summary["promoted_versions"], [2])
        modifies = [command for command in self.commands if "modify" in command]
        self.assertEqual(Path(modifies[1][modifies[1].index("--base-dir") + 1]).name, "h0")
        self.assertEqual(self.calls[-1], ("test", 2))

    def test_initialization_failure_retries_same_generation_context_without_feedback(self) -> None:
        config = copy.deepcopy(self.config)
        config["pipeline"].update(max_iterations=1, run_test=False)
        checks = []

        def check(**kwargs):
            checks.append(kwargs["candidate_dir"])
            report = {"passed": len(checks) > 1, "status": "complete", "stage": "build_agent", "llm_calls": 0}
            if not report["passed"]:
                report.update(exception_type="AttributeError", message="browser_get_state",
                              traceback="Synthetic traceback: BrowserToolSet.browser_get_state")
            kwargs["output_path"].write_text(json.dumps(report))
            return report

        with patch.object(pipeline, "check_candidate", side_effect=check):
            summary = self.run_quietly(self.experiment(config))
        modifies = [command for command in self.commands if "modify" in command]
        self.assertEqual(len(modifies), 2)
        self.assertTrue(all(Path(command[command.index("--target-dir") + 1]) == self.run_dir / "candidates/h1"
                            for command in modifies))
        self.assertTrue(all(Path(command[command.index("--base-dir") + 1]).name == "h0" for command in modifies))
        contexts = []
        for command in modifies:
            self.assertNotIn("--redo-feedback", command)
            context = list(command)
            del context[context.index("--trajectory-output"):context.index("--trajectory-output") + 2]
            contexts.append(context)
        self.assertEqual(contexts[0], contexts[1])
        self.assertFalse(list((self.run_dir / "iterations/v1").rglob("redo_feedback.json")))
        self.assertTrue(all(path.exists() for path in checks))
        self.assertEqual(summary["selected_version"], 1)
        self.assertEqual(self.calls, [("val", 0), ("train", 0), ("train", 1), ("val", 1)])
        report = json.loads((self.run_dir / "iterations/v1/iteration_report.json").read_text())
        self.assertEqual(report["candidate_checks"]["final_attempt"], 1)
        self.assertEqual(report["candidate_checks"]["attempts"][0]["check"]["exception_type"], "AttributeError")
        counts = len(self.calls), len(self.commands), len(checks)
        with patch.object(pipeline, "check_candidate", side_effect=AssertionError("Check reran")):
            self.assertEqual(self.run_quietly(self.experiment(config)), summary)
        self.assertEqual((len(self.calls), len(self.commands), len(checks)), counts)

    def test_exhausted_initialization_retries_record_rejection_and_feed_next_iteration(self) -> None:
        def fail_check(**kwargs):
            report = {"passed": False, "status": "complete", "stage": "import", "llm_calls": 0,
                      "exception_type": "ImportError", "message": "Synthetic incompatible SDK import",
                      "traceback": "Synthetic candidate import traceback"}
            kwargs["output_path"].write_text(json.dumps(report))
            return report

        with patch.object(pipeline, "check_candidate", side_effect=fail_check) as check:
            summary = self.run_quietly()
        self.assertEqual(check.call_count, 6)  # Initial attempt + two regenerations, for h1 and h2.
        self.assertTrue(all("--redo-feedback" not in command for command in self.commands if "modify" in command))
        self.assertEqual(self.calls, [("val", 0), ("train", 0), ("test", 0)])
        self.assertEqual(summary["selected_version"], 0)
        self.assertEqual(summary["stop_reason"], "max_promotion_failures")
        report = json.loads((self.run_dir / "iterations/v1/iteration_report.json").read_text())
        self.assertEqual(report["promotion"]["failure_reasons"], ["candidate_initialization_failed"])
        self.assertEqual(len(report["candidate_checks"]["attempts"]), 3)
        aggregates = [command for command in self.commands if "aggregate" in command]
        self.assertIn("--prev-iteration-report", aggregates[1])
        self.assertEqual(len((self.run_dir / "memory/rejected_repairs.jsonl").read_text().splitlines()), 2)

    def test_check_infrastructure_failure_does_not_trigger_candidate_regeneration(self) -> None:
        with patch.object(pipeline, "check_candidate", side_effect=RuntimeError("Missing SDK dependency")):
            with self.assertRaisesRegex(RuntimeError, "Missing SDK"):
                self.run_quietly()
        self.assertEqual(len([command for command in self.commands if "modify" in command]), 1)
        self.assertEqual(self.calls, [("val", 0), ("train", 0)])
        self.assertFalse((self.run_dir / "selection.json").exists())

    def test_retry_limits_validate_and_zero_disables_regeneration(self) -> None:
        for invalid in (-1, True, 1.5):
            config = copy.deepcopy(self.config)
            config["pipeline"]["max_candidate_retries"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "max_candidate_retries"):
                self.experiment(config)
        config = copy.deepcopy(self.config)
        config["pipeline"].update(max_candidate_retries=0, max_iterations=1, run_test=False)
        def fail_check(**kwargs):
            report = {"passed": False, "status": "complete", "stage": "build_agent", "llm_calls": 0}
            kwargs["output_path"].write_text(json.dumps(report))
        with patch.object(pipeline, "check_candidate", side_effect=fail_check):
            summary = self.run_quietly(self.experiment(config))
        self.assertEqual(len([command for command in self.commands if "modify" in command]), 1)
        self.assertEqual(summary["selected_version"], 0)

    def test_resume_interrupted_retry_preserves_previous_attempt_and_context(self) -> None:
        config = copy.deepcopy(self.config)
        config["pipeline"].update(max_iterations=1, run_test=False)
        failed_check_paths = []
        def check(**kwargs):
            report = {"passed": kwargs["candidate_dir"].parent.name != "0", "status": "complete",
                      "stage": "build_agent", "message": "Synthetic initialization failure", "llm_calls": 0}
            failed_check_paths.append(kwargs["output_path"])
            kwargs["output_path"].write_text(json.dumps(report))
        interrupted = False
        def invoke(command, accepted_codes=(0,)):
            nonlocal interrupted
            self.invoke(command, accepted_codes)
            if "modify" in command and Path(command[command.index("--trajectory-output") + 1]).parent.name == "1" \
                    and not interrupted:
                interrupted = True
                raise RuntimeError("Interrupted during regeneration")
        with patch.object(pipeline, "check_candidate", side_effect=check):
            with patch.object(pipeline, "_invoke", side_effect=invoke):
                with self.assertRaisesRegex(RuntimeError, "Interrupted during regeneration"):
                    self.run_quietly(self.experiment(config))
            first_check = self.run_dir / "iterations/v1/attempts/0/candidate_check.json"
            previous = first_check.read_bytes()
            self.assertTrue((self.run_dir / "candidates/h1").exists())
            self.run_quietly(self.experiment(config))
        self.assertEqual(first_check.read_bytes(), previous)
        self.assertTrue(all("--redo-feedback" not in command for command in self.commands if "modify" in command))
        self.assertEqual(len(failed_check_paths), 2)
        self.assertEqual(self.calls.count(("train", 0)), 1)
        self.assertEqual(len(list((self.run_dir / "incomplete_candidates").iterdir())), 1)

    def test_resume_after_attempt_archive_without_completion_marker_retries_from_base(self) -> None:
        config = copy.deepcopy(self.config)
        config["pipeline"].update(max_iterations=1, run_test=False)
        write_json = pipeline._write_json

        def interrupted(path, value):
            if path.name == "modify_v1_attempt0.json":
                raise RuntimeError("Interrupted before completion marker")
            write_json(path, value)

        with patch.object(pipeline, "_write_json", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "Interrupted before completion marker"):
                self.run_quietly(self.experiment(config))
        self.assertTrue((self.run_dir / "iterations/v1/attempts/0/h1").exists())
        self.assertFalse((self.run_dir / "candidates/h1").exists())
        self.run_quietly(self.experiment(config))
        self.assertEqual(len(list((self.run_dir / "incomplete_candidates").iterdir())), 1)
        self.assertEqual(self.calls.count(("train", 0)), 1)
        prompt = (self.run_dir / "candidates/h1/prompts/system.md").read_text()
        self.assertEqual(prompt.count("Fix h1"), 1)

    def test_resume_after_test_does_not_rerun_or_repair_on_held_out_feedback(self) -> None:
        first = self.run_quietly()
        counts = len(self.calls), len(self.commands)
        second = self.run_quietly()
        self.assertEqual(counts, (len(self.calls), len(self.commands)))
        self.assertEqual(first, second)
        summary_path = self.run_dir / "summary.json"
        summary_path.write_text(json.dumps({"selected_version": 999}))
        self.assertEqual(self.run_quietly(), first)

    def test_resume_from_interrupted_analysis_reuses_completed_rollouts(self) -> None:
        self.analysis_fallback = True
        with self.assertRaisesRegex(ValueError, "Analysis incomplete"):
            self.run_quietly()
        self.assertFalse((self.run_dir / "selection.json").exists())
        self.analysis_fallback = False
        self.run_quietly()
        self.assertEqual(self.calls.count(("train", 0)), 1)
        self.assertEqual(self.calls.count(("val", 0)), 1)

    def test_incomplete_evaluation_never_reaches_analysis_or_test(self) -> None:
        for unsupported in (False, True):
            with self.subTest(unsupported=unsupported):
                self.run_dir = self.repo / "artifacts" / f"incomplete_{unsupported}"
                self.missing_eval = not unsupported
                self.unsupported_eval = unsupported
                with self.assertRaisesRegex(ValueError, "Evaluation is incomplete|evaluated rollouts"):
                    self.run_quietly()
                self.assertFalse(self.commands)
                self.assertFalse((self.run_dir / "selection.json").exists())

    def test_all_train_passed_keeps_h0_and_runs_test_once(self) -> None:
        self.scores[("train", 0)] = set(range(6))
        summary = self.run_quietly()
        self.assertEqual(summary["stop_reason"], "all_train_resolved")
        self.assertEqual(summary["selected_version"], 0)
        self.assertFalse(self.commands)

    def test_split_ids_are_disjoint_even_when_local_example_numbers_match(self) -> None:
        experiment = self.experiment()
        ids = [set(experiment._ids(split)) for split in ("train", "val", "test")]
        self.assertEqual(len(set.union(*ids)), 18)
        self.assertTrue(all(len(item) == 6 for item in ids))

    def test_webarena_native_ids_and_judge_reach_generated_configs_and_bridge(self) -> None:
        experiment = self.experiment(self.webarena_config())
        experiment._prepare()
        ids = [set(experiment._ids(split)) for split in ("train", "val", "test")]
        self.assertEqual(len(set.union(*ids)), 18)
        self.assertEqual(experiment.snapshot["split_ids"], {"test": ["0", "1"], "train": ["2", "3"], "val": ["4", "5"]})
        for split in ("train", "val", "test"):
            config = yaml.safe_load((self.run_dir / "configs" / f"run_{split}.yaml").read_text())
            self.assertEqual(config["eval_lm"], "gemini")
            rows = json.loads(Path(config["data_path"]).read_text())
            self.assertEqual(rows, experiment.rows[split])
            self.assertTrue(all("id" not in row for row in rows))
        with patch.object(experiment, "_validate_candidate"), \
                patch.object(experiment, "_validate_run", return_value={}), \
                patch.object(experiment, "_stage", side_effect=lambda name, outputs, action: action()), \
                patch.object(pipeline, "run_better_harness") as bridge:
            experiment._execute("train", 0)
        self.assertEqual(bridge.call_args.kwargs["example_ids"], ["2", "3"])

    def test_webarena_fuzzy_scoring_cannot_run_without_judge(self) -> None:
        config = self.webarena_config()
        config.pop("eval_model")
        with self.assertRaisesRegex(ValueError, "fuzzy_match.*eval_model"):
            self.experiment(config)
        self.assertFalse(self.run_dir.exists())

    def test_webarena_unsupported_evaluation_fails_before_collecting(self) -> None:
        config = self.webarena_config()
        path = self.better / "data/webarena.json"
        rows = json.loads(path.read_text())
        rows[2]["eval"]["eval_types"] = ["program_html"]
        path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "supports only string_match"):
            self.experiment(config)
        self.assertFalse(self.calls)
        self.assertFalse(self.run_dir.exists())

    def test_separate_evaluation_model_settings_are_recorded_and_locked(self) -> None:
        config = self.webarena_config() | {"eval_model": "judge"}
        path = self.better / "configs/models.yaml"
        models = yaml.safe_load(path.read_text())
        models["models"].append({"name": "judge", "model": "openai/judge", "temperature": 0.2})
        path.write_text(yaml.safe_dump(models))
        experiment = self.experiment(config)
        self.assertEqual(experiment.model_settings["evaluation"], {"model": "openai/judge", "temperature": 0.2})
        experiment._prepare()
        models["models"][-1]["temperature"] = 0.9
        path.write_text(yaml.safe_dump(models))
        with self.assertRaisesRegex(ValueError, "model settings or pipeline changed"):
            self.experiment(config)

    def completed_frozen_experiment(self):
        original = Path(pipeline.__file__).read_bytes()
        live_path = self.repo / PIPELINE_RELATIVE
        live_path.write_bytes(original)
        summary = self.run_quietly()
        frozen_path = self.run_dir / "runtime/pipeline_original.py"
        frozen_path.parent.mkdir(exist_ok=True)
        frozen_path.write_bytes(original)
        live_path.write_text("raise AssertionError('Updated source must not execute')\n")
        module, snapshot = load_frozen_pipeline(self.repo, self.run_dir)
        return module, snapshot, summary

    def test_frozen_pipeline_resumes_cached_work_with_original_code(self) -> None:
        module, snapshot, expected = self.completed_frozen_experiment()
        experiment = module.OpenHandsPipeline(snapshot["config"], run_dir=self.run_dir)
        self.assertGreater(verify_completed_stages(module, self.run_dir), 0)
        with patch.object(module, "materialize_candidate", side_effect=AssertionError("H0 reran")), \
                patch.object(module, "run_better_harness", side_effect=AssertionError("Benchmark reran")), \
                patch.object(module, "_invoke", side_effect=AssertionError("Model reran")):
            self.assertEqual(self.run_quietly(experiment), expected)
        self.assertEqual(json.loads((self.run_dir / "experiment.json").read_text()), snapshot)

    def test_frozen_pipeline_rejects_an_unverified_archive(self) -> None:
        self.completed_frozen_experiment()
        path = self.run_dir / "runtime/pipeline_original.py"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "does not match.*SHA256"):
            load_frozen_pipeline(self.repo, self.run_dir)

    def test_frozen_recovery_preserves_model_and_artifact_guards(self) -> None:
        module, snapshot, _ = self.completed_frozen_experiment()
        path = self.better / "configs/models.yaml"
        original = path.read_text()
        models = yaml.safe_load(original)
        models["models"][0]["temperature"] = 0.9
        path.write_text(yaml.safe_dump(models))
        with self.assertRaisesRegex(ValueError, "model settings or pipeline changed"):
            module.OpenHandsPipeline(snapshot["config"], run_dir=self.run_dir)
        path.write_text(original)
        prompt = self.run_dir / "candidates/h0/prompts/system.md"
        prompt.write_text(prompt.read_text() + "Changed after completion.\n")
        with self.assertRaisesRegex(ValueError, "Completed stage artifact changed"):
            verify_completed_stages(module, self.run_dir)

    def interrupted_planner_update(self):
        (self.repo / PIPELINE_RELATIVE).write_bytes(Path(pipeline.__file__).read_bytes())

        def stop_at_second_aggregate(command, accepted_codes=(0,)):
            if "aggregate" in command:
                output = Path(command[command.index("--output") + 1])
                if output.parent.name == "v2":
                    output.with_suffix(".md.raw.txt").write_text("Original invalid plan")
                    output.with_suffix(".md.aggregate_agent.traj.json").write_text("{}")
                    raise RuntimeError("Interrupted aggregate")
            self.invoke(command, accepted_codes)

        with patch.object(pipeline, "_invoke", side_effect=stop_at_second_aggregate):
            with self.assertRaisesRegex(RuntimeError, "Interrupted aggregate"):
                self.run_quietly()
        (self.repo / AGGREGATE_RELATIVE).write_text("updated planner fixture\n")

    def test_planner_update_resume_reuses_completed_stages_and_records_adoption(self):
        self.interrupted_planner_update()
        original_snapshot = (self.run_dir / "experiment.json").read_bytes()
        module, experiment, record = prepare_resume(self.repo, self.run_dir)
        self.assertEqual(record["pending_stage"], "aggregate_v2")
        self.assertEqual(set(record["source_updates"]), {AGGREGATE_RELATIVE})
        archive = record_resume(self.repo, self.run_dir, record)
        self.assertEqual((archive / "plan.md.raw.txt").read_text(), "Original invalid plan")
        self.calls.clear()
        self.commands.clear()
        with patch.object(module, "_invoke", side_effect=self.invoke), \
                patch.object(module, "run_better_harness", side_effect=self.collect_evaluate), \
                patch.object(module, "check_candidate", side_effect=self.check_candidate):
            summary = self.run_quietly(experiment)
        self.assertEqual(self.calls, [("train", 2), ("val", 2), ("test", 2)])
        self.assertTrue(summary["closed_loop_complete"])
        self.assertFalse(any("analyze" in command for command in self.commands))
        self.assertEqual((self.run_dir / "experiment.json").read_bytes(), original_snapshot)

    def test_planner_update_resume_rejects_other_code_changes(self):
        self.interrupted_planner_update()
        (self.repo / "run_pipeline_openhands.py").write_text("another update\n")
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            prepare_resume(self.repo, self.run_dir)

    def test_planner_update_resume_preserves_model_and_artifact_guards(self):
        self.interrupted_planner_update()
        path = self.better / "configs/models.yaml"
        original = path.read_bytes()
        models = yaml.safe_load(original)
        models["models"][0]["temperature"] = 0.9
        path.write_text(yaml.safe_dump(models))
        with self.assertRaisesRegex(ValueError, "model settings or pipeline changed"):
            prepare_resume(self.repo, self.run_dir)
        path.write_bytes(original)
        prompt = self.run_dir / "candidates/h1/prompts/system.md"
        prompt.write_text(prompt.read_text() + "Modified after completion\n")
        with self.assertRaisesRegex(ValueError, "Completed stage artifact changed"):
            prepare_resume(self.repo, self.run_dir)

    def test_planner_update_resume_rejects_changes_after_verification(self):
        self.interrupted_planner_update()
        module, experiment, record = prepare_resume(self.repo, self.run_dir)
        (self.repo / AGGREGATE_RELATIVE).write_text("another planner revision\n")
        with self.assertRaisesRegex(ValueError, "source changed again"):
            self.run_quietly(experiment)
        with self.assertRaisesRegex(ValueError, "source changed before"):
            record_resume(self.repo, self.run_dir, record)

    def test_overlapping_splits_and_duplicate_source_ids_are_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["splits"]["train"] = {"start": 1, "stop": 3}
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.experiment(config)
        rows = copy.deepcopy(self.data)
        rows[4]["id"] = rows[0]["id"]
        (self.better / "data" / f"{self.task}.json").write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.experiment()

    def test_dry_run_creates_no_artifacts_and_calls_no_model_or_benchmark(self) -> None:
        with redirect_stdout(io.StringIO()):
            preview = self.experiment().run(dry_run=True)
        self.assertEqual(preview["splits"]["train"]["rollouts_per_candidate"], 6)
        self.assertFalse(self.run_dir.exists())
        self.assertFalse(self.commands)
        self.assertFalse(self.calls)

    def test_stock_optimization_runs_once_per_sample_and_final_test_three_times(self) -> None:
        self.data = [{"id": f"source_{i}", "seed": i, "prompt": f"Task {i}"} for i in range(50)]
        (self.better / "data" / f"{self.task}.json").write_text(json.dumps(self.data))
        self.config["splits"] = {"test": {"start": 0, "stop": 30},
                                 "train": {"start": 30, "stop": 40}, "val": {"start": 40, "stop": 50}}
        self.config["execution"]["n_responses"] = {"train": 1, "val": 1, "test": 3}
        self.config["pipeline"]["max_iterations"] = 1
        self.scores = {("val", 0): set(), ("val", 1): {0}, ("train", 0): set(), ("train", 1): {0}}
        with redirect_stdout(io.StringIO()):
            preview = self.experiment().run(dry_run=True)
        self.assertEqual({split: entry["rollouts_per_candidate"] for split, entry in preview["splits"].items()},
                         {"train": 10, "val": 10, "test": 90})
        summary = self.run_quietly()
        self.assertEqual(summary["selected_version"], 1)
        self.assertEqual(summary["val_rollouts"], 10)
        self.assertEqual(summary["test"]["rollouts"], 90)
        self.assertEqual(self.calls, [("val", 0), ("train", 0), ("train", 1), ("val", 1), ("test", 1)])
        for split, version in self.calls:
            result = json.loads((self.run_dir / f"runs/{split}_h{version}/results.json").read_text())
            self.assertEqual(len(result["all_ids"]), 90 if split == "test" else 10)
            if split != "test":
                self.assertTrue(all(instance_id.endswith("__rollout0") for instance_id in result["all_ids"]))
        train_analysis = [command for command in self.commands
                          if "analyze" in command and "/train_h0" in " ".join(command)]
        self.assertEqual(len(train_analysis), 1)
        command = train_analysis[0]
        ids = Path(command[command.index("--instance-ids-file") + 1]).read_text().split()
        self.assertEqual(len(ids), 10)
        self.assertFalse(any("test_h" in " ".join(command) for command in self.commands))
        previous_counts = len(self.calls), len(self.commands)
        self.assertEqual(self.run_quietly(), summary)
        self.assertEqual((len(self.calls), len(self.commands)), previous_counts)

    def test_response_counts_require_all_splits_and_positive_integers(self) -> None:
        for counts in (3, {"train": 1, "val": 1}, {"train": 0, "val": 1, "test": 3},
                       {"train": 1, "val": 1, "test": True}):
            with self.subTest(counts=counts):
                config = copy.deepcopy(self.config)
                config["execution"]["n_responses"] = counts
                with self.assertRaisesRegex(ValueError, "n_responses"):
                    self.experiment(config)

    def test_model_config_or_selected_candidate_changes_cannot_resume(self) -> None:
        self.run_quietly()
        prompt = self.run_dir / "candidates/h2/prompts/system.md"
        prompt.write_text("Changed after selection")
        with self.assertRaisesRegex(ValueError, "Candidate artifacts changed"):
            self.run_quietly()
        models = self.better / "configs/models.yaml"
        data = yaml.safe_load(models.read_text())
        data["models"][0]["temperature"] = 0.9
        models.write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(ValueError, "model settings or pipeline changed"):
            self.experiment()

    def test_deferred_test_can_run_only_after_selection(self) -> None:
        config = copy.deepcopy(self.config)
        config["pipeline"]["run_test"] = False
        experiment = self.experiment(config)
        with self.assertRaisesRegex(ValueError, "finish before test"):
            experiment.test()
        self.run_quietly(experiment)
        self.assertFalse(any(split == "test" for split, _ in self.calls))
        with redirect_stdout(io.StringIO()):
            result = experiment.test()
        self.assertEqual(result["test"]["candidate_version"], 2)

    def test_interrupted_modify_archives_partial_edits_and_reuses_previous_stages(self) -> None:
        invoke = self.invoke
        failed = False

        def interrupted(command, accepted_codes=(0,)):
            nonlocal failed
            invoke(command, accepted_codes)
            if "modify" in command and not failed:
                failed = True
                raise RuntimeError("Interrupted during modification")

        with patch.object(pipeline, "_invoke", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "Interrupted during modification"):
                self.run_quietly()
        self.assertTrue((self.run_dir / "candidates/h1").exists())
        self.assertFalse((self.run_dir / "stages/modify_v1.json").exists())
        self.run_quietly()
        self.assertEqual(len(list((self.run_dir / "incomplete_candidates").iterdir())), 1)
        self.assertEqual(self.calls.count(("train", 0)), 1)

    def test_changed_raw_trace_cannot_be_reused_for_later_evaluation(self) -> None:
        config = copy.deepcopy(self.config)
        config["pipeline"]["run_test"] = False
        experiment = self.experiment(config)
        self.run_quietly(experiment)
        experiment._validate_run("val", 2)
        manifest = next((self.run_dir / "runs/val_h2/traces").glob("*/manifest.json"))
        trace = Path(json.loads(manifest.read_text())["trace_path"])
        trace.write_text('{"events": []}')
        with self.assertRaisesRegex(ValueError, "Recorded trace changed"):
            experiment._validate_run("val", 2)

    def test_edited_final_selection_cannot_trigger_test_on_another_candidate(self) -> None:
        config = copy.deepcopy(self.config)
        config["pipeline"]["run_test"] = False
        experiment = self.experiment(config)
        self.run_quietly(experiment)
        path = self.run_dir / "selection.json"
        selection = json.loads(path.read_text())
        selection["selected_version"] = 0
        path.write_text(json.dumps(selection))
        with self.assertRaisesRegex(ValueError, "Completed stage artifacts changed"):
            experiment.test()

    def test_model_parameter_edit_between_stages_stops_before_next_model_call(self) -> None:
        original = self.invoke

        def edit_settings(command, accepted_codes=(0,)):
            original(command, accepted_codes)
            if "analyze" in command:
                models = self.better / "configs/models.yaml"
                data = yaml.safe_load(models.read_text())
                data["models"][0]["temperature"] = 0.9
                models.write_text(yaml.safe_dump(data))

        with patch.object(pipeline, "_invoke", side_effect=edit_settings):
            with self.assertRaisesRegex(ValueError, "model settings or pipeline changed"):
                self.run_quietly()
        self.assertFalse(any("aggregate" in command for command in self.commands))
        self.assertFalse((self.run_dir / "selection.json").exists())


class OpenHandsStageCLITests(unittest.TestCase):
    def test_analysis_passes_explicit_ids_to_existing_failure_analyzer(self) -> None:
        import run_pipeline_openhands as driver

        argv = ["run_pipeline_openhands.py", "analyze", "--better-root", "/tmp/better",
                "--model", "gemini", "--traces-dir", "/tmp/traces", "--eval-results", "/tmp/results.json",
                "--agent-source-dir", "/tmp/h0", "--output-file", "/tmp/analysis.jsonl",
                "--instance-ids-file", "/tmp/regressions.txt"]
        with patch.dict(driver.os.environ, {}), patch("sys.argv", argv), \
                patch.object(driver, "stage_references", return_value=Path("/tmp/references")), \
                patch.object(driver, "load_model", return_value={"model": "offline"}), \
                patch.object(driver, "_run") as execute:
            driver.main()
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index("--mode") + 1], "openhands")
        self.assertEqual(command[command.index("--instance-ids-file") + 1], "/tmp/regressions.txt")


class SharedPromotionTests(unittest.TestCase):
    def test_swe_and_openhands_share_exact_promotion_result(self) -> None:
        import run_pipeline_swe as swe

        val = {"net_change": 2, "regression_count": 3, "improvement_count": 5,
               "regressed_ids": ["a", "b", "c"], "improved_ids": ["d"], "cost_ratio": 9.0,
               "target_metric_results": {"improved_metric_count": 1},
               "baseline_metrics": {"metrics": {"error_rate": 0.1}},
               "current_metrics": {"metrics": {"error_rate": 0.2}}}
        train = {"net_change": -3}
        audit = {"passed": True}
        expected = decide_promotion(audit, train, val)
        self.assertTrue(expected["promoted"])
        self.assertFalse(expected["cost_gate_enabled"])
        with tempfile.TemporaryDirectory() as temporary, patch.object(swe, "FAILURE_ANALYSIS_DIR", Path(temporary)):
            with redirect_stdout(io.StringIO()):
                actual = swe.step_promotion_decision(1, audit, train, val, 1, 1, 0.15, 0.15, False)
        self.assertEqual(actual, expected)

    def test_audit_target_metrics_and_error_delta_are_required(self) -> None:
        val = {"net_change": 1, "target_metric_results": {"improved_metric_count": 1},
               "baseline_metrics": {"metrics": {"error_rate": 0.0}},
               "current_metrics": {"metrics": {"error_rate": 0.0}}}
        for audit, changes, reason in (
            ({"passed": False}, {}, "audit_passed"),
            ({"passed": True}, {"net_change": 0}, "net_improvement"),
            ({"passed": True}, {"target_metric_results": {"improved_metric_count": 0}}, "target_metric_improved"),
            ({"passed": True}, {"current_metrics": {"metrics": {"error_rate": 0.2}}}, "error_delta_within_limit"),
        ):
            with self.subTest(reason=reason):
                result = decide_promotion(audit, {}, val | changes)
                self.assertFalse(result["promoted"])
                self.assertIn(reason, result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()

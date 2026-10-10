from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_pipeline_swe as pipeline


class SweEvalReportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        ids = self.root / "instance_ids.txt"
        ids.write_text("test-instance\n")
        patches = patch.multiple(
            pipeline,
            REPO_ROOT=self.root,
            EVAL_DIR=self.root / "eval",
            TRAIN_IDS_FILE=ids,
            VAL_IDS_FILE=ids,
            RUN_LABEL="report_test",
        )
        patches.start()
        self.addCleanup(patches.stop)
        self.model = "openai/Qwen/test-model"

    def write_report(self, directory: Path, run_id: str, resolved: bool = False) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{pipeline.eval_model_slug(self.model)}.{run_id}.json"
        path.write_text(json.dumps({"resolved_ids": ["test-instance"] if resolved else []}))
        return path

    def evaluation_steps(self):
        traces = self.root / "traces"
        return (
            (pipeline.train_run_id(self.model, 0), lambda: pipeline.step_train_evaluate(
                self.model, 0, traces, 1, force=False, dry_run=False)),
            (pipeline.train_run_id(self.model, 1), lambda: pipeline.step_train_evaluate(
                self.model, 1, traces, 1, force=False, dry_run=False)),
            (pipeline.val_baseline_run_id(self.model), lambda: pipeline.step_val_baseline_evaluate(
                self.model, traces, 1, force=False, dry_run=False)),
            (pipeline.val_enhanced_run_id(self.model, 1), lambda: pipeline.step_val_evaluate(
                1, traces, self.model, 1, dry_run=False)),
        )

    def test_cached_root_reports_resume_all_evaluation_stages_without_rerunning(self) -> None:
        for run_id, evaluate in self.evaluation_steps():
            with self.subTest(run_id=run_id):
                source = self.write_report(self.root, run_id)
                with patch.object(pipeline, "run") as execute:
                    directory = evaluate()
                execute.assert_not_called()
                self.assertEqual((directory / source.name).read_bytes(), source.read_bytes())
                self.assertTrue(source.exists())

    def test_fresh_root_reports_are_collected_after_all_evaluation_stages(self) -> None:
        for run_id, evaluate in self.evaluation_steps():
            with self.subTest(run_id=run_id):
                def evaluator(command):
                    self.assertEqual(command[command.index("-id") + 1], run_id)
                    self.write_report(self.root, run_id)

                with patch.object(pipeline, "run", side_effect=evaluator) as execute:
                    directory = evaluate()
                execute.assert_called_once()
                self.assertIsNotNone(pipeline.find_eval_json(directory))

    def test_newer_root_report_replaces_an_older_collected_report(self) -> None:
        run_id = "rerun"
        destination = self.write_report(self.root / "eval", run_id)
        source = self.write_report(self.root, run_id, resolved=True)
        os.utime(destination, ns=(1, 1))
        os.utime(source, ns=(2, 2))
        self.assertEqual(pipeline.collect_eval_report(destination.parent, self.model, run_id), destination)
        self.assertEqual(json.loads(destination.read_text())["resolved_ids"], ["test-instance"])

    def test_report_written_to_requested_directory_takes_precedence_over_old_root_report(self) -> None:
        run_id = "direct_output"
        source = self.write_report(self.root, run_id)
        destination = self.write_report(self.root / "eval", run_id, resolved=True)
        os.utime(source, ns=(1, 1))
        os.utime(destination, ns=(2, 2))
        self.assertEqual(pipeline.collect_eval_report(destination.parent, self.model, run_id), destination)
        self.assertEqual(json.loads(destination.read_text())["resolved_ids"], ["test-instance"])

    def test_reports_for_other_models_or_runs_are_not_reused(self) -> None:
        self.write_report(self.root, "another_run")
        (self.root / "another_model.target_run.json").write_text("{}")
        self.assertIsNone(pipeline.collect_eval_report(self.root / "eval", self.model, "target_run"))

    def test_missing_report_is_detected_immediately_after_evaluation(self) -> None:
        for run_id, evaluate in self.evaluation_steps():
            with self.subTest(run_id=run_id), patch.object(pipeline, "run"):
                with self.assertRaisesRegex(FileNotFoundError, "Evaluator report"):
                    evaluate()

    def test_subprocesses_run_from_repository_root(self) -> None:
        with patch.object(pipeline.subprocess, "run") as execute:
            pipeline.run(["python", "-m", "swebench.harness.run_evaluation"])
        self.assertEqual(execute.call_args.kwargs["cwd"], self.root)


if __name__ == "__main__":
    unittest.main()

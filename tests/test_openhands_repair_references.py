from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from task_agent.openhands_agent.repair_references import (
    REFERENCE_DEFAULTS, REFERENCE_ENV, reference_options, stage_references, technical_reference_prompt, validate_references,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "task_agent/mini-swe-agent/src"))

from minisweagent.exceptions import Submitted


class RecordingModel:
    """Capture the actual first model request and end the offline agent run."""

    def __init__(self, submission: str):
        self.submission = submission
        self.requests = []

    def get_template_vars(self):
        return {}

    def format_message(self, **kwargs):
        return kwargs

    def serialize(self):
        return {}

    def query(self, messages):
        self.requests.append(copy.deepcopy(messages))
        raise Submitted({"role": "exit", "content": "Submitted", "extra": {
            "exit_status": "Submitted", "submission": self.submission,
        }})


class RepairReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.better = self.root / "better"
        docs = self.better / "docs"
        (docs / "sdk_reference_details").mkdir(parents=True)
        for name, text in {
            "sdk_reference.md": "SDK_REFERENCE_INPUT\nLiteral example: {{ example }}",
            "read_trajectory.md": "RAW_TRAJECTORY_REFERENCE_INPUT\nRaw events are a list.",
            "sdk_reference_details/02_agent.md": "Agent API details, read on demand.",
            "adaptation.md": "EXCLUDED_ADAPTATION_GUIDE\nCreate tools to fix tool-use failures.",
            "benchmark_tasks_overview.md": "EXCLUDED_BENCHMARK_OVERVIEW",
        }.items():
            (docs / name).write_text(text)
        (self.better / "data").mkdir()
        (self.better / "data/test.json").write_text("EXCLUDED_TEST_DATA")
        self.references = stage_references(self.better, self.root / "references")
        env = patch.dict(os.environ, {REFERENCE_ENV: str(self.references), "LITELLM_LOCAL_MODEL_COST_MAP": "True"})
        env.start()
        self.addCleanup(env.stop)

    def assert_references_in_request(self, model):
        self.assertEqual(len(model.requests), 1)
        system = model.requests[0][0]["content"]
        self.assertIn("SDK_REFERENCE_INPUT", system)
        self.assertIn("RAW_TRAJECTORY_REFERENCE_INPUT", system)
        self.assertIn("{{ example }}", system)  # Reference content is not re-rendered as Jinja.
        self.assertIn("sanitized trace is an OBJECT", system)
        self.assertIn("implementation spec", system)
        self.assertNotIn("EXCLUDED_", system)

    def test_document_snapshot_excludes_strategy_and_rejects_changed_references(self):
        manifest = validate_references(self.references)
        self.assertEqual(set(manifest["files"]), {
            "sdk_reference.md", "read_trajectory.md", "sdk_reference_details/02_agent.md",
        })
        self.assertFalse((self.references / "adaptation.md").exists())
        (self.references / "read_trajectory.md").write_text("Changed event schemas")
        with self.assertRaisesRegex(ValueError, "Technical reference changed"):
            technical_reference_prompt()

    def test_options_allow_no_references_but_cannot_enable_adaptation_guide(self):
        disabled = stage_references(self.root / "missing_better", self.root / "disabled",
                                    dict.fromkeys(REFERENCE_DEFAULTS, False))
        with patch.dict(os.environ, {REFERENCE_ENV: str(disabled)}):
            self.assertEqual(technical_reference_prompt(), "")
        trace_only = stage_references(self.better, self.root / "trace_only", {"sdk_reference": False})
        with patch.dict(os.environ, {REFERENCE_ENV: str(trace_only)}):
            self.assertIn("RAW_TRAJECTORY_REFERENCE_INPUT", technical_reference_prompt())
            self.assertNotIn("SDK_REFERENCE_INPUT", technical_reference_prompt())
        for invalid in ({"sdk_reference": "yes"}, {"adaptation_guide": True}, {"teacher_answers": True}, []):
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                reference_options(invalid)

    def test_reference_manifest_cannot_include_dataset_paths(self):
        manifest_path = self.references / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        name = "../better/data/test.json"
        manifest["files"][name] = hashlib.sha256((self.references / name).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Unexpected technical reference path"):
            technical_reference_prompt()

    def test_analysis_receives_technical_inputs_in_real_rendered_request(self):
        from failure_analysis import run_analysis as analysis

        model = RecordingModel(json.dumps({"failure_reason": "Synthetic schema mismatch"}))
        paths = {"traj_path": str(self.root / "trace.json"), "raw_traj_path": str(self.root / "raw.json"),
                 "manifest_path": str(self.root / "manifest.json"), "output_path": str(self.root / "analysis.traj.json")}
        config = yaml.safe_load((REPO_ROOT / "failure_analysis/analysis_config_openhands.yaml").read_text())
        with (
            patch.object(analysis, "_openhands_get_paths", return_value=paths),
            patch.object(analysis, "_openhands_manifest", return_value={"task_instance_id": "synthetic", "rollout_id": 0}),
            patch.object(analysis, "_openhands_evidence_anchor", return_value={"score": 0}),
            patch.object(analysis, "_openhands_task_description", return_value="Synthetic task"),
            patch.object(analysis, "load_openhands_trace", return_value={"events": []}),
            patch.object(analysis, "_sanitized_traj_path", return_value=self.root / "sanitized.json"),
            patch.object(analysis, "compile_openhands_htir", return_value={}),
            patch.object(analysis, "write_bundle", return_value=self.root / "trace.htir.json"),
            patch.object(analysis, "_agent_source_context", return_value={"agent_source_dir": str(self.root / "h0"),
                                                                        "agent_source_root": str(self.root)}),
            patch.object(analysis, "_enrich_analysis_output", side_effect=lambda parsed, *args: parsed),
            patch.object(analysis, "LitellmTextbasedModel", return_value=model),
            patch.object(analysis, "selected_model_kwargs", return_value={}),
        ):
            result = analysis._openhands_run_analysis("synthetic", "unresolved", "offline", config, "", logging.getLogger(__name__))
        self.assertEqual(result["failure_reason"], "Synthetic schema mismatch")
        self.assert_references_in_request(model)

    def test_aggregate_receives_technical_inputs_only_in_openhands_mode(self):
        from failure_analysis import aggregate_results as aggregate

        context = {key: self.root / f"{key}.txt" for key in (
            "distribution", "layer_buckets", "analyses", "operator_registry", "memory", "val_regressions", "previous_context",
        )}
        for mode in ("openhands", "swe"):
            model = RecordingModel("Synthetic plan " + "x" * 250)
            with patch.object(aggregate, "LitellmTextbasedModel", return_value=model):
                aggregate.call_aggregate_agent(model="offline", mode=mode, mode_system_prompt="Scoped repair plan",
                                               context_files=context, output_path=self.root / f"{mode}.md")
            if mode == "openhands":
                self.assert_references_in_request(model)
            else:
                self.assertNotIn("SDK_REFERENCE_INPUT", model.requests[0][0]["content"])

    def test_modify_receives_references_without_adding_them_to_task_candidate(self):
        import run_pipeline_openhands as driver
        from minisweagent.models import litellm_textbased_model

        original = self.root / "h0"
        original.mkdir()
        (original / "agent.py").write_text("def build_agent(base_dir, llm):\n    return llm\n")
        model = RecordingModel("Completed synthetic edit")
        with patch.object(litellm_textbased_model, "LitellmTextbasedModel", return_value=model), \
                patch.object(driver, "selected_model_kwargs", return_value={}):
            driver._modify_candidate(base_dir=original, target_dir=self.root / "h1", plan_path=self.root / "plan.md",
                                     spec_path=self.root / "plan.json", model_name="offline", redo_feedback=None,
                                     trajectory_output=self.root / "modify.traj.json")
        self.assert_references_in_request(model)
        self.assertEqual((self.root / "h1/agent.py").read_bytes(), (original / "agent.py").read_bytes())
        self.assertEqual(list((self.root / "h1").iterdir()), [self.root / "h1/agent.py"])

    def test_modifier_retry_has_identical_model_context_in_a_fresh_session(self):
        import run_pipeline_openhands as driver
        from minisweagent.models import litellm_textbased_model

        original = self.root / "h0"
        original.mkdir()
        source = "def build_agent(base_dir, llm):\n    return llm\n"
        (original / "agent.py").write_text(source)
        contexts = []
        for attempt in range(2):
            model = RecordingModel("Completed synthetic edit")
            with patch.object(litellm_textbased_model, "LitellmTextbasedModel", return_value=model), \
                    patch.object(driver, "selected_model_kwargs", return_value={}):
                driver._modify_candidate(base_dir=original, target_dir=self.root / "h1", plan_path=self.root / "plan.md",
                                         spec_path=self.root / "plan.json", model_name="offline", redo_feedback=None,
                                         trajectory_output=self.root / f"modify_{attempt}.traj.json")
            self.assert_references_in_request(model)
            contexts.append([{key: message[key] for key in ("role", "content")} for message in model.requests[0]])
            self.assertEqual((self.root / "h1/agent.py").read_text(), source)
            # A failed candidate remains on disk for review, outside the next modifier's input.
            (self.root / "h1/agent.py").write_text("raise AttributeError('synthetic-previous-failure')\n")
            (self.root / "h1").rename(self.root / f"attempt_{attempt}")
        self.assertEqual(contexts[0], contexts[1])
        self.assertNotIn("synthetic-previous-failure", str(contexts[1]))


if __name__ == "__main__":
    unittest.main()

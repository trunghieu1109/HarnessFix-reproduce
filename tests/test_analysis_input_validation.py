from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from failure_analysis.analysis_records import analysis_record_errors, analysis_record_succeeded
from failure_analysis.artifact_sanitizer import sanitize_for_prompt
from failure_analysis.print_openhands_trace import render_trace
from tests.diagnosis_fixtures import valid_diagnosis


def bundle(mode="openhands"):
    return {
        "mode": mode, "instance_id": "one",
        "agent_trace_steps": [{"step_id": "s1", "execution_status": "submitted", "legacy_event_ids": ["e1"]}],
        "graph": {"nodes": [{"node_id": "e1", "source_ref": "trace.events[0]"}]},
        "node_facets": {"s1": {"harness_layer": {"implicated_layers": ["Verification"]}}},
    }


class ModelEvidenceInputTests(unittest.TestCase):
    def test_normalized_model_and_usage_survive_sanitize_and_selected_projection(self):
        call = {
            "response_id": "r1", "model": "recorded-task-model",
            "usage": {"prompt_tokens": 123, "completion_tokens": 0, "total_tokens": 123},
            "request_messages": [{"role": "user", "content": "Check the result."}],
            "request_payload": {"tools": [{"type": "function", "function": {"name": "check"}}]},
            "response_message": {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "check", "arguments": "{}"}},
            ]},
            "generation_settings": {"temperature": 0.2},
            "request_evidence": "sdk_completion_log", "response_evidence": "sdk_completion_log",
            "api_key": "synthetic-credential",
            "raw_response": {"model": "duplicate", "usage": {"total_tokens": 123}},
        }
        trace = {"events": [{"kind": "ActionEvent", "llm_response_id": "r1", "action": {"kind": "MCPToolAction"}}],
                 "model_calls": [call]}
        original = copy.deepcopy(trace)
        sanitized = sanitize_for_prompt(trace)
        kept = sanitized["model_calls"][0]
        for field in ("model", "usage", "request_messages", "response_message", "generation_settings"):
            self.assertEqual(kept[field], call[field])
        self.assertNotIn("synthetic-credential", json.dumps(sanitized))
        self.assertNotIn("raw_response", kept)
        output = render_trace(sanitized, response_ids=["r1"])
        projection = next(json.loads(line) for line in output.splitlines() if '"request_messages_count"' in line)
        self.assertEqual(projection["model"], call["model"])
        self.assertEqual(projection["usage"], call["usage"])
        self.assertEqual(trace, original)

    def test_other_normalized_benchmark_and_subagent_calls_keep_metadata(self):
        call = {"call_index": 1, "model": "task-model", "usage": {"total_tokens": 17},
                "request_messages": [{"role": "user", "content": "Task"}],
                "response_message": {"role": "assistant", "content": "Answer"}}
        clean = sanitize_for_prompt({"model_calls": [call], "subagents": {"worker": {"model_calls": [call]}},
                                     "provider": {"model": "duplicate", "usage": {"total_tokens": 17},
                                                  "choices": [{"message": {"content": "duplicate"}}]}})
        self.assertEqual(clean["model_calls"][0], call)
        self.assertEqual(clean["subagents"]["worker"]["model_calls"][0], call)
        self.assertNotIn("provider", clean)


class DiagnosisValidationTests(unittest.TestCase):
    def test_complete_diagnosis_does_not_require_attribution(self):
        record = valid_diagnosis()
        self.assertEqual(analysis_record_errors(record, htir_bundle=bundle()), [])
        self.assertTrue(analysis_record_succeeded(record))

    def test_missing_content_and_invalid_types_are_rejected(self):
        bad_records = [None, [], "text", {}, {"instance_id": "one"}]
        for field, value in (("failure_reason", ""), ("agent_design_issue", "FILL_design_issue"),
                             ("exit_status", {}), ("confidence", []), ("severity", "critical"),
                             ("api_calls", True), ("api_calls", -1), ("api_calls", "3"),
                             ("responsible_steps", "s1"), ("responsible_nodes", [123]),
                             ("evidence_spans", {}), ("evidence_spans", [{}]),
                             ("evidence_spans", [{"step_id": ["s1"]}])):
            bad_records.append(valid_diagnosis(**{field: value}))
        for record in bad_records:
            with self.subTest(record=record):
                self.assertTrue(analysis_record_errors(record))
                self.assertFalse(analysis_record_succeeded(record))

    def test_wrong_case_and_nonexistent_ids_are_rejected(self):
        for changes in (
            {"instance_id": "other-rollout"}, {"failure_category": "error"},
            {"responsible_steps": ["s999"]}, {"candidate_responsible_steps": ["s999"]},
            {"responsible_nodes": ["e999"]}, {"downstream_events": ["e999"]},
            {"evidence_spans": [{"step_id": "s999", "node_id": "e1"}]},
            {"evidence_spans": [{"step_id": "s1", "event_id": "e999"}]},
        ):
            with self.subTest(changes=changes):
                self.assertTrue(analysis_record_errors(valid_diagnosis(**changes),
                    expected_instance_id="one", expected_failure_category="unresolved", htir_bundle=bundle()))
        valid = valid_diagnosis(responsible_steps=["s1"], responsible_events=["e1"],
                               evidence_spans=[{"step_id": "s1", "node_id": "e1", "source_ref": "trace.events[0]"}])
        self.assertEqual(analysis_record_errors(valid, htir_bundle=bundle()), [])

    def test_saved_references_are_checked_when_bundle_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "htir.json"
            path.write_text(json.dumps(bundle()))
            self.assertTrue(analysis_record_succeeded(valid_diagnosis(htir_path=str(path), responsible_steps=["s1"])))
            self.assertFalse(analysis_record_succeeded(valid_diagnosis(htir_path=str(path), responsible_steps=["s999"])))
            self.assertFalse(analysis_record_succeeded(valid_diagnosis("another-rollout", htir_path=str(path), responsible_steps=["s1"])))
            path.write_text("not JSON")
            self.assertFalse(analysis_record_succeeded(valid_diagnosis(htir_path=str(path))))
            path.unlink()
            self.assertTrue(analysis_record_succeeded(valid_diagnosis(htir_path=str(path))))


class DiagnosisFinalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from failure_analysis import run_analysis
        cls.runner = run_analysis

    def finalize(self, record, mode="openhands"):
        return self.runner._finalize_analysis_output(
            record, instance_id="one", failure_category="unresolved", api_calls=3,
            evidence_anchor={"score": 0}, bundle=bundle(mode), htir_path=Path("htir.json"),
        )

    def test_invalid_submission_is_failed_before_enrichment_for_every_mode(self):
        for mode in ("swe", "gaia", "appworld", "terminal_bench", "openhands"):
            for record in ({}, [], valid_diagnosis(responsible_steps=["s999"]), valid_diagnosis(instance_id="other")):
                with self.subTest(mode=mode, record=record):
                    result = self.finalize(record, mode)
                    self.assertEqual(result["exit_status"], "analysis_invalid_diagnosis")
                    self.assertEqual(result["instance_id"], "one")
                    self.assertEqual(result["evidence_anchor"], {"score": 0})
                    self.assertTrue(result["_analysis_validation_errors"])
                    self.assertFalse(analysis_record_succeeded(result))

    def test_valid_diagnosis_preserves_existing_attribution_and_shared_defaults(self):
        record = valid_diagnosis(responsible_steps=["s1"], responsible_nodes=["e1"], confidence="high")
        result = self.finalize(record)
        for key, value in record.items():
            self.assertEqual(result[key], value)
        self.assertTrue(analysis_record_succeeded(result))
        self.assertNotIn("_analysis_validation_errors", result)
        result = self.finalize(valid_diagnosis())
        self.assertEqual(result["responsible_steps"], ["s1"])
        self.assertEqual(result["responsible_nodes"], ["e1"])


if __name__ == "__main__":
    unittest.main()

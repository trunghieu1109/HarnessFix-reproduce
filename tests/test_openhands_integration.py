from __future__ import annotations

import importlib.util
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from failure_analysis.consolidation import normalize_diagnosis
from failure_analysis.htir import compile_openhands_htir
from failure_analysis.openhands_io import normalize_rollout
from failure_analysis.operator_registry import operator_allowed_paths
from failure_analysis.plan_diff_audit import audit_candidate
from failure_analysis.validation_metrics import compute_openhands_metrics
from task_agent.openhands_agent.bridge import pack_candidate


def _write_trace(
    better_root: Path,
    example_index: int,
    *,
    score: float | None,
    rollout_id: int = 0,
    trace_error: str | None = None,
) -> dict:
    workspace = (
        better_root
        / "results"
        / "refactorbench"
        / "model_default"
        / "rollouts"
        / "baseline"
        / f"example{example_index}_rollout{rollout_id}"
    )
    workspace.mkdir(parents=True)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    log_dir.mkdir()
    events = [
        {"kind": "SystemPromptEvent", "system_prompt": "Solve the task."},
        {
            "kind": "MessageEvent",
            "source": "user",
            "llm_message": {"role": "user", "content": [{"type": "text", "text": "Refactor module A."}]},
        },
        {
            "kind": "ActionEvent",
            "source": "agent",
            "tool_name": "terminal",
            "tool_call_id": "call-1",
            "thought": [{"type": "text", "text": "Run the tests."}],
            "action": {"kind": "TerminalAction", "command": "pytest -q"},
        },
        {
            "kind": "ObservationEvent",
            "tool_call_id": "call-1",
            "observation": {
                "kind": "TerminalObservation",
                "content": [{"type": "text", "text": "1 passed"}],
                "is_error": False,
                "exit_code": 0,
            },
        },
        {
            "kind": "ActionEvent",
            "source": "agent",
            "tool_name": "finish",
            "action": {"kind": "FinishAction", "message": "Done"},
        },
    ]
    filtered = {
        "conversation_id": f"conversation-{example_index}-{rollout_id}",
        "eval_output": "Done",
        "events": events,
        "metrics": {"accumulated_cost": 0.12},
        "metrics_breakdown": {},
        "subagents": {},
        "error": trace_error,
    }
    (log_dir / f"trace_conversation-{example_index}-{rollout_id}.json").write_text(
        json.dumps(filtered), encoding="utf-8"
    )
    (log_dir / f"raw_trace_conversation-{example_index}-{rollout_id}.json").write_text(
        json.dumps(events), encoding="utf-8"
    )
    return {
        "workspace_dir": str(workspace.relative_to(better_root)),
        "score": score,
        "feedback": "Official evaluator reports that the requested repository state was not reached.",
    }


class OpenHandsArtifactTests(unittest.TestCase):
    def test_normalize_and_compile_use_evaluator_as_outcome_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            better_root = root / "better"
            better_root.mkdir()
            eval_results = [
                _write_trace(better_root, 0, score=0.0, rollout_id=0),
                _write_trace(
                    better_root,
                    0,
                    score=1.0,
                    rollout_id=1,
                    trace_error="post-run cleanup failed",
                ),
            ]
            eval_path = better_root / "eval_results.yaml"
            eval_path.write_text(yaml.safe_dump(eval_results), encoding="utf-8")

            results_path, traces_root = normalize_rollout(
                eval_results_path=eval_path,
                better_root=better_root,
                output_dir=root / "normalized",
                task_id="refactorbench",
            )
            normalized = json.loads(results_path.read_text(encoding="utf-8"))
            instance_id = "refactorbench__example0__rollout0"
            second_instance_id = "refactorbench__example0__rollout1"
            self.assertEqual(normalized["unresolved_ids"], [instance_id])
            self.assertEqual(normalized["resolved_ids"], [second_instance_id])
            self.assertEqual(normalized["error_ids"], [])
            self.assertEqual(
                normalized["rollout_groups"],
                {"refactorbench__example0": [instance_id, second_instance_id]},
            )

            manifest_path = traces_root / instance_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bundle = compile_openhands_htir(
                instance_id,
                "unresolved",
                {
                    "manifest_path": str(manifest_path),
                    "traj_path": manifest["trace_path"],
                },
                "Refactor module A.",
            )
            nodes = bundle["graph"]["nodes"]
            self.assertTrue(any(node["type"] == "ToolResultRecord" for node in nodes))
            self.assertTrue(any(node["type"] == "VerificationEvent" for node in nodes))
            self.assertFalse(any(node["type"] == "StateDeltaRecord" for node in nodes))
            self.assertEqual(bundle["stats"]["score"], 0.0)
            node_types = {node["node_id"]: node["type"] for node in nodes}
            self.assertTrue(any(
                edge["relation"] == "tool-invocation"
                and node_types.get(edge["from"]) == "ToolCallEvent"
                and node_types.get(edge["to"]) == "ToolResultRecord"
                for edge in bundle["graph"]["edges"]
            ))
            self.assertIn("score=0.0", bundle["evaluator_anchors"][0]["summary"])

    def test_none_score_is_fail_closed_as_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            better_root = root / "better"
            better_root.mkdir()
            eval_result = _write_trace(better_root, 1, score=None)
            eval_path = better_root / "eval_results.yaml"
            eval_path.write_text(yaml.safe_dump([eval_result]), encoding="utf-8")
            results_path, _ = normalize_rollout(
                eval_results_path=eval_path,
                better_root=better_root,
                output_dir=root / "normalized",
                task_id="webarena",
            )
            normalized = json.loads(results_path.read_text(encoding="utf-8"))
            instance_id = "webarena__example1__rollout0"
            self.assertEqual(normalized["unsupported_ids"], [instance_id])
            self.assertEqual(normalized["error_ids"], [instance_id])
            self.assertNotIn(instance_id, normalized["resolved_ids"])

    def test_compiler_uses_raw_trace_when_filtered_trace_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_result = _write_trace(root, 0, score=0.0)
            eval_path = root / "eval_results.yaml"
            eval_path.write_text(yaml.safe_dump([eval_result]), encoding="utf-8")
            _, traces_root = normalize_rollout(
                eval_results_path=eval_path,
                better_root=root,
                output_dir=root / "normalized",
                task_id="refactorbench",
            )
            instance_id = "refactorbench__example0__rollout0"
            manifest_path = traces_root / instance_id / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            Path(manifest["trace_path"]).unlink()
            bundle = compile_openhands_htir(
                instance_id,
                "unresolved",
                {"manifest_path": str(manifest_path), "traj_path": manifest["raw_trace_path"]},
            )
            self.assertTrue(any(node["type"] == "ToolResultRecord" for node in bundle["graph"]["nodes"]))
            self.assertEqual(bundle["stats"]["score"], 0.0)
            self.assertIn("Refactor module A.", bundle["graph"]["nodes"][0]["summary"])

    def test_validation_metrics_accept_event_list_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_result = _write_trace(root, 0, score=0.0)
            eval_path = root / "eval_results.yaml"
            eval_path.write_text(yaml.safe_dump([eval_result]), encoding="utf-8")
            results_path, traces_root = normalize_rollout(
                eval_results_path=eval_path,
                better_root=root,
                output_dir=root / "normalized",
                task_id="refactorbench",
            )
            manifest_path = traces_root / "refactorbench__example0__rollout0" / "manifest.json"
            trace_path = Path(json.loads(manifest_path.read_text())["trace_path"])
            events = json.loads(trace_path.read_text())["events"]
            trace_path.write_text(json.dumps(events), encoding="utf-8")
            metrics = compute_openhands_metrics(traces_root, results_path)["metrics"]
            self.assertEqual(metrics["resolved_rate"], 0.0)
            self.assertEqual(metrics["avg_steps"], 2.0)
            self.assertEqual(metrics["missing_evidence_rate"], 0.0)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("litellm", "dotenv", "pydantic")),
    "Analysis runner dependencies are not installed",
)
class OpenHandsAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from failure_analysis import run_analysis

        cls.runner = run_analysis

    def test_analysis_keeps_rollout_evidence_on_exception_or_invalid_submission(self) -> None:
        scenarios = (
            ("exception", RuntimeError("analysis backend unavailable"), ""),
            ("invalid_json", None, "not a diagnosis JSON"),
            ("submitted", None, json.dumps({"agent_design_issue": "Inspect the system prompt.", "confidence": "high"})),
            ("model_alias", None, json.dumps({"agent_design_issue": "Inspect the system prompt.", "confidence": "high"})),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_result = _write_trace(root, 0, score=0.0)
            eval_path = root / "eval_results.yaml"
            eval_path.write_text(yaml.safe_dump([eval_result]), encoding="utf-8")
            _, traces_root = normalize_rollout(
                eval_results_path=eval_path,
                better_root=root,
                output_dir=root / "normalized",
                task_id="refactorbench",
            )
            config = {"model": {"model_kwargs": {"temperature": 0.2}}}
            logger = logging.getLogger("tests.openhands_analysis")
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
            for name, exception, submission in scenarios:
                with self.subTest(name=name):
                    alias_kwargs = {"api_base": "http://localhost:8000/v1", "temperature": 0.0}
                    agent = Mock(n_calls=3)
                    agent.run.side_effect = exception
                    agent.run.return_value = {"exit_status": "Submitted", "submission": submission}
                    with (
                        patch.multiple(
                            self.runner,
                            _TRACES_DIR_OVERRIDE=traces_root,
                            _ALL_RESULTS_PATH_OVERRIDE=root / "analysis.jsonl",
                            RESULTS_DIR=root / "analysis_outputs",
                        ),
                        patch.dict(os.environ, {"HARNESSFIX_MODEL_ALIAS": "test-alias" if name == "model_alias" else ""}),
                        patch.object(self.runner, "DefaultAgent", return_value=agent),
                        patch.object(self.runner, "LitellmTextbasedModel") as model,
                        patch.object(self.runner, "selected_model_kwargs", return_value=alias_kwargs) as bridge_kwargs,
                    ):
                        result = self.runner._openhands_run_analysis(
                            "refactorbench__example0__rollout0", "unresolved", "test-model", config, "", logger,
                        )
                    self.assertEqual(result["instance_id"], "refactorbench__example0__rollout0")
                    self.assertEqual(result["task_instance_id"], "refactorbench__example0")
                    self.assertEqual(result["rollout_id"], 0)
                    self.assertEqual(result["evidence_anchor"]["score"], 0.0)
                    self.assertTrue(Path(result["htir_path"]).is_file())
                    expected_kwargs = config["model"]["model_kwargs"]
                    if name == "model_alias":
                        expected_kwargs = expected_kwargs | alias_kwargs
                        bridge_kwargs.assert_called_once_with()
                    else:
                        bridge_kwargs.assert_not_called()
                    self.assertEqual(model.call_args.kwargs["model_kwargs"], expected_kwargs)
                    if name == "exception":
                        self.assertTrue(result["_analysis_fallback"])
                        self.assertEqual(result["exit_status"], "analysis_agent_exception")
                        self.assertEqual(result["confidence"], "low")
                    elif name == "invalid_json":
                        self.assertTrue(result["_analysis_fallback"])
                        self.assertTrue(result["_parse_error"])
                    else:
                        self.assertFalse(result.get("_analysis_fallback", False))
                        self.assertEqual(result["confidence"], "high")


class OpenHandsBundleTests(unittest.TestCase):
    def test_packed_candidate_preserves_sibling_components(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / "candidate"
            (candidate / "skills").mkdir(parents=True)
            (candidate / "config.json").write_text('{"task_id": "refactorbench"}', encoding="utf-8")
            (candidate / "skills" / "verify.md").write_text("verify state", encoding="utf-8")
            (candidate / "agent.py").write_text(
                "from pathlib import Path\n"
                "ROOT = Path(__file__).resolve().parent\n"
                "def build_agent(base_dir, llm):\n"
                "    return (ROOT / 'skills' / 'verify.md').read_text()\n",
                encoding="utf-8",
            )
            packed = pack_candidate(candidate, root / "packed.py")
            spec = importlib.util.spec_from_file_location("packed_candidate_test", packed)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.build_agent("unused", None), "verify state")

    def test_component_taxonomy_maps_to_full_bundle_surface(self) -> None:
        self.assertIn("monitoring/**", operator_allowed_paths("openhands", "instrumentation"))
        self.assertIn("subagents/**", operator_allowed_paths("openhands", "orchestration"))
        self.assertIn("workspace_scripts/**", operator_allowed_paths("openhands", "tool_affordance"))
        self.assertIn("agent.py", operator_allowed_paths("openhands", "verification"))
        diagnosis = normalize_diagnosis({"affected_component": "monitoring", "failure_category": "error"})
        self.assertEqual(diagnosis["recommended_operator_family"], "instrumentation")
        self.assertEqual(diagnosis["implicated_harness_layers"], ["Observability"])

    def test_plan_diff_audit_accepts_only_selected_component(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "original"
            candidate = root / "candidate"
            for directory in (original, candidate):
                (directory / "skills").mkdir(parents=True)
                (directory / "skills" / "verify.md").write_text("old", encoding="utf-8")
                (directory / "agent.py").write_text("def build_agent(base_dir, llm): pass\n", encoding="utf-8")
            (candidate / "skills" / "verify.md").write_text("new", encoding="utf-8")
            spec = {
                "edit_budget": {
                    "allowed_paths": ["skills/**"],
                    "forbidden_paths": ["agent.py"],
                    "max_files_to_modify": 1,
                },
                "fixes": [{"id": "F1", "target_files": ["skills/**"]}],
            }
            result = audit_candidate(original, candidate, spec)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["changed_files"], ["skills/verify.md"])


if __name__ == "__main__":
    unittest.main()

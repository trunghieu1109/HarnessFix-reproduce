from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from failure_analysis.htir import compile_openhands_htir
from failure_analysis.openhands_io import normalize_rollout
from failure_analysis.openhands_trace import load_openhands_trace, openhands_model_calls
from failure_analysis.validation_metrics import compute_openhands_metrics
from task_agent.openhands_agent.bridge import pack_candidate


class OpenHandsCompletionEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "example0_rollout0"
        self.workspace.mkdir()
        self.logs = self.root / "example0_rollout0_logs"
        self.logs.mkdir()
        self.completions = self.logs / "llm_completions"
        self.completions.mkdir()

    def normalize(self, trace: dict | list) -> tuple[Path, dict]:
        (self.logs / "trace_conversation.json").write_text(json.dumps(trace))
        (self.logs / "raw_trace_conversation.json").write_text(json.dumps(trace.get("events", []) if isinstance(trace, dict) else trace))
        evaluation = self.root / "eval_results.yaml"
        evaluation.write_text(yaml.safe_dump([{"workspace_dir": str(self.workspace), "score": 0.0}]))
        self.results, self.traces = normalize_rollout(
            eval_results_path=evaluation, better_root=self.root,
            output_dir=self.root / "normalized", task_id="refactorbench",
        )
        path = next(self.traces.glob("*/manifest.json"))
        return path, json.loads(path.read_text())

    def compile(self, path: Path, manifest: dict) -> dict:
        return compile_openhands_htir(manifest["instance_id"], "unresolved", {
            "manifest_path": str(path), "traj_path": manifest["trace_path"],
        })

    def write_completion(self, response_id: str, timestamp: float, **fields) -> None:
        payload = {
            "messages": [{"role": "user", "content": "Inspect inventory."}],
            "response": {"id": response_id, "model": "test-model", "choices": [{
                "message": {"role": "assistant", "content": "Recorded answer."},
            }], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            "timestamp": timestamp,
        } | fields
        (self.completions / f"{response_id}.json").write_text(json.dumps(payload))

    def test_logged_requests_auxiliary_calls_and_responses_api_input_reach_htir(self) -> None:
        request = [{"role": "system", "content": "Use inventory tools."}, {"role": "user", "content": "Inspect inventory and notify the manager about each low-stock item."}]
        tools = [{"type": "function", "function": {"name": "inventory", "parameters": {}}}]
        self.write_completion("aux", 1.0, instructions="Validate tool syntax.", input=[{"role": "user", "content": "Check arguments."}], messages=[])
        self.write_completion("reply", 2.0, messages=request, tools=tools,
                              raw_response={"choices": [{"message": {"role": "assistant", "content": "Original unconverted output."}}]},
                              kwargs={"temperature": 0.6, "api_key": "must-not-copy"})
        events = [
            {"kind": "MessageEvent", "source": "user", "llm_message": request[-1]},
            {"kind": "MessageEvent", "source": "agent", "llm_response_id": "reply", "llm_message": {"role": "assistant", "content": "Event projection."}},
        ]
        path, manifest = self.normalize({"events": events})
        self.assertEqual(len(manifest["llm_completion_paths"]), 2)
        trace = load_openhands_trace(manifest)
        calls = openhands_model_calls(trace)
        self.assertEqual([c["response_id"] for c in calls], ["aux", "reply"])
        self.assertEqual(calls[1]["request_messages"], request)
        self.assertEqual(calls[1]["request_payload"]["tools"], tools)
        self.assertEqual(calls[1]["generation_settings"], {"temperature": 0.6})
        bundle = self.compile(path, manifest)
        model_nodes = [n for n in bundle["graph"]["nodes"] if n["type"] == "ModelInvocationEvent"]
        self.assertEqual(len(model_nodes), 2)
        self.assertEqual(len({n["step_id"] for n in model_nodes}), 2)
        self.assertEqual(model_nodes[1]["attributes"]["response_message"]["content"], "Recorded answer.")
        self.assertEqual(model_nodes[1]["attributes"]["original_response_message"]["content"], "Original unconverted output.")
        self.assertEqual(model_nodes[1]["attributes"]["usage"]["prompt_tokens"], 12)
        self.assertIn("Check arguments.", model_nodes[0]["attributes"]["request_summary"])
        self.assertEqual(bundle["stats"]["recorded_request_count"], 2)
        self.assertTrue(any(e["relation"] == "context-dependency" for e in bundle["graph"]["edges"]))

    def test_partial_events_keep_reasoning_group_actions_and_link_parse_errors(self) -> None:
        events = [{"kind": "MessageEvent", "source": "user", "llm_message": {"role": "user", "content": "Inspect inventory."}}]
        for call_id in ("tool-a", "tool-b"):
            events.append({"kind": "ActionEvent", "source": "agent", "llm_response_id": "multi", "tool_call_id": call_id,
                           "tool_call": {"id": call_id, "name": "inventory", "arguments": "{}"},
                           "action": {"kind": "InventoryAction"}, "reasoning_content": "Inspect both stores."})
            events.append({"kind": "ObservationEvent", "tool_call_id": call_id, "observation": {"content": "Store checked."}})
        events.extend([
            {"kind": "MessageEvent", "source": "agent", "llm_response_id": "reasoning", "llm_message": {"role": "assistant", "content": [], "reasoning_content": "Recover tool arguments."}},
            {"kind": "ActionEvent", "llm_response_id": "invalid", "tool_call_id": "bad", "tool_call": {"id": "bad", "name": "inventory", "arguments": "{}"}, "action": None},
            {"kind": "AgentErrorEvent", "tool_call_id": "bad", "error": "Required argument missing."},
        ])
        metrics = {"response_latencies": [{"response_id": key} for key in ("multi", "reasoning", "invalid", "unrecorded")],
                   "token_usages": [{"response_id": "multi", "prompt_tokens": 100}]}
        path, manifest = self.normalize({"events": events, "metrics": metrics})
        bundle = self.compile(path, manifest)
        nodes = bundle["graph"]["nodes"]
        calls = [n for n in nodes if n["type"] == "ModelInvocationEvent"]
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(calls[0]["attributes"]["response_message"]["tool_calls"]), 2)
        self.assertEqual(calls[0]["attributes"]["response_message"]["reasoning_content"], "Inspect both stores.")
        self.assertIn("Recover tool arguments.", calls[1]["attributes"]["response_summary"])
        self.assertTrue(all(c["attributes"]["request_evidence"] == "not_recorded" for c in calls))
        self.assertTrue(all(not c["attributes"]["request_summary"] for c in calls))
        self.assertEqual(sum(n["type"] == "ContextAssemblyEvent" for n in nodes), 1)
        self.assertEqual(sum(n["type"] == "ToolCallEvent" for n in nodes), 2)
        trace = load_openhands_trace(manifest)
        trace["model_calls"] = openhands_model_calls(trace)
        self.assertEqual(openhands_model_calls(trace), trace["model_calls"])
        invalid_parser = next(n for n in nodes if n["type"] == "ParserEvent" and not n["attributes"]["parsed"])
        error = next(n for n in nodes if n["type"] == "ExceptionEvent")
        self.assertIn({"from": invalid_parser["node_id"], "to": error["node_id"], "relation": "causal"}, bundle["graph"]["edges"])
        self.assertEqual(bundle["stats"]["api_calls"], 4)
        self.assertEqual(bundle["stats"]["missing_model_call_records"], 1)
        metrics = compute_openhands_metrics(self.traces, self.results)["metrics"]
        self.assertEqual(metrics["avg_steps"], 3)
        self.assertEqual(metrics["avg_api_calls"], 4)

    def test_raw_state_recovers_metrics_and_missing_completion_is_reported(self) -> None:
        events = [{"kind": "ConversationStateUpdateEvent", "key": "full_state", "value": {"stats": {"usage_to_metrics": {
            "default": {"response_latencies": [{"response_id": "one"}, {"response_id": "two"}], "accumulated_cost": 0.2},
        }}}}, {"kind": "ActionEvent", "llm_response_id": "one", "action": {"kind": "InventoryAction"}}]
        path, manifest = self.normalize(events)
        Path(manifest["trace_path"]).unlink()
        metrics = compute_openhands_metrics(self.traces, self.results)["metrics"]
        self.assertEqual(metrics["avg_api_calls"], 2)
        self.assertEqual(metrics["missing_evidence_rate"], 0)
        self.assertEqual(metrics["avg_instance_cost"], 0.2)
        manifest["llm_completion_paths"] = [str(self.completions / "missing.json")]
        with self.assertRaisesRegex(FileNotFoundError, "completion log is missing"):
            load_openhands_trace(manifest)

    def test_request_error_log_keeps_request_and_exception(self) -> None:
        (self.completions / "error.json").write_text(json.dumps({
            "messages": [{"role": "user", "content": "Inspect inventory."}],
            "error": {"type": "RuntimeError", "message": "Backend failed."}, "timestamp": 1,
        }))
        path, manifest = self.normalize({"events": []})
        bundle = self.compile(path, manifest)
        self.assertEqual(bundle["stats"]["api_calls"], 1)
        self.assertTrue(any(n["type"] == "ExceptionEvent" and "Backend failed." in n["summary"] for n in bundle["graph"]["nodes"]))

    def test_streamed_completion_event_is_ingested_without_duplicate_model_nodes(self) -> None:
        self.write_completion("streamed", 1)
        payload = (self.completions / "streamed.json").read_text()
        events = [{"kind": "LLMCompletionLogEvent", "filename": "streamed.json", "log_data": payload}]
        path, manifest = self.normalize({"events": events})
        self.assertEqual(len(load_openhands_trace(manifest)["model_calls"]), 1)
        (self.completions / "streamed.json").unlink()
        manifest["llm_completion_paths"] = []
        path.write_text(json.dumps(manifest))
        trace = load_openhands_trace(manifest)
        self.assertEqual(trace["model_calls"][0]["request_messages"][0]["content"], "Inspect inventory.")
        bundle = self.compile(path, manifest)
        self.assertEqual(bundle["stats"]["model_call_count"], 1)
        self.assertFalse(any(n["type"] == "OrchestrationEvent" for n in bundle["graph"]["nodes"]))

    def test_packed_launcher_initializes_logging_and_preserves_model_config(self) -> None:
        from pydantic import BaseModel, PrivateAttr

        class FakeLLM(BaseModel):
            model: str
            api_key: str
            base_url: str
            log_completions: bool = False
            log_completions_folder: str = "logs/completions"
            _logging_initialized: bool = PrivateAttr(default=False)

            def model_post_init(self, context):
                self._logging_initialized = self.log_completions

        candidate = self.root / "candidate"
        candidate.mkdir()
        (candidate / "agent.py").write_text("def build_agent(base_dir, llm): return llm\n")
        launcher = pack_candidate(candidate, self.root / "launcher.py")
        namespace = {}
        exec(compile(launcher.read_text(), str(launcher), "exec"), namespace)
        original = FakeLLM(model="qwen", api_key="test-key", base_url="http://localhost/v1")
        configured = namespace["build_agent"](str(self.workspace), original)
        self.assertTrue(configured.log_completions)
        self.assertTrue(configured._logging_initialized)
        self.assertFalse(original._logging_initialized)
        self.assertEqual(Path(configured.log_completions_folder), self.completions)
        self.assertEqual(configured.api_key, original.api_key)
        self.assertEqual(configured.base_url, original.base_url)


if __name__ == "__main__":
    unittest.main()

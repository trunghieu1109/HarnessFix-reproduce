from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from failure_analysis.print_openhands_trace import render_trace, select_evidence


REPO_ROOT = Path(__file__).resolve().parent.parent


def _trace() -> dict:
    return {
        "events": [
            {"kind": "SystemPromptEvent", "system_prompt": "Solve task"},
            {"kind": "LLMCompletionLogEvent", "filename": "recorded.json"},
            {"kind": "ActionEvent", "source": "agent", "llm_response_id": "recorded-response",
             "tool_name": "update_cells", "action": {"kind": "MCPToolAction", "tool_call_id": "call-a",
                                                        "data": {"ranges": {"A10:H11": [[2, "Product"]]}}}},
            {"kind": "ActionEvent", "source": "agent", "llm_response_id": "recorded-response",
             "tool_name": "check_cells", "tool_call_id": "call-b", "action": {"kind": "MCPToolAction"}},
            {"kind": "ObservationEvent", "observation": {"kind": "MCPToolObservation", "tool_call_id": "call-a",
                                                         "content": [{"text": "updated A10:H11"}], "is_error": False}},
            {"kind": "ObservationEvent", "tool_call_id": "call-b", "observation": {
                "content": [{"text": "blank rows remain"}], "is_error": False}},
            {"kind": "ActionEvent", "source": "agent", "llm_response_id": "finish-response",
             "action": {"kind": "FinishAction", "message": "Done"}},
        ],
        "model_calls": [{
            "response_id": "recorded-response",
            "request_messages": [{"role": "user", "content": f"message-{index}"} for index in range(18)],
            "request_payload": {"tools": [
                {"kind": "MCPToolDefinition", "title": "update_cells", "mcp_tool": {
                    "name": "update_cells", "inputSchema": {"properties": {"ranges": {"type": "object"}}}}},
                {"type": "function", "function": {"name": "check_cells", "parameters": {"type": "object"}}},
                {"kind": "TerminalTool", "title": "terminal"},
            ]},
            "response_message": {"role": "assistant", "tool_calls": [
                {"id": "call-a", "function": {"name": "update_cells", "arguments": '{"ranges":{"A10:H11":[]}}'}},
                {"id": "call-b", "function": {"name": "check_cells", "arguments": "{}"}},
            ]},
            "request_evidence": "sdk_completion_log", "response_evidence": "sdk_completion_log",
            "source_ref": "completion.json",
        }, {
            "response_id": "auxiliary-response", "request_messages": [{"role": "user", "content": "auxiliary request"}],
            "response_message": {"role": "assistant", "content": "auxiliary response"},
            "request_evidence": "sdk_completion_log", "response_evidence": "sdk_completion_log",
        }],
    }


class OpenHandsTraceProjectionTests(unittest.TestCase):
    def test_timeline_keeps_middle_actions_and_final_event_with_bounded_output(self) -> None:
        trace = _trace()
        trace["events"] = [{"kind": "MessageEvent", "source": "user", "llm_message": {
            "content": "large message " * 2000}} for _ in range(60)]
        trace["events"][30] = _trace()["events"][2]
        trace["events"][-1] = _trace()["events"][-1]
        output = render_trace(trace, char_limit=12000)
        self.assertLessEqual(len(output), 12000)
        self.assertEqual(len(re.findall(r"^\[\d+\]", output, re.MULTILINE)), 60)
        self.assertIn("A10:H11", output)
        self.assertIn("[59] ActionEvent", output)
        self.assertNotIn("output truncated", output)

    def test_observation_selection_includes_shared_response_actions_and_nested_pairs(self) -> None:
        events, calls = select_evidence(_trace(), [4], [])
        self.assertEqual([index for index, _ in events], [2, 3, 4, 5])
        self.assertEqual([call["response_id"] for call in calls], ["recorded-response"])
        output = render_trace(_trace(), event_indices=[4])
        projection = next(json.loads(line) for line in output.splitlines() if '"request_messages_count"' in line)
        self.assertEqual(projection["request_messages_count"], 18)
        self.assertEqual(projection["request_messages_shown"], 4)
        self.assertEqual(projection["request_messages"][0]["content"], "message-14")
        self.assertEqual(projection["available_tool_count"], 3)
        self.assertEqual(projection["used_tool_names"], ["check_cells", "update_cells"])
        self.assertEqual(len(projection["used_tool_schemas"]), 2)
        self.assertIn("A10:H11", json.dumps(projection["response_message"]))

    def test_auxiliary_call_is_visible_without_event_and_missing_request_is_explicit(self) -> None:
        events, calls = select_evidence(_trace(), [], ["auxiliary-response"])
        self.assertEqual(events, [])
        self.assertEqual(len(calls), 1)
        summary = render_trace(_trace())
        self.assertIn('"model_calls_without_events": ["auxiliary-response"]', summary)
        output = render_trace(_trace(), event_indices=[6])
        projection = next(json.loads(line) for line in output.splitlines() if '"request_messages_count"' in line)
        self.assertEqual(projection["request_messages_count"], 0)
        self.assertEqual(projection["request_evidence"], "not_recorded")
        self.assertEqual(projection["response_evidence"], "trace_events")

    def test_large_details_shrink_previews_without_losing_request_response_or_pairs(self) -> None:
        trace = _trace()
        call = trace["model_calls"][0]
        call["source_ref"] = "/root/" + "nested_artifact_path/" * 40 + "completion.json"
        for message in call["request_messages"]:
            message["content"] = "long request history " * 2000
        call["response_message"]["reasoning_content"] = "long reasoning " * 2000
        for event in trace["events"][2:6]:
            event["reasoning_content"] = "duplicate reasoning " * 2000
        for event in trace["events"][4:6]:
            event["observation"]["content"][0]["text"] = "long tool output " * 2000
        output = render_trace(trace, response_ids=["recorded-response"], field_limit=6000, char_limit=12000)
        self.assertLessEqual(len(output), 12000)
        self.assertNotIn("output truncated", output)
        records = [json.loads(line) for line in output.splitlines() if line.startswith('{"event_index"')]
        self.assertEqual([record["event_index"] for record in records], [2, 3, 4, 5])
        projection = next(json.loads(line) for line in output.splitlines() if '"request_messages_count"' in line)
        self.assertEqual(projection["request_messages_count"], 18)
        self.assertEqual(projection["source_ref"], call["source_ref"])
        self.assertIn("response_message", projection)
        self.assertEqual(len(projection["used_tool_schemas"]), 2)

    def test_invalid_selection_fails_instead_of_returning_empty_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "not recorded"):
            select_evidence(_trace(), [], ["typo-response"])
        with self.assertRaisesRegex(ValueError, "outside"):
            select_evidence(_trace(), [100], [])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.json"
            path.write_text(json.dumps(_trace()["events"]))
            result = subprocess.run(["python3", str(REPO_ROOT / "failure_analysis/print_openhands_trace.py"),
                                     str(path), "--event-index", "100"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("outside", result.stderr)


class OpenHandsAnalysisPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from failure_analysis import run_analysis
        cls.runner = run_analysis
        cls.config = yaml.safe_load((REPO_ROOT / "failure_analysis/analysis_config_openhands.yaml").read_text())

    def test_real_observation_formatter_reports_current_budget_and_submission_reminder(self) -> None:
        from minisweagent.models.utils.actions_text import format_observation_messages

        model = Mock()
        model.get_template_vars.return_value = {}
        agent = self.runner.DefaultAgent(model, self.runner.LocalEnvironment(),
                                         **self.runner._agent_config_from_analysis_config(self.config))
        for count, urgent in ((1, False), (27, True), (29, True)):
            with self.subTest(count=count):
                agent.n_calls = count
                content = format_observation_messages(
                    [{"output": "evidence", "returncode": 0, "exception_info": ""}],
                    observation_template=self.config["model"]["observation_template"],
                    template_vars=agent.get_template_vars(),
                )[0]["content"]
                self.assertIn(f"Analysis calls used: {count}. Remaining: {30 - count}.", content)
                self.assertEqual("Stop optional reads" in content, urgent)

    def test_rendered_prompt_commands_work_with_recorded_requests(self) -> None:
        from jinja2 import StrictUndefined, Template

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / "trace.json"
            trace_path.write_text(json.dumps(_trace()))
            source = root / "candidate"
            (source / "prompts").mkdir(parents=True)
            (source / "agent.py").write_text("def build_agent(): pass\n")
            (source / "config.json").write_text('{"task_id":"fixture"}')
            (source / "prompts/system.md").write_text("Complete the task.")
            variables = {
                "step_limit": 30, "instance_id": "fixture", "failure_category": "unresolved",
                "task_description": "Fixture task", "agent_source_dir": str(source),
                "agent_source_root": str(root), "traj_path": str(trace_path), "raw_traj_path": str(trace_path),
                "manifest_path": str(root / "manifest.json"), "htir_path": str(root / "htir.json"),
            }
            for key in ("system_template", "instance_template"):
                prompt = Template(self.config["agent"][key], undefined=StrictUndefined).render(**variables)
                self.assertIn("diagnosis", prompt)
                if key != "instance_template":
                    continue
                commands = re.findall(r"`(python3 failure_analysis/print_openhands_trace.py [^`]+)`", prompt)
                self.assertEqual(len(commands), 3)
                for command in commands:
                    command = command.replace("RESPONSE_ID", "recorded-response").replace("EVENT_INDEX", "4")
                    result = subprocess.run(command, shell=True, cwd=REPO_ROOT, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertLessEqual(len(result.stdout), 12000)
                    if "--response-id" in command or "--event-index" in command:
                        self.assertIn('"request_messages_count": 18', result.stdout)
                        self.assertIn('"response_message"', result.stdout)


if __name__ == "__main__":
    unittest.main()

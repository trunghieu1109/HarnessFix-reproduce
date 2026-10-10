from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from task_agent.openhands_agent.runtime_logging import install_sdk_generation_config_fix, install_sdk_logging_fix


class RegistryCallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        class Telemetry:
            def __init__(self):
                self._log_completions_callback = None
                self._stats_update_callback = None

            def set_log_completions_callback(self, callback):
                self._log_completions_callback = callback

            def set_stats_update_callback(self, callback):
                self._stats_update_callback = callback

        class Registry:
            def __init__(self):
                self._metrics_ids = set()

            def add(self, llm):
                if id(llm.metrics) in self._metrics_ids:
                    llm.metrics = object()
                self._metrics_ids.add(id(llm.metrics))
                # RegistryEvent revalidates this existing LLM in SDK 1.19.
                llm.telemetry = Telemetry()

        module = ModuleType("openhands.sdk.llm.llm_registry")
        module.LLMRegistry = Registry
        context = patch.dict(sys.modules, {module.__name__: module})
        context.start()
        self.addCleanup(context.stop)
        self.Telemetry = Telemetry
        self.Registry = Registry

    def test_registration_preserves_both_callbacks_and_install_is_idempotent(self):
        telemetry = self.Telemetry()
        completion = lambda filename, data: None
        stats = lambda: None
        telemetry.set_log_completions_callback(completion)
        telemetry.set_stats_update_callback(stats)
        llm = SimpleNamespace(metrics=object(), telemetry=telemetry)
        install_sdk_logging_fix()
        wrapped = self.Registry.add
        install_sdk_logging_fix()
        self.assertIs(self.Registry.add, wrapped)
        self.Registry().add(llm)
        self.assertIsNot(llm.telemetry, telemetry)
        self.assertIs(llm.telemetry._log_completions_callback, completion)
        self.assertIs(llm.telemetry._stats_update_callback, stats)

    def test_copied_llm_keeps_independent_metrics_without_parent_callback(self):
        install_sdk_logging_fix()
        registry = self.Registry()
        parent = SimpleNamespace(metrics=object(), telemetry=self.Telemetry())
        parent.telemetry.set_log_completions_callback(lambda filename, data: None)
        registry.add(parent)
        copied = SimpleNamespace(metrics=parent.metrics, telemetry=parent.telemetry)
        registry.add(copied)
        self.assertIsNot(copied.metrics, parent.metrics)
        self.assertIsNone(copied.telemetry._log_completions_callback)
        self.assertIsNotNone(parent.telemetry._log_completions_callback)


@unittest.skipUnless(importlib.util.find_spec("openhands"), "Run with the Better Harness SDK environment")
class RealSDKLoggingTests(unittest.TestCase):
    def test_qwen_thinking_off_preserves_temperature_in_sdk_requests(self):
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "OPENHANDS_SUPPRESS_BANNER": "1"}):
            from litellm import ModelResponse
            from openhands.sdk import LLM, Message, TextContent
            from openhands.sdk.llm import llm as llm_module

        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        llm = LLM(model="openai/Qwen/Qwen3.5-9B", api_key="offline-test-key",
                  temperature=0.2, reasoning_effort=None, litellm_extra_body=extra_body,
                  max_input_tokens=120000, max_output_tokens=8192,
                  input_cost_per_token=0, output_cost_per_token=0)
        response = ModelResponse(model=llm.model, choices=[{
            "index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop",
        }], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        with patch.object(llm_module, "select_chat_options", llm_module.select_chat_options):
            install_sdk_generation_config_fix()
            wrapped = llm_module.select_chat_options
            install_sdk_generation_config_fix()
            self.assertIs(llm_module.select_chat_options, wrapped)
            with patch.object(LLM, "_transport_call", return_value=response) as transport:
                llm.completion([Message(role="user", content=[TextContent(text="Reply OK")])])
            kwargs = transport.call_args.kwargs
            self.assertEqual(kwargs["temperature"], 0.2)
            self.assertEqual(kwargs["extra_body"], extra_body)
            self.assertEqual(kwargs["max_completion_tokens"], 8192)
            self.assertNotIn("reasoning_effort", kwargs)
            self.assertEqual(wrapped(llm, {"temperature": 0.4}, False)["temperature"], 0.4)
            llm.litellm_extra_body = {}
            self.assertNotIn("temperature", wrapped(llm, {}, False))

    def test_packed_runtime_keeps_streaming_through_mcp_initialization(self):
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "OPENHANDS_SUPPRESS_BANNER": "1"}):
            from litellm import ModelResponse
            from openhands.sdk import Conversation, LLM, Message, TextContent
            from openhands.sdk.workspace import LocalWorkspace
            from openhands.agent_server.event_service import EventService
        from task_agent.openhands_agent.bridge import pack_candidate

        logging.getLogger("openhands").setLevel(logging.ERROR)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "agent.py").write_text(
                'from openhands.sdk import Agent\n'
                'def build_agent(base_dir, llm):\n'
                '    return Agent(llm=llm, tools=[], include_default_tools=[], '
                'system_prompt="Offline test.", mcp_config={"mcpServers": {}})\n'
            )
            launcher = pack_candidate(candidate, root / "packed.py")
            scope = {}
            exec(compile(launcher.read_text(), str(launcher), "exec"), scope)
            llm = LLM(model="openai/offline-test", api_key="test-key", max_input_tokens=32768,
                      max_output_tokens=128, input_cost_per_token=0, output_cost_per_token=0)
            agent = scope["build_agent"](str(root / "example0_rollout0"), llm)
            conv = Conversation(agent=agent, workspace=LocalWorkspace(working_dir=str(root)),
                                persistence_dir=str(root / "conversations"), visualizer=None)
            self.addCleanup(conv.close)
            received = []
            EventService._setup_llm_log_streaming(SimpleNamespace(_emit_event_from_thread=received.append), conv.agent)
            original = conv.agent.llm.telemetry._log_completions_callback
            with patch("openhands.sdk.agent.base.create_mcp_tools", return_value=[]):
                conv.send_message("Offline request.")
            self.assertIs(conv.agent.llm.telemetry._log_completions_callback, original)
            response = ModelResponse(id="offline-response", model=llm.model, choices=[{
                "message": {"role": "assistant", "content": "Offline answer."},
            }], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
            with patch.object(LLM, "_transport_call", return_value=response):
                conv.agent.llm.completion([Message(role="user", content=[TextContent(text="Offline request.")])])
            self.assertEqual(len(received), 1)
            payload = json.loads(received[0].log_data)
            self.assertIn("Offline request.", json.dumps(payload["messages"]))
            self.assertEqual(payload["response"]["choices"][0]["message"]["content"], "Offline answer.")


if __name__ == "__main__":
    unittest.main()

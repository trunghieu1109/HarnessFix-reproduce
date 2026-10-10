from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


@unittest.skipUnless(importlib.util.find_spec("litellm"), "LiteLLM is not installed")
class OpenHandsModelConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from task_agent.openhands_agent import model_config_bridge
        cls.bridge = model_config_bridge

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "configs").mkdir()
        entries = []
        for alias in ("gemini", "qwen"):
            prefix = alias.upper()
            entries.append({
                "name": alias,
                "model": "${" + prefix + "_MODEL}",
                "api_base": "${" + prefix + "_API_BASE}",
                "api_key": "${" + prefix + "_API_KEY}",
                "temperature": "${" + prefix + "_TEMPERATURE}",
                "max_input_tokens": "${" + prefix + "_CONTEXT_WINDOW}",
                "max_output_tokens": "${" + prefix + "_MAX_OUTPUT_TOKENS}",
                "reasoning_effort": "${" + prefix + "_REASONING_EFFORT}",
            })
        (self.root / "configs/models.yaml").write_text(yaml.safe_dump({"models": entries}))
        (self.root / ".env").write_text(
            "GEMINI_MODEL=gemini/gemini-test-model\n"
            "GEMINI_API_BASE=https://gemini.example\n"
            "GEMINI_API_KEY=gemini-test-key\n"
            "GEMINI_TEMPERATURE=1.0\n"
            "GEMINI_CONTEXT_WINDOW=250000\n"
            "GEMINI_MAX_OUTPUT_TOKENS=8192\n"
            "GEMINI_REASONING_EFFORT=medium\n"
            "QWEN_MODEL=openai/qwen-test-model\n"
            "QWEN_API_BASE=http://qwen.example/v1\n"
            "QWEN_API_KEY=qwen-test-key\n"
            "QWEN_TEMPERATURE=0.6\n"
            "QWEN_CONTEXT_WINDOW=120000\n"
            "QWEN_MAX_OUTPUT_TOKENS=4096\n"
            "QWEN_REASONING_EFFORT=low\n"
        )

    def test_models_resolve_dotenv_values_with_correct_types_and_provider(self) -> None:
        for alias, expected_provider, expected_temperature, expected_output, expected_context in (
            ("gemini", "gemini", 1.0, 8192, 250000),
            ("qwen", "openai", 0.6, 4096, 120000),
        ):
            with self.subTest(alias=alias), patch.dict(os.environ, {
                "BETTER_HARNESS_ROOT": str(self.root),
                "QWEN_API_BASE": "http://stale.example/v1",
            }), patch.object(self.bridge.litellm, "register_model") as register:
                model = self.bridge.load_model(alias)
                self.assertNotIn("${", model["model"])
                self.assertEqual(model["temperature"], expected_temperature)
                self.assertIsInstance(model["temperature"], float)
                self.assertEqual(model["max_output_tokens"], expected_output)
                self.assertIsInstance(model["max_output_tokens"], int)
                self.assertIsInstance(model["max_input_tokens"], int)
                self.assertEqual(model["max_input_tokens"], expected_context)
                registered = register.call_args.args[0][model["model"]]
                self.assertEqual(registered["litellm_provider"], expected_provider)
                self.assertEqual(registered["max_input_tokens"], expected_context)
                self.assertEqual(registered["max_output_tokens"], expected_output)
                if alias == "qwen":
                    self.assertEqual(model["api_base"], "http://qwen.example/v1")

    def test_generation_settings_reach_analysis_and_repair_kwargs(self) -> None:
        with patch.dict(os.environ, {
            "BETTER_HARNESS_ROOT": str(self.root), "HARNESSFIX_MODEL_ALIAS": "qwen",
        }), patch.object(self.bridge.litellm, "register_model"):
            kwargs = self.bridge.selected_model_kwargs()
        self.assertEqual(kwargs, {
            "api_base": "http://qwen.example/v1",
            "api_key": "qwen-test-key",
            "reasoning_effort": "low",
            "temperature": 0.6,
            "max_tokens": 4096,
        })

    def test_invalid_token_limit_is_rejected(self) -> None:
        path = self.root / ".env"
        path.write_text(path.read_text().replace("QWEN_MAX_OUTPUT_TOKENS=4096", "QWEN_MAX_OUTPUT_TOKENS=invalid"))
        with patch.dict(os.environ, {"BETTER_HARNESS_ROOT": str(self.root)}):
            with self.assertRaises(ValueError):
                self.bridge.load_model("qwen")

    def test_qwen_thinking_switch_reaches_analysis_and_repair(self) -> None:
        path = self.root / "configs/models.yaml"
        models = yaml.safe_load(path.read_text())
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        models["models"][1].update(reasoning_effort=None, extra_body=extra_body)
        path.write_text(yaml.safe_dump(models))
        with patch.dict(os.environ, {
            "BETTER_HARNESS_ROOT": str(self.root), "HARNESSFIX_MODEL_ALIAS": "qwen",
        }), patch.object(self.bridge.litellm, "register_model"):
            kwargs = self.bridge.selected_model_kwargs()
        self.assertIsNone(kwargs["reasoning_effort"])
        self.assertEqual(kwargs["extra_body"], extra_body)
        self.assertEqual(kwargs["temperature"], 0.6)


if __name__ == "__main__":
    unittest.main()

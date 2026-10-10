from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from dotenv import dotenv_values

from failure_analysis.analysis_records import analysis_record_succeeded
from scripts.configure_models import configure
import run_pipeline_swe as swe_pipeline


class AnalysisModelConfigTests(unittest.TestCase):
    def test_thinking_settings_sync_without_leaking_between_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            better = Path(tmp) / "better"
            for directory in (repo / "task_agent", repo / "failure_analysis", repo / "enhancement_implementation", better / "src", better / "configs"):
                directory.mkdir(parents=True)
            (better / "src/collect.py").write_text("")
            (repo / "task_agent/model_registry.json").write_text("{}")
            config_paths = [repo / relative for relative in (
                "failure_analysis/analysis_config_swe.yaml", "enhancement_implementation/config_swe.yaml",
            )]
            for path in config_paths:
                path.write_text("agent:\n  system_template: keep this prompt\nmodel:\n  model_kwargs:\n    temperature: 1.0\n")
            for root in (repo, better):
                (root / ".env").write_text("QWEN_API_KEY=offline-qwen-key\nGEMINI_API_KEY=offline-gemini-key\n")
            config_path = repo / "model.yaml"
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            common = {"max_input_tokens": 120000, "max_output_tokens": 4096, "temperature": 0.2}
            config_path.write_text(yaml.safe_dump(common | {
                "name": "qwen", "model": "openai/Qwen/Qwen3.5-9B",
                "api_base": "https://qwen.example/v1", "api_key_env": "QWEN_API_KEY",
                "reasoning_effort": None, "extra_body": extra_body,
            }))
            with patch.dict(os.environ, {"QWEN_API_KEY": "offline-qwen-key", "GEMINI_API_KEY": "offline-gemini-key"}), contextlib.redirect_stdout(io.StringIO()):
                configure(repo, better, config_path)
                registry = json.loads((repo / "task_agent/model_registry.json").read_text())
                self.assertEqual(registry["openai/Qwen/Qwen3.5-9B"]["model_kwargs_override"]["extra_body"], extra_body)
                for path in config_paths:
                    self.assertEqual(yaml.safe_load(path.read_text())["model"]["model_kwargs"]["extra_body"], extra_body)
                config_path.write_text(yaml.safe_dump(common | {
                    "name": "gemini", "model": "openai/ag/gemini-3.1-pro-low",
                    "api_base": "https://gemini.example/v1", "api_key_env": "GEMINI_API_KEY",
                    "reasoning_effort": "low",
                }))
                configure(repo, better, config_path, analysis_only=True)
            aliases = {entry["name"]: entry for entry in yaml.safe_load((better / "configs/models.yaml").read_text())["models"]}
            self.assertEqual(aliases["qwen"]["extra_body"], extra_body)
            self.assertEqual(aliases["gemini"]["temperature"], 0.2)
            self.assertEqual(aliases["gemini"]["reasoning_effort"], "low")
            self.assertNotIn("extra_body", aliases["gemini"])
            for path in config_paths:
                parsed = yaml.safe_load(path.read_text())
                self.assertEqual(parsed["agent"]["system_template"], "keep this prompt")
                self.assertEqual(parsed["model"]["model_kwargs"]["reasoning_effort"], "low")
                self.assertEqual(parsed["model"]["model_kwargs"]["temperature"], 0.2)
                self.assertNotIn("extra_body", parsed["model"]["model_kwargs"])

    def test_analysis_gateway_preserves_task_settings_and_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            better = Path(tmp) / "better"
            for directory in (repo / "task_agent", repo / "failure_analysis", repo / "enhancement_implementation", better / "src", better / "configs"):
                directory.mkdir(parents=True)
            (better / "src/collect.py").write_text("")
            (repo / "task_agent/model_registry.json").write_text("{}")
            for name in ("failure_analysis/analysis_config_swe.yaml", "enhancement_implementation/config_swe.yaml"):
                (repo / name).write_text("agent:\n  system_template: keep this prompt\nmodel:\n  model_kwargs:\n    temperature: 0.2\n")
            (better / "configs/models.yaml").write_text(yaml.safe_dump({"models": [{"name": "qwen", "model": "openai/qwen"}]}))
            original = (
                "HARNESSFIX_TASK_MODEL=openai/qwen\nHARNESSFIX_MODEL_NAME=qwen\n"
                "OPENAI_API_BASE=http://qwen.example/v1\nOPENAI_API_KEY=qwen-key\n"
                "VLLM_MODEL=openai/qwen\nGEMINI_API_KEY=gemini-key\n"
            )
            for root in (repo, better):
                (root / ".env").write_text(original)
            config = repo / "gateway.yaml"
            config.write_text(yaml.safe_dump({"name": "gemini-api", "model": "openai/ag/gemini-test", "api_base": "https://gateway.example/v1",
                "api_key_env": "GEMINI_API_KEY", "max_input_tokens": 250000, "max_output_tokens": 8192,
                "temperature": 1.0, "reasoning_effort": None}))
            with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}), contextlib.redirect_stdout(io.StringIO()):
                configure(repo, better, config, analysis_only=True)
            for root in (repo, better):
                current = dotenv_values(root / ".env")
                for key, value in dotenv_values(stream=io.StringIO(original)).items():
                    self.assertEqual(current[key], value)
                self.assertEqual(current["HARNESSFIX_ANALYSIS_MODEL"], "openai/ag/gemini-test")
            registry = json.loads((repo / "task_agent/model_registry.json").read_text())
            self.assertEqual(registry["openai/ag/gemini-test"]["api_key_env"], "GEMINI_API_KEY")
            self.assertEqual(registry["openai/ag/gemini-test"]["model_kwargs_override"]["api_base"], "https://gateway.example/v1")
            aliases = yaml.safe_load((better / "configs/models.yaml").read_text())["models"]
            self.assertEqual(aliases[0], {"name": "qwen", "model": "openai/qwen"})
            self.assertEqual(aliases[1]["api_key"], "${GEMINI_API_KEY}")


@unittest.skipUnless(importlib.util.find_spec("litellm"), "LiteLLM is not installed")
class AnalysisExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from failure_analysis import run_analysis, aggregate_results
            from task_agent.openhands_agent import model_config_bridge
        cls.runner = run_analysis
        cls.aggregate = aggregate_results
        cls.bridge = model_config_bridge

    def test_cli_step_limit_overrides_both_agent_and_top_level_config(self):
        config = {"step_limit": 30, "agent": {"step_limit": 30}}
        record = {"instance_id": "one", "agent_design_issue": "Inspect verification."}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(sys, "argv", [
                "run_analysis.py", "--mode", "openhands", "--model", "offline", "--step-limit", "10",
                "--output-file", str(Path(tmp) / "analysis.jsonl"),
            ]),
            patch.object(self.runner, "RESULTS_DIR", Path(tmp)),
            patch.object(self.runner, "_ALL_RESULTS_PATH_OVERRIDE", None),
            patch.object(self.runner, "load_config", return_value=config),
            patch.object(self.runner, "load_failed_instances", return_value={"one": "unresolved"}),
            patch.object(self.runner, "load_completed_ids", return_value=set()),
            patch.object(self.runner, "_openhands_run_analysis", return_value=record) as analyze,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.runner.main()
        passed_config = analyze.call_args.args[3]
        self.assertEqual(passed_config["step_limit"], 10)
        self.assertEqual(self.runner._agent_config_from_analysis_config(passed_config)["step_limit"], 10)

    def test_failed_records_are_retryable_and_excluded_from_aggregate(self):
        rows = [
            {"instance_id": "valid", "agent_design_issue": "Inspect verification."},
            {"instance_id": "api-failed", "_analysis_fallback": True, "exit_status": "analysis_agent_exception"},
            {"instance_id": "parse-failed", "_parse_error": True},
            {"instance_id": "legacy-failed", "exit_status": "analysis_failed"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with patch.object(self.runner, "_ALL_RESULTS_PATH_OVERRIDE", path):
                self.assertEqual(self.runner.load_completed_ids(), {"valid"})
            self.assertEqual(swe_pipeline.completed_analysis_ids(path), {"valid"})
            self.assertEqual(self.aggregate.load_results(path), [rows[0]])

    def test_cli_counts_saved_fallback_as_failed_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.jsonl"
            record = {"instance_id": "one", "_analysis_fallback": True, "exit_status": "analysis_agent_exception", "api_calls": 1}
            stdout = io.StringIO()
            with (
                patch.object(sys, "argv", ["run_analysis.py", "--mode", "openhands", "--model", "offline", "--output-file", str(path)]),
                patch.object(self.runner, "RESULTS_DIR", Path(tmp)),
                patch.object(self.runner, "_ALL_RESULTS_PATH_OVERRIDE", None),
                patch.object(self.runner, "load_failed_instances", return_value={"one": "unresolved"}),
                patch.object(self.runner, "load_completed_ids", return_value=set()),
                patch.object(self.runner, "_openhands_run_analysis", return_value=record),
                contextlib.redirect_stdout(stdout),
            ):
                with self.assertRaises(SystemExit) as raised:
                    self.runner.main()
            self.assertEqual(raised.exception.code, 1)
            self.assertIn("0 succeeded, 1 failed", stdout.getvalue())
            self.assertTrue(json.loads(path.read_text())["_analysis_fallback"])

    def test_gateway_uses_chat_completions_and_preserves_model_namespace(self):
        import httpx
        import litellm
        from openai import OpenAI

        captured = []
        def transport(request):
            captured.append(request)
            return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "created": 0,
                "model": "ag/gemini-test", "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        with httpx.Client(transport=httpx.MockTransport(transport)) as http:
            client = OpenAI(api_key="offline-test-key", base_url="https://gateway.example/v1", http_client=http)
            with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
                response = litellm.completion(model="openai/ag/gemini-test", api_base="https://gateway.example/v1", api_key="offline-test-key",
                    messages=[{"role": "user", "content": "Reply OK"}], max_tokens=8,
                    temperature=0.2, reasoning_effort="low", drop_params=True, client=client)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].url.path, "/v1/chat/completions")
        body = json.loads(captured[0].content)
        self.assertEqual(body["model"], "ag/gemini-test")
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(response.choices[0].message.content, "OK")

    def test_gateway_credential_is_resolved_without_changing_qwen_globals(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"GEMINI_API_KEY": "offline-gemini-key", "OPENAI_API_KEY": "qwen-key"}):
            registry = Path(tmp) / "registry.json"
            registry.write_text(json.dumps({"openai/ag/gemini-3.1-pro-low": {
                "api_key_env": "GEMINI_API_KEY", "model_kwargs_override": {
                    "api_base": "https://gateway.example/v1", "temperature": 1.0,
                },
            }}))
            with patch.object(self.bridge, "MODEL_REGISTRY_PATH", registry):
                kwargs = self.bridge.configured_connection_kwargs("openai/ag/gemini-3.1-pro-low", {"api_base": "http://qwen.example/v1", "temperature": 0.2})
            self.assertEqual(kwargs["api_key"], "offline-gemini-key")
            self.assertEqual(kwargs["temperature"], 1.0)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "qwen-key")
            self.assertNotEqual(kwargs["api_base"], "http://qwen.example/v1")


if __name__ == "__main__":
    unittest.main()

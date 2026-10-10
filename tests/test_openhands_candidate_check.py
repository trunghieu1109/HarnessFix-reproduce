from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from task_agent.openhands_agent.candidate_check import _check_packed_candidate, _run_check_process, check_candidate


class FakeAgent:
    pass


class FakeLLM:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def completion(self, *args, **kwargs):
        raise AssertionError("A model request escaped the check")

    def responses(self, *args, **kwargs):
        raise AssertionError("A model request escaped the check")


class CandidateCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidate = self.root / "candidate"
        self.candidate.mkdir()
        (self.candidate / "agent.py").write_text("def build_agent(base_dir, llm):\n    return llm\n")
        sdk = types.ModuleType("openhands.sdk")
        sdk.Agent, sdk.LLM = FakeAgent, FakeLLM
        self.sdk = sdk

    def worker_check(self, source, model_settings=None):
        packed = self.root / "packed.py"
        packed.write_text(source)
        status = self.root / "status.json"
        with patch.dict(sys.modules, {"openhands.sdk": self.sdk}):
            result = _check_packed_candidate(packed, self.root, status, model_settings)
        self.assertEqual(result, json.loads(status.read_text()))
        return result

    def test_success_imports_and_builds_sdk_agent_without_model_calls(self):
        report = self.worker_check("from openhands.sdk import Agent\ndef build_agent(base_dir, llm):\n    return Agent()\n")
        self.assertTrue(report["passed"])
        self.assertEqual(report["stage"], "build_agent")
        self.assertEqual(report["llm_calls"], 0)

    def test_check_uses_task_model_settings_with_placeholder_connection(self):
        source = (
            "from openhands.sdk import Agent\n"
            "def build_agent(base_dir, llm):\n"
            "    assert llm.model == 'openai/Qwen/test'\n"
            "    assert llm.temperature == 0.2 and llm.max_output_tokens == 8192\n"
            "    assert llm.reasoning_effort == 'none'\n"
            "    assert llm.extra_body['chat_template_kwargs']['enable_thinking'] is False\n"
            "    assert llm.api_key == 'EMPTY' and llm.api_base == 'http://127.0.0.1:1/v1'\n"
            "    return Agent()\n"
        )
        report = self.worker_check(source, {
            "model": "openai/Qwen/test", "temperature": 0.2, "max_output_tokens": 8192,
            "reasoning_effort": "none", "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "api_key": "must-not-be-used", "api_base": "http://served-model",
        })
        self.assertTrue(report["passed"])

    def test_import_and_build_errors_keep_stage_and_traceback(self):
        for source, stage in (
            ("raise ImportError('Synthetic missing SDK symbol')\n", "import"),
            ("class BrowserToolSet:\n    pass\ndef build_agent(base_dir, llm):\n"
             "    return BrowserToolSet.browser_get_state\n", "build_agent"),
        ):
            with self.subTest(stage=stage):
                report = self.worker_check(source)
                self.assertFalse(report["passed"])
                self.assertEqual(report["stage"], stage)
                self.assertIn(report["exception_type"], report["traceback"])
                self.assertEqual(report["status"], "complete")

    def test_invalid_return_and_model_requests_fail_the_check(self):
        for body, error, blocked in (
            ("return None", "TypeError", 0),
            ("return llm.completion([])", "RuntimeError", 1),
            ("return llm.responses([])", "RuntimeError", 1),
        ):
            with self.subTest(body=body):
                report = self.worker_check(f"def build_agent(base_dir, llm):\n    {body}\n")
                self.assertFalse(report["passed"])
                self.assertEqual(report["exception_type"], error)
                self.assertEqual(report["llm_calls"], 0)
                self.assertEqual(report["llm_calls_blocked"], blocked)

    @staticmethod
    def status_path(command):
        return Path(command[command.index("--status-file") + 1])

    def test_subprocess_uses_better_environment_and_redacts_failure_feedback(self):
        secret = "synthetic-check-secret-value"
        def run(command, **kwargs):
            self.assertEqual(command[:4], ["uv", "run", "--offline", "python"])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertEqual(kwargs["timeout"], 60)
            self.assertEqual(kwargs["env"]["LITELLM_LOCAL_MODEL_COST_MAP"], "True")
            model_settings = json.loads(command[command.index("--model-settings") + 1])
            self.assertEqual(model_settings, {"model": "openai/Qwen/test", "temperature": 0.2})
            self.status_path(command).write_text(json.dumps({
                "passed": False, "status": "complete", "stage": "build_agent", "llm_calls": 0,
                "exception_type": "ValueError", "message": f"Invalid connection: {secret}",
                "traceback": f"Authorization: Bearer {secret}",
            }))
            return subprocess.CompletedProcess(command, 1, "", "")
        output = self.root / "check.json"
        with patch.dict(os.environ, {"SYNTHETIC_API_KEY": secret}), \
                patch("task_agent.openhands_agent.candidate_check._run_check_process", side_effect=run):
            report = check_candidate(better_root=self.root, candidate_dir=self.candidate, output_path=output,
                                     model_settings={"model": "openai/Qwen/test", "temperature": 0.2,
                                                     "api_key": secret, "api_base": "http://served-model"})
        self.assertEqual(report, json.loads(output.read_text()))
        self.assertNotIn(secret, output.read_text())
        self.assertIn("[REDACTED]", report["message"])

    def test_timeout_after_worker_start_is_bounded_candidate_failure(self):
        def run(command, **kwargs):
            self.status_path(command).write_text(json.dumps({"passed": False, "status": "running", "stage": "import"}))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with patch("task_agent.openhands_agent.candidate_check._run_check_process", side_effect=run):
            report = check_candidate(better_root=self.root, candidate_dir=self.candidate,
                                     output_path=self.root / "check.json", timeout=1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["exception_type"], "TimeoutError")
        self.assertEqual(report["stage"], "import")

    def test_worker_startup_or_process_crash_does_not_become_candidate_feedback(self):
        def crash(command, **kwargs):
            self.status_path(command).write_text(json.dumps({"passed": False, "status": "running", "stage": "import"}))
            return subprocess.CompletedProcess(command, 3, "", "Worker died")
        for implementation in (
            lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "Missing SDK package"),
            crash,
        ):
            with self.subTest(implementation=implementation), \
                    patch("task_agent.openhands_agent.candidate_check._run_check_process", side_effect=implementation):
                with self.assertRaisesRegex(RuntimeError, "infrastructure|did not complete"):
                    check_candidate(better_root=self.root, candidate_dir=self.candidate,
                                    output_path=self.root / "check.json")
        self.assertFalse((self.root / "check.json").exists())

    @unittest.skipUnless(os.name == "posix", "Process group cleanup uses POSIX signals")
    def test_timeout_terminates_worker_child_processes(self):
        child_pid = self.root / "child.pid"
        code = (
            "import subprocess, sys, time; from pathlib import Path; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            f"Path({str(child_pid)!r}).write_text(str(child.pid)); time.sleep(30)"
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            _run_check_process([sys.executable, "-c", code], cwd=self.root,
                               env=dict(os.environ), timeout=1)
        pid = int(child_pid.read_text())
        # A terminated child can briefly remain as a zombie until init reaps it.
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            self.assertEqual(stat.read_text().split()[2], "Z")


if __name__ == "__main__":
    unittest.main()

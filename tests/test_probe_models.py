"""Check request settings and distinguish hidden reasoning from zero tokens."""

import argparse
import json
import unittest
from unittest.mock import Mock, patch

import requests

from scripts.probe_models import REPO_ROOT, probe, reasoning_metadata, response_evidence, temperature_fields


class TestModelProbe(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            api_base=None, max_tokens=2048, timeout=5,
            skip_discovery=True, no_auth=False, ngrok_skip_browser_warning=False,
            include_raw_response=False,
        )
        self.session = Mock()
        self.session.headers = {}
        self.response = Mock(ok=True, status_code=200)
        self.response.headers = {"Content-Type": "application/json"}
        self.response.json.return_value = {
            "model": "test-model", "choices": [{
                "message": {"content": "1210"}, "finish_reason": "stop",
            }], "usage": {},
        }
        self.session.post.return_value = self.response

    def call_probe(self, name, env):
        with patch("scripts.probe_models.requests.Session") as session_class:
            session_class.return_value.__enter__.return_value = self.session
            return probe(REPO_ROOT / f"configs/{name}.yaml", env, self.args)

    def test_qwen_request_sends_temperature_and_disables_thinking(self):
        result = self.call_probe("qwen", {"QWEN_API_KEY": "test-secret"})
        body = self.session.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("extra_body", body)
        self.assertNotIn("reasoning_effort", body)
        self.assertTrue(result["completion"]["answer_correct"])
        self.assertNotIn("test-secret", json.dumps(result))

    def test_gemini_request_sends_low_reasoning_and_temperature(self):
        self.call_probe("gemini", {"GEMINI_API_KEY": "test-secret"})
        body = self.session.post.call_args.kwargs["json"]
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["model"], "ag/gemini-3.8-flash-low")

    def test_no_auth_does_not_require_credentials_or_use_netrc(self):
        self.args.no_auth = True
        result = self.call_probe("qwen", {})
        self.assertEqual(result["authentication"], "none")
        self.assertNotIn("Authorization", self.session.headers)
        self.assertFalse(self.session.trust_env)

    def test_wrong_answer_is_recorded_separately_from_connectivity(self):
        self.response.json.return_value["choices"][0]["message"]["content"] = "502"
        result = self.call_probe("qwen", {"QWEN_API_KEY": "test-secret"})
        self.assertTrue(result["completion"]["ok"])
        self.assertFalse(result["completion"]["answer_correct"])

    def test_ngrok_warning_header_and_response_content_type(self):
        self.args.ngrok_skip_browser_warning = True
        result = self.call_probe("gemini", {"GEMINI_API_KEY": "test-secret"})
        self.assertEqual(self.session.headers["ngrok-skip-browser-warning"], "1")
        self.assertEqual(result["completion"]["content_type"], "application/json")

    def test_nested_temperature_includes_zero_and_explicit_null(self):
        body = {"metadata": {"generation_config": {"temperature": 0}},
                "models": [{"maxTemperature": 2}], "temperature": None}
        self.assertEqual(temperature_fields(body), {
            "$.metadata.generation_config.temperature": 0,
            "$.models[0].maxTemperature": 2,
            "$.temperature": None,
        })

    def test_raw_evidence_finds_header_temperature_and_redacts_secrets(self):
        evidence = response_evidence(
            {"metadata": {"samplingTemperature": 0.7}, "debug": "test-secret"},
            {"Set-Cookie": "session=private", "X-Temperature": "0.7"},
            "test-secret", True,
        )
        self.assertEqual(evidence["temperature_fields"], {"$.metadata.samplingTemperature": 0.7})
        self.assertEqual(evidence["header_temperature_fields"], {"headers.X-Temperature": "0.7"})
        self.assertNotIn("test-secret", json.dumps(evidence))
        self.assertNotIn("session=private", json.dumps(evidence))
        self.assertEqual(evidence["raw_json"]["debug"], "<redacted>")

    def test_transport_failure_redacts_credential(self):
        self.session.post.side_effect = requests.ConnectionError("failed: test-secret")
        result = self.call_probe("qwen", {"QWEN_API_KEY": "test-secret"})
        self.assertFalse(result["completion"]["ok"])
        self.assertIn("<redacted>", result["completion"]["error"])
        self.assertNotIn("test-secret", json.dumps(result))

    def test_absent_reasoning_is_distinct_from_explicit_zero(self):
        self.assertEqual(reasoning_metadata({}, {})["verdict"], "not_exposed")
        zero = {"completion_tokens_details": {"reasoning_tokens": 0}}
        self.assertEqual(reasoning_metadata({}, zero)["verdict"], "zero_reported_tokens")

    def test_positive_reasoning_detected_without_returning_reasoning_text(self):
        for message, usage in (
            ({"reasoning_content": "private-thought-marker"}, {}),
            ({"content": "<think>private-thought-marker</think>1210"}, {}),
            ({}, {"completion_tokens_details": {"reasoning_tokens": 42}}),
        ):
            with self.subTest(message=message, usage=usage):
                metadata = reasoning_metadata(message, usage)
                self.assertTrue(metadata["observed"])
                self.assertNotIn("private-thought-marker", json.dumps(metadata))


if __name__ == "__main__":
    unittest.main()

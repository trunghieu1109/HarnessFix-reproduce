from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from jinja2 import Environment, StrictUndefined


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "task_agent/mini-swe-agent/src"))

from failure_analysis.artifact_sanitizer import sanitize_for_prompt
from failure_analysis.print_openhands_trace import render_trace
from failure_analysis.prompt_safety import PromptSafeAgent, PromptSafeLocalEnvironment
from failure_analysis.secret_redaction import REDACTED, known_secret_values, redact_secrets
from minisweagent.exceptions import Submitted


class SecretRedactionTests(unittest.TestCase):
    def test_shell_directory_variables_preserve_artifact_paths_and_credentials_stay_redacted(self):
        secret = "synthetic-directory-test-model-key"
        directory_vars = {"PWD": str(REPO_ROOT), "OLDPWD": str(REPO_ROOT.parent)}
        path = str(REPO_ROOT / "artifacts" / "trace.json")
        with patch.dict(os.environ, directory_vars | {"OPENAI_API_KEY": secret}):
            secrets = known_secret_values()
            cleaned = redact_secrets({
                "environment": directory_vars,
                "source_ref": path,
                "content": f"Read {path}; api_key=\"{secret}\"",
                "arguments": json.dumps({"pwd": "synthetic-login-password", "path": path}),
            })
            sanitized = sanitize_for_prompt({"events": [], "source_ref": path})
        self.assertTrue(set(directory_vars.values()).isdisjoint(secrets))
        self.assertEqual(cleaned["environment"], directory_vars)
        self.assertEqual(cleaned["source_ref"], path)
        self.assertIn(path, cleaned["content"])
        self.assertNotIn(secret, json.dumps(cleaned))
        self.assertEqual(json.loads(cleaned["arguments"]), {"pwd": REDACTED, "path": path})
        self.assertEqual(sanitized["source_ref"], path)

    def test_nested_json_and_free_text_credentials_are_redacted_without_losing_evidence(self):
        secret = "synthetic-model-key-123"
        original = {
            "api_key": secret,
            "arguments": json.dumps({"password": "synthetic-login-password", "range": "A10:H11"}),
            "content": f"OPENAI_API_KEY='{secret}' Authorization: Bearer synthetic-bearer-token",
            "usage": {"completion_tokens": 12, "total_tokens": 24},
            "tool_call_id": "call-1", "source_ref": "trace.events[30]",
            "api_key_env": "OPENAI_API_KEY",
        }
        cleaned = redact_secrets(original, secrets={secret})
        text = json.dumps(cleaned)
        for value in (secret, "synthetic-login-password", "synthetic-bearer-token"):
            self.assertNotIn(value, text)
        self.assertEqual(json.loads(cleaned["arguments"])["range"], "A10:H11")
        self.assertEqual(cleaned["usage"], original["usage"])
        self.assertEqual(cleaned["tool_call_id"], "call-1")
        self.assertEqual(cleaned["api_key_env"], "OPENAI_API_KEY")
        self.assertEqual(original["api_key"], secret)

    def test_sanitized_trace_and_cli_projection_keep_model_evidence_but_mask_keys(self):
        trace = {"events": [{"kind": "MessageEvent", "source": "user", "llm_message": {
            "content": 'api_key="synthetic-trace-secret"; inspect inventory',
        }}], "model_calls": [{
            "response_id": "response-1", "source_ref": "completion.json",
            "request_evidence": "sdk_completion_log", "response_evidence": "sdk_completion_log",
            "request_messages": [{"role": "user", "content": "Inspect inventory"}],
            "request_payload": {},
            "response_message": {"role": "assistant", "content": "Authorization: Bearer synthetic-response-token"},
        }]}
        sanitized = sanitize_for_prompt(trace)
        self.assertNotIn("synthetic-trace-secret", json.dumps(sanitized))
        self.assertEqual(sanitized["model_calls"][0]["request_messages"][0]["content"], "Inspect inventory")
        projected = render_trace(trace, response_ids=["response-1"])
        self.assertNotIn("synthetic-response-token", projected)
        self.assertIn("response-1", projected)


class PromptEnvironmentTests(unittest.TestCase):
    def test_model_receives_readable_artifact_path_and_shell_directory_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "trace.json"
            artifact.write_text(json.dumps({"evidence": "artifact-read-ok"}))
            directory_vars = {"PWD": tmp, "OLDPWD": str(Path(tmp).parent)}
            requests = []

            def query(messages):
                requests.append(json.loads(json.dumps(messages)))
                if len(requests) == 1:
                    received_path = json.loads(messages[1]["content"])["artifact_path"]
                    program = 'import json, sys; print(json.load(open(sys.argv[1]))["evidence"])'
                    command = f"python3 -c {shlex.quote(program)} {shlex.quote(received_path)}"
                    return {"role": "assistant", "content": "Read the supplied artifact.",
                            "extra": {"actions": [{"command": command}]}}
                raise Submitted({"role": "exit", "content": "Submitted", "extra": {
                    "exit_status": "Submitted", "submission": "offline check complete",
                }})

            model = SimpleNamespace(
                query=query, get_template_vars=lambda: {}, format_message=lambda **kwargs: kwargs,
                format_observation_messages=lambda message, outputs, variables: [
                    {"role": "user", "content": json.dumps(outputs)},
                ],
                serialize=lambda: {},
            )
            with patch.dict(os.environ, directory_vars):
                env = PromptSafeLocalEnvironment(cwd=tmp, env=directory_vars)
                agent = PromptSafeAgent(
                    model, env, system_template="Read the supplied evidence.",
                    instance_template='{"artifact_path": {{artifact_path | tojson}}}', output_path=None,
                )
                result = agent.run(artifact_path=str(artifact))
                serialized = agent.serialize()
                template_vars = env.get_template_vars()
            self.assertEqual(result["exit_status"], "Submitted")
            self.assertEqual(json.loads(requests[0][1]["content"])["artifact_path"], str(artifact))
            observation = json.loads(requests[1][-1]["content"])[0]
            self.assertEqual(observation["returncode"], 0)
            self.assertEqual(observation["output"].strip(), "artifact-read-ok")
            for name, value in directory_vars.items():
                self.assertEqual(env.config.env[name], value)
                self.assertEqual(template_vars[name], value)
            self.assertIn(str(artifact), json.dumps(serialized))

    def test_shell_cannot_inherit_model_key_and_output_is_redacted(self):
        secret = "synthetic-process-model-key"
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
            env = PromptSafeLocalEnvironment()
            result = env.execute({"command": "python3 - <<'PY'\nimport os\nprint('child-key=' + os.environ.get('OPENAI_API_KEY', ''))\nprint('api_key=\"synthetic-log-key\"')\nPY"})
            self.assertEqual(os.environ["OPENAI_API_KEY"], secret)
        self.assertNotIn(secret, result["output"])
        self.assertNotIn("synthetic-log-key", result["output"])
        self.assertIn("child-key=", result["output"])
        self.assertIn(REDACTED, result["output"])

    def test_submission_is_redacted_before_exit_and_is_not_truncated(self):
        env = PromptSafeLocalEnvironment(observation_max_chars=64)
        payload = {"api_key": "synthetic-submission-key", "diagnosis": "x" * 1000}
        command = "python3 - <<'PY'\nprint('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')\nprint(" + repr(json.dumps(payload)) + ")\nPY"
        with self.assertRaises(Submitted) as raised:
            env.execute({"command": command})
        record = json.loads(raised.exception.messages[0]["extra"]["submission"])
        self.assertEqual(record["api_key"], REDACTED)
        self.assertEqual(record["diagnosis"], payload["diagnosis"])

    def test_agent_trajectory_masks_model_credentials_and_messages(self):
        model = SimpleNamespace(serialize=lambda: {"info": {"config": {"model": {
            "model_kwargs": {"api_key": "synthetic-serialized-model-key", "temperature": 0.2},
        }}}})
        agent = PromptSafeAgent(model, PromptSafeLocalEnvironment(),
                                system_template="Analyze evidence", instance_template="Inspect trace")
        agent.add_messages({"role": "user", "content": 'api_key="synthetic-message-key"'})
        serialized = agent.serialize()
        text = json.dumps(serialized)
        self.assertNotIn("synthetic-serialized-model-key", text)
        self.assertNotIn("synthetic-message-key", text)
        self.assertEqual(serialized["info"]["config"]["model"]["model_kwargs"]["temperature"], 0.2)


class PromptContractTests(unittest.TestCase):
    def test_aggregate_templates_and_context_preserve_boundaries_and_mask_secrets(self):
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from failure_analysis import aggregate_results as aggregate
        variables = {
            "step_limit": 40, "mode": "openhands", "source_roots": "/tmp/h0", "key_files": "agent.py",
            "mode_system_prompt": aggregate.OPENHANDS_SYSTEM_PROMPT,
            **{name: "/tmp/context/" + name for name in (
                "distribution_path", "cluster_path", "analyses_path", "operator_path", "memory_path",
                "val_regression_path", "previous_context_path",
            )},
        }
        engine = Environment(undefined=StrictUndefined)
        system = engine.from_string(aggregate.AGGREGATE_AGENT_SYSTEM_TEMPLATE).render(**variables)
        instance = engine.from_string(aggregate.AGGREGATE_AGENT_INSTANCE_TEMPLATE).render(**variables)
        self.assertIn("held-out test", system)
        self.assertIn("Reserve the last 3", system)
        self.assertIn("ground truth", instance)
        with tempfile.TemporaryDirectory() as tmp:
            paths = aggregate._write_aggregate_context_files(
                context_dir=Path(tmp), operator_str="operators", cluster_str="layers", distribution_str="counts",
                memory_str='api_key="synthetic-memory-key"', analyses_str="Authorization: Bearer synthetic-analysis-key",
                val_regression_section="validation", prev_plan_section="previous plan",
            )
            text = "\n".join(path.read_text() for path in paths.values())
            self.assertNotIn("synthetic-memory-key", text)
            self.assertNotIn("synthetic-analysis-key", text)
            self.assertIn("validation", text)

    def test_analysis_and_modify_templates_render_with_strict_variables(self):
        engine = Environment(undefined=StrictUndefined)
        context = {
            "agent_source_dir": "/tmp/h0", "agent_source_root": "/tmp/run",
            "instance_id": "stock__example0__rollout0", "failure_category": "unresolved",
            "task_description": "Inspect stock", "traj_path": "/tmp/sanitized.json",
            "raw_traj_path": "/tmp/raw.json", "manifest_path": "/tmp/manifest.json",
            "htir_path": "/tmp/trace.htir.json", "n_model_calls": 28,
            "output": {"exception_info": "", "returncode": 0, "output": "Recorded evidence"},
        }
        for relative in ("failure_analysis/analysis_config_openhands.yaml", "enhancement_implementation/config_openhands.yaml"):
            with self.subTest(config=relative):
                config = yaml.safe_load((REPO_ROOT / relative).read_text())
                variables = context | {"step_limit": config["step_limit"]}
                system = engine.from_string(config["agent"]["system_template"]).render(**variables)
                instance = engine.from_string(config["agent"]["instance_template"]).render(**variables)
                observation = engine.from_string(config["model"]["observation_template"]).render(**variables)
                self.assertIn("held-out test", system)
                self.assertIn(".env", system)
                self.assertIn("Step 6", instance)
                self.assertIn("Recorded evidence", observation)
                self.assertEqual(config["model"]["model_kwargs"]["max_tokens"], 8192)

    def test_compact_htir_cli_masks_secrets_in_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.htir.json"
            path.write_text(json.dumps({"instance_id": "sample", "views": {"failure": [{
                "node_id": "evt_1", "summary": 'api_key="synthetic-htir-key"',
            }]}}))
            result = subprocess.run([sys.executable, str(REPO_ROOT / "failure_analysis/print_htir_compact.py"), str(path)],
                                    capture_output=True, text=True, check=True)
        self.assertNotIn("synthetic-htir-key", result.stdout)
        self.assertIn("evt_1", result.stdout)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def valid_spec() -> dict:
    return {
        "plan_metadata": {"mode": "openhands", "summary": "Repair tool argument guidance"},
        "edit_budget": {
            "recommended_budget": "low",
            "max_files_to_modify": 1,
            "allowed_paths": ["prompts/system.md"],
            "forbidden_paths": [],
            "rationale": "One scoped instruction change",
        },
        "fixes": [{
            "id": "fix_tool_schema",
            "title": "Use the declared tool schema",
            "priority": "high",
            "target_files": ["prompts/system.md"],
            "target_symbols": [],
            "problem_statement": "The agent omits required tool arguments.",
            "required_behavior_delta": "Check the declared tool argument names.",
            "implementation_steps": ["Clarify argument validation guidance."],
            "tests": ["Verify required arguments are included."],
            "risk_level": "low",
            "dependencies": [],
            "regression_risks": ["Preserve existing task instructions."],
            "must_not_change": ["Model and runtime settings"],
        }],
    }


def response(spec: dict, title: str = "Scoped repair plan") -> str:
    return (
        f"# {title}\n\n"
        "Inspect the failure evidence and tool schema before constructing arguments. "
        "Preserve passing behaviors while updating the tool argument guidance.\n\n"
        f"```json\n{json.dumps(spec)}\n```\n"
    )


class AggregateRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from failure_analysis import aggregate_results
        cls.aggregate = aggregate_results

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "plan.md"
        self.spec_output = self.root / "plan.json"
        self.results = self.root / "analysis.jsonl"
        self.results.write_text(json.dumps({
            "instance_id": "failed-task",
            "failure_category": "unresolved",
            "affected_component": "prompts",
            "agent_design_issue": "Required tool arguments were omitted.",
        }) + "\n")
        self.models = []
        self.environments = []

    @contextlib.contextmanager
    def run_context(self, outcomes: list, mode: str = "openhands", *, direct: bool = False):
        from minisweagent.exceptions import Submitted

        aggregate = self.aggregate
        environments = self.environments
        models = self.models
        original_environment = aggregate.PromptSafeLocalEnvironment
        remaining = iter(outcomes)

        class SubmittingModel:
            def __init__(self, outcome):
                self.outcome = outcome
                self.requests = []

            def get_template_vars(self):
                return {}

            def format_message(self, **kwargs):
                return kwargs

            def serialize(self):
                return {}

            def query(self, messages):
                self.requests.append(copy.deepcopy(messages))
                plan_path = Path(environments[-1].config.env["AGGREGATE_PLAN_PATH"])
                if plan_path.exists():
                    raise AssertionError("The new session inherited an old submission file")
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                plan_path.write_text(self.outcome)
                raise Submitted({"role": "exit", "content": "Submitted", "extra": {
                    "exit_status": "Submitted", "submission": self.outcome,
                }})

        def make_model(**kwargs):
            model = SubmittingModel(next(remaining))
            models.append(model)
            return model

        def make_environment(**kwargs):
            environment = original_environment(**kwargs)
            environments.append(environment)
            return environment

        empty_spec = valid_spec()
        empty_spec["fixes"] = []
        argv = [
            "aggregate_results.py", "--mode", mode, "--model", "offline",
            "--results-file", str(self.results), "--output", str(self.output),
            "--spec-output", str(self.spec_output), "--memory-root", str(self.root / "memory"),
        ]
        if direct:
            argv.append("--direct-llm")
        with (
            patch.object(sys, "argv", argv),
            patch.object(aggregate, "_aggregate_context_dir", return_value=self.root / "context"),
            patch.object(aggregate, "technical_reference_prompt", return_value=""),
            patch.object(aggregate, "LitellmTextbasedModel", side_effect=make_model),
            patch.object(aggregate, "PromptSafeLocalEnvironment", side_effect=make_environment),
            patch.object(aggregate, "call_llm", return_value=response(empty_spec)) as converter,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            yield converter

    def assert_fresh_retry(self, expected_attempts: int = 2):
        self.assertEqual(len(self.models), expected_attempts)
        for model in self.models[1:]:
            self.assertEqual(self.models[0].requests[0], model.requests[0])
        paths = [Path(env.config.env["AGGREGATE_PLAN_PATH"]) for env in self.environments]
        self.assertEqual(len(set(paths)), expected_attempts)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_invalid_plan_and_failed_conversion_regenerate_for_every_benchmark(self):
        broken = response({
            "plan_metadata": {}, "edit_budget": {"max_files_changed": 1}, "fixes": [],
        }, title="INVALID_PLAN_FROM_FIRST_ATTEMPT")
        repaired = response(valid_spec())
        for mode in ("swe", "gaia", "appworld", "terminal_bench", "openhands"):
            with self.subTest(mode=mode):
                self.models.clear()
                self.environments.clear()
                with self.run_context([broken] * 5 + [repaired], mode) as converter:
                    self.aggregate.main()
                self.assertEqual(converter.call_count, 5)
                self.assert_fresh_retry(expected_attempts=6)
                self.assertEqual(json.loads(self.spec_output.read_text())["fixes"][0]["id"], "fix_tool_schema")
                for attempt in range(5):
                    self.assertEqual((self.root / f"plan.md.attempt{attempt}.raw.txt").read_text(), broken)
                    old_trajectory = json.loads((self.root / f"plan.md.attempt{attempt}.aggregate_agent.traj.json").read_text())
                    self.assertEqual(old_trajectory["info"]["submission"], broken)
                self.assertEqual((self.root / "plan.md.raw.txt").read_text(), repaired)
                new_trajectory = json.loads((self.root / "plan.md.aggregate_agent.traj.json").read_text())
                self.assertEqual(new_trajectory["info"]["submission"], repaired)
                self.output.unlink()
                self.spec_output.unlink()

    def test_five_failed_retries_stop_without_writing_final_outputs(self):
        invalid = valid_spec()
        invalid["fixes"] = []
        broken = response(invalid)
        with self.run_context([broken] * 6) as converter:
            with self.assertRaisesRegex(ValueError, "non-empty fixes list"):
                self.aggregate.main()
        self.assertEqual(converter.call_count, 6)
        self.assert_fresh_retry(expected_attempts=6)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.spec_output.exists())
        self.assertFalse((self.root / "plan.clusters.json").exists())
        for attempt in range(5):
            self.assertTrue((self.root / f"plan.md.attempt{attempt}.aggregate_agent.traj.json").exists())
        self.assertTrue((self.root / "plan.md.aggregate_agent.traj.json").exists())

    def test_generation_exception_and_empty_submission_retry_from_scratch(self):
        for failed_outcome in (TimeoutError("synthetic provider failure"), ""):
            with self.subTest(outcome=failed_outcome):
                self.models.clear()
                self.environments.clear()
                with self.run_context([failed_outcome, response(valid_spec())]) as converter:
                    self.aggregate.main()
                converter.assert_not_called()
                self.assert_fresh_retry()
                self.assertTrue(self.spec_output.exists())
                self.output.unlink()
                self.spec_output.unlink()

    def test_valid_initial_plan_does_not_retry(self):
        with self.run_context([response(valid_spec())]) as converter:
            self.aggregate.main()
        self.assertEqual(len(self.models), 1)
        converter.assert_not_called()
        self.assertFalse((self.root / "plan.md.attempt0.raw.txt").exists())

    def test_existing_conversion_fallback_can_succeed_without_regeneration(self):
        markdown = "# Repair plan\n" + "Inspect and correct the tool argument guidance.\n" * 8
        with self.run_context([markdown]) as converter:
            converter.return_value = response(valid_spec())
            self.aggregate.main()
        self.assertEqual(len(self.models), 1)
        converter.assert_called_once()
        self.assertTrue(self.spec_output.exists())

    def test_legacy_direct_planner_retries_with_identical_original_messages(self):
        invalid = valid_spec()
        invalid["fixes"] = []
        with self.run_context([], direct=True) as llm:
            llm.side_effect = [response(invalid)] * 10 + [response(valid_spec())]
            self.aggregate.main()
        self.assertEqual(llm.call_count, 11)
        for attempt in range(1, 6):
            self.assertEqual(llm.call_args_list[0], llm.call_args_list[attempt * 2])
        self.assertEqual(self.models, [])
        self.assertTrue(self.spec_output.exists())


if __name__ == "__main__":
    unittest.main()

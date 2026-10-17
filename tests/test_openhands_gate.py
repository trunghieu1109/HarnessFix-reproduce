from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_pipeline_openhands as driver


class OpenHandsStandaloneGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ids = [f"task{i}" for i in range(10)]
        (self.root / "ids.txt").write_text("\n".join(self.ids))
        (self.root / "spec.json").write_text(json.dumps({
            "fixes": [{"target_metrics": ["resolved_rate"]}],
        }))
        for label in ("baseline", "candidate"):
            traces = self.root / label / "traces"
            traces.mkdir(parents=True)
            for instance in self.ids:
                folder = traces / instance
                folder.mkdir()
                trace = folder / "trace.json"
                trace.write_text(json.dumps({"events": [], "metrics": {"accumulated_cost": 0.1}}))
                (folder / "manifest.json").write_text(json.dumps({
                    "trace_path": str(trace), "llm_completion_paths": [],
                }))
        self._write_eval("baseline", self.ids[:5], [])

    def _write_eval(self, label, solved, errors):
        (self.root / label / "eval.json").write_text(json.dumps({
            "all_ids": self.ids, "resolved_ids": solved, "error_ids": errors,
            "score_mean": len(solved) / len(self.ids),
        }))

    def _run_gate(self, *, solved=None, errors=(), audit=None, extra=(), include_audit=True):
        self._write_eval("candidate", self.ids[:6] if solved is None else solved, list(errors))
        (self.root / "audit.json").write_text(json.dumps({"passed": True} if audit is None else audit))
        output = self.root / "gate.json"
        argv = ["run_pipeline_openhands.py", "gate",
                "--baseline-traces", str(self.root / "baseline/traces"),
                "--baseline-eval", str(self.root / "baseline/eval.json"),
                "--current-traces", str(self.root / "candidate/traces"),
                "--current-eval", str(self.root / "candidate/eval.json"),
                "--ids-file", str(self.root / "ids.txt"),
                "--plan-spec", str(self.root / "spec.json"), "--output", str(output)]
        if include_audit:
            argv += ["--audit", str(self.root / "audit.json")]
        with patch.object(sys, "argv", argv + list(extra)), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                driver.main()
        return raised.exception.code, json.loads(output.read_text()) if output.exists() else None

    def test_swe_policy_allows_net_gain_with_more_than_two_regressions(self):
        code, result = self._run_gate(solved=self.ids[3:9])
        self.assertEqual(code, 0)
        self.assertTrue(result["promoted"])
        self.assertEqual(result["regression_count"], 3)
        self.assertEqual(result["net_change"], 1)
        self.assertEqual(result["promotion_policy"], "swe_shared")
        self.assertIn("invalid_delta_within_limit", result["skipped_checks"])

    def test_error_rate_regression_rejects_otherwise_improving_candidate(self):
        code, result = self._run_gate(errors=self.ids[6:])
        self.assertEqual(code, 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reasons"], ["error_delta_within_limit"])
        self.assertAlmostEqual(result["error_rate_delta"], 0.4)

    def test_audit_rejection_prevents_promotion(self):
        code, result = self._run_gate(audit={"passed": False})
        self.assertEqual(code, 1)
        self.assertEqual(result["failure_reasons"], ["audit_passed"])

    def test_custom_threshold_matches_configured_promotion_policy(self):
        code, result = self._run_gate(errors=self.ids[6:], extra=["--max-error-rate-delta", "0.5"])
        self.assertEqual(code, 0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["promotion_config"]["max_error_rate_delta"], 0.5)

    def test_audit_is_required_and_must_be_an_actual_boolean(self):
        code, result = self._run_gate(include_audit=False)
        self.assertEqual(code, 2)
        self.assertIsNone(result)
        code, result = self._run_gate(audit={"passed": "false"})
        self.assertEqual(code, 2)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

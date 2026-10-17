from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from failure_analysis.consolidation import consolidate_diagnoses, enrich_spec_with_clusters
from failure_analysis.operator_registry import OPENHANDS_TARGET_METRICS, format_operator_registry_for_prompt
from failure_analysis.validation_metrics import compute_openhands_metrics, evaluate_target_metrics
from tests.diagnosis_fixtures import valid_diagnosis
from tests.test_aggregate_retry import valid_spec


class OpenHandsAggregateMetricsTests(unittest.TestCase):
    def _cost_metrics(self, root, costs):
        traces = root / "traces"
        for index, cost in enumerate(costs):
            folder = traces / str(index)
            folder.mkdir(parents=True)
            trace = folder / "trace.json"
            trace.write_text(json.dumps({"events": [], "metrics": cost}))
            (folder / "manifest.json").write_text(json.dumps({
                "trace_path": str(trace), "llm_completion_paths": [],
            }))
        report = root / "eval.json"
        report.write_text(json.dumps({"all_ids": list(map(str, range(len(costs)))), "resolved_ids": []}))
        return compute_openhands_metrics(traces, report)

    def test_missing_or_partial_cost_is_not_an_improvement(self):
        for costs in ([{}], [{"accumulated_cost": None}], [{"accumulated_cost": 0.2}, {}]):
            with self.subTest(costs=costs), tempfile.TemporaryDirectory() as temporary:
                current = self._cost_metrics(Path(temporary), costs)
                self.assertIsNone(current["metrics"]["avg_instance_cost"])
                result = evaluate_target_metrics(
                    {"fixes": [{"target_metrics": ["avg_instance_cost"]}]},
                    {"mode": "openhands", "metrics": {"avg_instance_cost": 0.2}}, current,
                )
                self.assertEqual(result["metrics"], [])
                self.assertEqual(result["improved_metric_count"], 0)

    def test_recorded_cost_including_zero_has_the_correct_average(self):
        for costs, expected in (([0.0], 0.0), ([0.0, 0.2], 0.1)):
            with self.subTest(costs=costs), tempfile.TemporaryDirectory() as temporary:
                result = self._cost_metrics(Path(temporary), [{"accumulated_cost": cost} for cost in costs])
                self.assertAlmostEqual(result["metrics"]["avg_instance_cost"], expected)

    def test_available_targets_match_metrics_actually_computed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "eval.json"
            report.write_text(json.dumps({"all_ids": [], "resolved_ids": []}))
            measured = compute_openhands_metrics(root / "traces", report)["metrics"]
        self.assertEqual(set(measured), OPENHANDS_TARGET_METRICS)
        self.assertNotIn("empty_patch_rate", measured)

    def test_registry_and_layer_buckets_do_not_advertise_unavailable_metrics(self):
        prompt = format_operator_registry_for_prompt("openhands")
        for metric in ("loop_rate", "empty_patch_rate", "invalid_submission_rate", "completion_without_effect_rate"):
            self.assertNotIn(f"metrics={metric}", prompt)
            self.assertNotIn(f", {metric}", prompt)
        record = valid_diagnosis(defect_class="termination", recommended_operator_family="guardrail")
        clusters = consolidate_diagnoses([record], "openhands")
        self.assertTrue(clusters)
        self.assertTrue(all(cluster["target_metrics"] == [] for cluster in clusters))

    def test_spec_keeps_supported_targets_without_substituting_other_metrics(self):
        spec = valid_spec()
        spec["fixes"][0].update(
            operator_family="guardrail", target_defect_class="termination",
            target_metrics=["resolved_rate", "loop_rate", "avg_api_calls", "empty_patch_rate"],
        )
        enriched = enrich_spec_with_clusters(spec, [], "openhands")
        self.assertEqual(enriched["fixes"][0]["target_metrics"], ["resolved_rate", "avg_api_calls"])
        self.assertEqual(spec["fixes"][0]["target_metrics"],
                         ["resolved_rate", "loop_rate", "avg_api_calls", "empty_patch_rate"])

    def test_unavailable_registry_defaults_leave_no_spec_targets(self):
        spec = valid_spec()
        spec["fixes"][0].update(operator_family="guardrail", target_defect_class="termination")
        enriched = enrich_spec_with_clusters(spec, [], "openhands")
        self.assertEqual(enriched["fixes"][0]["target_metrics"], [])

    def test_other_benchmark_specs_keep_their_existing_targets(self):
        for mode in ("swe", "gaia", "appworld", "terminal_bench"):
            with self.subTest(mode=mode):
                spec = valid_spec()
                targets = ["loop_rate", "empty_patch_rate", "invalid_submission_rate"]
                spec["fixes"][0]["target_metrics"] = targets
                enriched = enrich_spec_with_clusters(spec, [], mode)
                self.assertEqual(enriched["fixes"][0]["target_metrics"], targets)


class SpecInspectionPathTests(unittest.TestCase):
    def test_prompt_targets_under_wildcards_stay_concrete(self):
        for target in ("prompts/system.md", "skills/nested/check.md"):
            with self.subTest(target=target):
                spec = valid_spec()
                spec["fixes"][0].update(operator_family="prompt", target_files=[target],
                                         active_files=[], must_inspect_files=[])
                enriched = enrich_spec_with_clusters(spec, [], "openhands")
                fix = enriched["fixes"][0]
                self.assertEqual(fix["active_files"], [target])
                self.assertEqual(fix["must_inspect_files"], [target])
                self.assertEqual(enriched["edit_budget"]["active_files"], [target])
                self.assertEqual(enriched["edit_budget"]["must_inspect_files"], [target])
                self.assertIn("prompts/**", enriched["edit_budget"]["allowed_paths"])
                self.assertIn("skills/**", enriched["edit_budget"]["allowed_paths"])

    def test_scope_patterns_are_not_fabricated_inspection_files(self):
        spec = valid_spec()
        spec["fixes"][0].update(operator_family="prompt", target_files=["prompts_other/system.md"],
                                 active_files=[], must_inspect_files=[])
        enriched = enrich_spec_with_clusters(spec, [], "openhands")
        self.assertEqual(enriched["fixes"][0]["active_files"], [])
        self.assertEqual(enriched["fixes"][0]["must_inspect_files"], [])


class TargetMetricComparisonTests(unittest.TestCase):
    def test_missing_or_unrecorded_values_do_not_dilute_real_improvement(self):
        spec = {"fixes": [{"target_metrics": [
            "resolved_rate", "loop_rate", "avg_steps", "error_rate", "avg_instance_cost",
        ]}]}
        baseline = {"mode": "openhands", "metrics": {
            "resolved_rate": 0.0, "avg_steps": 10, "error_rate": 0.2, "avg_instance_cost": None,
        }}
        current = {"mode": "openhands", "metrics": {
            "resolved_rate": 0.2, "loop_rate": 0.0, "error_rate": 0.0, "avg_instance_cost": 0.1,
        }}
        result = evaluate_target_metrics(spec, baseline, current)
        self.assertEqual([item["metric"] for item in result["metrics"]], ["resolved_rate", "error_rate"])
        self.assertAlmostEqual(result["aggregate_delta"], 0.2)
        self.assertEqual(result["improved_metric_count"], 2)

    def test_entirely_missing_targets_do_not_create_measurements(self):
        spec = {"fixes": [{"target_metrics": ["loop_rate"]}]}
        run = {"mode": "openhands", "metrics": {"resolved_rate": 0.5}}
        result = evaluate_target_metrics(spec, run, run)
        self.assertEqual(result["metrics"], [])
        self.assertEqual(result["improved_metric_count"], 0)

    def test_recorded_zero_is_a_valid_measurement_for_every_benchmark(self):
        spec = {"fixes": [{"target_metrics": ["error_rate"]}]}
        for mode in ("swe", "gaia", "appworld", "terminal_bench", "openhands"):
            with self.subTest(mode=mode):
                baseline = {"mode": mode, "metrics": {"error_rate": 0.5}}
                current = {"mode": mode, "metrics": {"error_rate": 0.0}}
                result = evaluate_target_metrics(spec, baseline, current)
                self.assertEqual(len(result["metrics"]), 1)
                self.assertEqual(result["metrics"][0]["current"], 0.0)
                self.assertEqual(result["aggregate_delta"], 0.5)
                self.assertEqual(result["improved_metric_count"], 1)

    def test_existing_default_target_still_uses_recorded_outcomes(self):
        for mode, metric in (("swe", "resolved_rate"), ("openhands", "resolved_rate"), ("gaia", "accuracy")):
            with self.subTest(mode=mode):
                result = evaluate_target_metrics(
                    {"fixes": []}, {"mode": mode, "metrics": {metric: 0.0}},
                    {"mode": mode, "metrics": {metric: 0.25}},
                )
                self.assertEqual([item["metric"] for item in result["metrics"]], [metric])
                self.assertEqual(result["improved_metric_count"], 1)


class SpecMetricContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from failure_analysis import aggregate_results
        cls.aggregate = aggregate_results

    def test_metrics_remain_optional_and_unknown_names_can_be_filtered(self):
        for targets in (None, [], ["not_computed"]):
            with self.subTest(targets=targets):
                spec = valid_spec()
                if targets is not None:
                    spec["fixes"][0]["target_metrics"] = targets
                self.aggregate.validate_spec(spec)
                enriched = enrich_spec_with_clusters(spec, [], "openhands")
                self.aggregate.validate_spec(enriched)
                self.assertTrue(set(enriched["fixes"][0]["target_metrics"]) <= OPENHANDS_TARGET_METRICS)

    def test_invalid_metric_types_fail_before_spec_enrichment(self):
        for value in (None, "resolved_rate", {}, [None], [{}], [" "]):
            with self.subTest(value=value):
                spec = valid_spec()
                spec["fixes"][0]["target_metrics"] = value
                with self.assertRaisesRegex(ValueError, "target_metrics must be a list"):
                    self.aggregate.validate_spec(spec)


if __name__ == "__main__":
    unittest.main()

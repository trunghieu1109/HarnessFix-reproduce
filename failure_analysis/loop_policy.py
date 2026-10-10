"""Shared comparison and promotion rules for SWE and OpenHands closed loops."""

from __future__ import annotations

import json
from pathlib import Path

from failure_analysis.validation_metrics import (
    compute_cost_ratio,
    compute_run_metrics,
    evaluate_target_metrics,
    load_eval_json,
)


def compare_runs(*, mode: str, split: str, ids_file: Path, baseline_eval: Path,
                 current_eval: Path, baseline_traces: Path, current_traces: Path,
                 spec_path: Path, max_cost_ratio: float = 1.25) -> dict:
    subset = set(ids_file.read_text().split())
    baseline_path, baseline_data = load_eval_json(baseline_eval)
    current_path, current_data = load_eval_json(current_eval)
    baseline_resolved = set(baseline_data.get("resolved_ids", [])) & subset
    current_resolved = set(current_data.get("resolved_ids", [])) & subset
    regressed = sorted(baseline_resolved - current_resolved)
    improved = sorted(current_resolved - baseline_resolved)
    baseline_metrics = compute_run_metrics(mode, baseline_traces, baseline_path)
    current_metrics = compute_run_metrics(mode, current_traces, current_path)
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {"fixes": []}
    return {
        "split": split,
        "comparison_type": f"{split}_current_base_vs_enhanced",
        "baseline_count": len(baseline_resolved),
        "current_count": len(current_resolved),
        "net_change": len(current_resolved) - len(baseline_resolved),
        "regression_count": len(regressed),
        "improvement_count": len(improved),
        "regressed_ids": regressed,
        "improved_ids": improved,
        "cost_ratio": compute_cost_ratio(baseline_metrics, current_metrics),
        "target_metric_results": evaluate_target_metrics(spec, baseline_metrics, current_metrics),
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "comparison_config": {
            "mode": mode, "ids_file": str(ids_file), "cost_report_threshold": max_cost_ratio,
        },
    }


def decide_promotion(audit: dict, train_compare: dict, val_compare: dict, *,
                     min_improvement: int = 1, min_target_metrics: int = 1,
                     max_error_rate_delta: float = 0.15,
                     max_invalid_rate_delta: float = 0.15) -> dict:
    """Match SWE: train and cost are reported; audit and validation decide."""
    baseline = val_compare.get("baseline_metrics", {}).get("metrics", {})
    current = val_compare.get("current_metrics", {}).get("metrics", {})
    error_delta = current.get("error_rate", 0.0) - baseline.get("error_rate", 0.0)
    invalid_delta = current.get("invalid_submission_rate", 0.0) - baseline.get("invalid_submission_rate", 0.0)
    target_improved = val_compare.get("target_metric_results", {}).get("improved_metric_count", 0)
    checks = {
        "audit_passed": bool(audit.get("passed", True)),
        "net_improvement": val_compare.get("net_change", 0) >= min_improvement,
        "target_metric_improved": target_improved >= min_target_metrics,
        "error_delta_within_limit": error_delta <= max_error_rate_delta,
        "invalid_delta_within_limit": invalid_delta <= max_invalid_rate_delta,
    }
    promoted = all(checks.values())
    return {
        "passed": promoted, "promoted": promoted,
        "decision": "promote" if promoted else "do_not_promote",
        "failure_reasons": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "val_net_change": val_compare.get("net_change"),
        "val_regression_count": val_compare.get("regression_count"),
        "val_improvement_count": val_compare.get("improvement_count"),
        "train_net_change": train_compare.get("net_change"),
        "train_regression_count": train_compare.get("regression_count"),
        "train_improvement_count": train_compare.get("improvement_count"),
        "target_improved_metric_count": target_improved,
        "error_rate_delta": error_delta, "invalid_submission_rate_delta": invalid_delta,
        "cost_ratio": val_compare.get("cost_ratio"), "cost_gate_enabled": False,
        "regressed_ids": val_compare.get("regressed_ids", []),
        "improved_ids": val_compare.get("improved_ids", []),
        "promotion_config": {
            "min_improvement": min_improvement, "min_target_metrics": min_target_metrics,
            "max_error_rate_delta": max_error_rate_delta,
            "max_invalid_rate_delta": max_invalid_rate_delta,
        },
    }

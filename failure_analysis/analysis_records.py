"""Validate diagnoses consistently for analysis, resume, and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TEXT_FIELDS = (
    "instance_id", "failure_category", "exit_status", "failure_manifestation",
    "failure_reason", "agent_design_issue", "affected_component",
)
STEP_FIELDS = ("candidate_responsible_steps", "responsible_steps", "responsible_step_ids")
NODE_FIELDS = ("responsible_nodes", "responsible_events", "downstream_nodes", "downstream_events")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not value.strip().startswith("FILL_")


def analysis_record_errors(
    record: Any, *, expected_instance_id: str | None = None,
    expected_failure_category: str | None = None, htir_bundle: dict | None = None,
) -> list[str]:
    """Check format and recorded references, without judging the proposed cause.

    Attribution remains optional. If supplied, its IDs must refer to the bundle
    that belongs to this analysis rather than another rollout or a raw index.
    """
    if not isinstance(record, dict):
        return ["Diagnosis must be a JSON object"]
    errors = []
    if (
        record.get("_analysis_fallback") or record.get("_parse_error")
        or record.get("_analysis_validation_errors")
        or record.get("exit_status") in (
            "analysis_agent_exception", "analysis_parse_error", "analysis_failed", "analysis_invalid_diagnosis",
        )
    ):
        errors.append("Record represents an analysis failure")
    for field in REQUIRED_TEXT_FIELDS:
        if not _nonempty_text(record.get(field)):
            errors.append(f"{field} must be nonempty text without a FILL_ placeholder")
    for field, expected in (("instance_id", expected_instance_id), ("failure_category", expected_failure_category)):
        if expected is not None and record.get(field) != expected:
            errors.append(f"{field} does not match the analyzed case")
    bundle_instance = (htir_bundle or {}).get("instance_id")
    if bundle_instance is not None and record.get("instance_id") != bundle_instance:
        errors.append("HTIR bundle belongs to a different analyzed case")
    api_calls = record.get("api_calls")
    if type(api_calls) is not int or api_calls < 0:
        errors.append("api_calls must be a nonnegative integer")
    for field in ("severity", "confidence"):
        if record.get(field) not in ("low", "medium", "high"):
            errors.append(f"{field} must be low, medium, or high")
    for field in ("defect_class", "recommended_operator_family", "fix_scope", "htir_path"):
        if field in record and not _nonempty_text(record[field]):
            errors.append(f"{field} must be nonempty text when supplied")
    for field in STEP_FIELDS + NODE_FIELDS + ("implicated_harness_layers", "harness_layers", "defect_labels"):
        if field in record and (
            not isinstance(record[field], list) or not all(_nonempty_text(value) for value in record[field])
        ):
            errors.append(f"{field} must be a list of nonempty strings")

    steps = (htir_bundle or {}).get("agent_trace_steps") or []
    step_ids = {step["step_id"] for step in steps if step.get("step_id")}
    graph = (htir_bundle or {}).get("graph") or {}
    nodes = graph.get("nodes") or graph.get("events") or []
    node_ids = {node.get("node_id") or node.get("event_id") for node in nodes}
    node_ids.update(node_id for step in steps for node_id in step.get("legacy_event_ids", []))
    for view in ((htir_bundle or {}).get("views") or {}).values():
        if isinstance(view, list):
            node_ids.update(node.get("node_id") or node.get("event_id") for node in view if isinstance(node, dict))

    def check_reference(field: str, value: Any, valid_ids: set) -> None:
        if htir_bundle is not None and _nonempty_text(value) and value not in valid_ids:
            errors.append(f"{field} references an ID absent from this HTIR bundle")

    for fields, valid_ids in ((STEP_FIELDS, step_ids), (NODE_FIELDS, node_ids)):
        for field in fields:
            if isinstance(record.get(field), list):
                for value in record[field]:
                    check_reference(field, value, valid_ids)
    if "evidence_spans" in record:
        spans = record["evidence_spans"]
        if not isinstance(spans, list):
            errors.append("evidence_spans must be a list")
        else:
            for span in spans:
                if not isinstance(span, dict) or not span:
                    errors.append("Each evidence span must be a nonempty object")
                    continue
                for field in ("step_id", "node_id", "event_id", "source_ref"):
                    if field in span and span[field] is not None:
                        if not _nonempty_text(span[field]):
                            errors.append(f"evidence_spans.{field} must be nonempty text")
                        elif field != "source_ref":
                            check_reference(f"evidence_spans.{field}", span[field],
                                            step_ids if field == "step_id" else node_ids)
    return errors


def analysis_record_succeeded(record: Any) -> bool:
    if analysis_record_errors(record):
        return False
    # Validate saved references too when the original bundle is still available.
    # Historical records can remain usable after artifacts have been archived.
    htir_path = record.get("htir_path")
    try:
        if isinstance(htir_path, str) and Path(htir_path).is_file():
            bundle = json.loads(Path(htir_path).read_text())
            if not isinstance(bundle, dict):
                return False
            return not analysis_record_errors(record, htir_bundle=bundle)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    return True

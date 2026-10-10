"""Distinguish model diagnoses from saved analysis-error evidence."""

from __future__ import annotations

from typing import Any


def analysis_record_succeeded(record: Any) -> bool:
    if not isinstance(record, dict) or not record:
        return False
    return not (
        record.get("_analysis_fallback")
        or record.get("_parse_error")
        or record.get("exit_status") in {
            "analysis_agent_exception", "analysis_parse_error", "analysis_failed"
        }
    )

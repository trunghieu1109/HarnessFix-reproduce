"""Read OpenHands event traces and optional SDK completion logs."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _response_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if choices:
        return choices[0].get("message") or {}
    return {"role": "assistant", "content": response.get("output", [])}


def load_openhands_trace(manifest: dict[str, Any], fallback_path: str | None = None) -> dict[str, Any]:
    path = next((Path(value) for value in (
        manifest.get("trace_path"), fallback_path, manifest.get("raw_trace_path")
    ) if value and Path(value).is_file()), None)
    if path is None:
        raise FileNotFoundError("Neither filtered nor raw OpenHands trace is available")
    loaded = json.loads(path.read_text())
    trace = dict(loaded) if isinstance(loaded, dict) else {"events": loaded}
    if not trace.get("metrics"):
        for event in reversed(trace.get("events", [])):
            if event.get("kind") == "ConversationStateUpdateEvent" and event.get("key") == "full_state":
                usages = (event.get("value", {}).get("stats", {}).get("usage_to_metrics") or {})
                if "default" in usages:
                    trace["metrics"] = usages["default"]
                    break
    paths = manifest.get("llm_completion_paths")
    if paths is None:
        paths = sorted((Path(manifest["log_dir"]) / "llm_completions").glob("*.json")) if manifest.get("log_dir") else []
    calls = list(trace.get("model_calls") or [])
    known_ids = {str(call.get("response_id")) for call in calls if call.get("response_id")}
    payloads = []
    for value in paths:
        log_path = Path(value)
        if not log_path.is_file():
            raise FileNotFoundError(f"Referenced LLM completion log is missing: {log_path}")
        payloads.append((str(log_path), log_path.name, json.loads(log_path.read_text())))
    for index, event in enumerate(trace.get("events", [])):
        if event.get("kind") == "LLMCompletionLogEvent" and event.get("log_data"):
            payloads.append((f"trace.events[{index}].log_data", event.get("filename"), json.loads(event["log_data"])))
    for source_ref, filename, payload in payloads:
        response = payload.get("response") or {}
        response_id = response.get("id") or filename or source_ref
        if response_id in known_ids:
            continue
        request = {key: payload[key] for key in ("messages", "instructions", "input", "tools") if key in payload}
        calls.append({
            "response_id": response_id,
            "model": response.get("model"),
            "request_messages": payload.get("messages") or [],
            "request_payload": request,
            "response_message": _response_message(response),
            "original_response_message": _response_message(payload["raw_response"]) if payload.get("raw_response") else None,
            "generation_settings": {key: value for key, value in (payload.get("kwargs") or {}).items()
                                    if key in {"temperature", "top_p", "max_tokens", "max_completion_tokens", "reasoning_effort", "stop"}},
            "usage": response.get("usage") or payload.get("usage_summary") or {},
            "timestamp": payload.get("timestamp", 0),
            "error": payload.get("error"),
            "source_ref": source_ref,
            "request_evidence": "sdk_completion_log",
            "response_evidence": "sdk_completion_log" if response else "not_recorded",
        })
        known_ids.add(response_id)
    if calls:
        trace["model_calls"] = sorted(calls, key=lambda call: float(call.get("timestamp") or 0))
    return trace


def event_response_key(event: dict[str, Any], index: int) -> str:
    return str(event.get("llm_response_id") or f"event:{index}")


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp()
    return 0.0


def openhands_model_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Use recorded requests when present; retain explicitly partial event evidence."""
    calls = deepcopy(trace.get("model_calls") or [])
    by_id = {str(call["response_id"]): call for call in calls}
    usage = {str(row.get("response_id")): row for row in (trace.get("metrics") or {}).get("token_usages", []) if row.get("response_id")}
    for index, event in enumerate(trace.get("events") or []):
        kind = event.get("kind")
        is_response = kind == "ActionEvent" or (kind == "MessageEvent" and event.get("source") == "agent")
        if not is_response:
            continue
        key = event_response_key(event, index)
        if key not in by_id:
            call = {
                "response_id": key,
                "request_messages": [],
                "response_message": {"role": "assistant", "content": [], "tool_calls": []},
                "usage": usage.get(key, {}),
                "request_evidence": "not_recorded",
                "response_evidence": "trace_events",
                "source_ref": f"trace.events[{index}]",
                "timestamp": _timestamp(event.get("timestamp")),
            }
            calls.append(call)
            by_id[key] = call
        call = by_id[key]
        if call.get("response_evidence") == "sdk_completion_log":
            continue
        message = call["response_message"]
        if kind == "MessageEvent":
            incoming = deepcopy(event.get("llm_message") or event.get("message") or {})
            prior_tools = message.get("tool_calls") or []
            message.update(incoming)
            if prior_tools and not message.get("tool_calls"):
                message["tool_calls"] = prior_tools
        else:
            thought = event.get("thought")
            if thought and not message.get("content"):
                message["content"] = thought
            tool_call = event.get("tool_call")
            if tool_call:
                if not message.get("tool_calls"):
                    message["tool_calls"] = []
                if tool_call not in message["tool_calls"]:
                    message["tool_calls"].append(deepcopy(tool_call))
            for field in ("reasoning_content", "thinking_blocks", "responses_reasoning_item"):
                if event.get(field):
                    message[field] = event[field]
            if event.get("action") not in call.setdefault("actions", []):
                call["actions"].append(event.get("action"))
    # SDK metric records include auxiliary requests that have no ActionEvent.
    response_order = {str(row["response_id"]): index for index, row in enumerate(
        (trace.get("metrics") or {}).get("response_latencies", [])
    ) if row.get("response_id")}
    if response_order:
        return sorted(calls, key=lambda call: response_order.get(str(call["response_id"]), len(response_order)))
    return sorted(calls, key=lambda call: _timestamp(call.get("timestamp")))


def openhands_api_call_count(trace: dict[str, Any]) -> int:
    metrics = trace.get("metrics") or {}
    ids = {str(row["response_id"]) for key in ("token_usages", "response_latencies")
           for row in metrics.get(key, []) if row.get("response_id")}
    calls = openhands_model_calls(trace)
    ids.update(str(call["response_id"]) for call in calls if not str(call["response_id"]).startswith("event:"))
    return max(len(ids), len(calls), len(metrics.get("response_latencies") or []), len(metrics.get("token_usages") or []))

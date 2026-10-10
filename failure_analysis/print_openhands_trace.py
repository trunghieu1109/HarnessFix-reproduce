#!/usr/bin/env python3
"""Print a bounded OpenHands timeline or targeted request/action/result evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from failure_analysis.openhands_trace import event_response_key, openhands_api_call_count, openhands_model_calls
from failure_analysis.secret_redaction import redact_secrets


METADATA_EVENTS = {"LLMCompletionLogEvent", "ConversationStateUpdateEvent"}


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _bounded(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return _clip(value, limit)
    if isinstance(value, dict):
        return {key: _bounded(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_bounded(item, limit) for item in value]
    return value


def _tool_call_id(event: dict) -> str | None:
    return (event.get("tool_call_id") or (event.get("action") or {}).get("tool_call_id")
            or (event.get("observation") or {}).get("tool_call_id"))


def _tool_name(tool: dict) -> str | None:
    return (tool.get("function", {}).get("name") or tool.get("mcp_tool", {}).get("name")
            or tool.get("name") or tool.get("title"))


def _event_line(index: int, event: dict, preview_limit: int) -> str:
    action = event.get("action") or {}
    observation = event.get("observation") or {}
    parts = [f"[{index}] {event.get('kind', 'unknown')}"]
    for label, value in (
        ("source", event.get("source")),
        ("response_id", event.get("llm_response_id")),
        ("tool_call_id", _tool_call_id(event)),
        ("tool", event.get("tool_name") or observation.get("tool_name")),
        ("action", action.get("kind")),
    ):
        if value:
            parts.append(f"{label}={value}")
    if action:
        arguments = action.get("data", {key: value for key, value in action.items() if key != "kind"})
        parts.append("args=" + _clip(arguments, preview_limit))
    elif observation:
        for key in ("is_error", "exit_code"):
            if key in observation:
                parts.append(f"{key}={observation[key]}")
        content = observation.get("content", observation)
        if isinstance(content, list):
            content = " ".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
        parts.append("result=" + _clip(content, preview_limit))
    else:
        message = event.get("llm_message") or event.get("message") or {}
        content = message.get("content") or message.get("reasoning_content") or event.get("error") or event.get("system_prompt")
        if content:
            parts.append("content=" + _clip(content, preview_limit))
    return " ".join(parts)


def select_evidence(trace: dict, event_indices: list[int], response_ids: list[str]) -> tuple[list[tuple[int, dict]], list[dict]]:
    """Include the selected calls and their actions/results, paired by tool_call_id."""
    events = trace.get("events") or []
    calls = openhands_model_calls(trace)
    for index in event_indices:
        if index < 0 or index >= len(events):
            raise ValueError(f"event index {index} is outside 0..{len(events) - 1}")
    known_ids = {str(call["response_id"]) for call in calls}
    for response_id in response_ids:
        if response_id not in known_ids:
            raise ValueError(f"response_id {response_id!r} is not recorded; inspect the timeline first")
    selected_indices = set(event_indices)
    selected_ids = set(response_ids)
    paired_ids = {_tool_call_id(events[index]) for index in selected_indices} - {None}
    for index, event in enumerate(events):
        if index in selected_indices or (_tool_call_id(event) and _tool_call_id(event) in paired_ids):
            selected_indices.add(index)
            if event.get("llm_response_id"):
                selected_ids.add(event["llm_response_id"])
    for index, event in enumerate(events):
        if event.get("llm_response_id") in selected_ids:
            selected_indices.add(index)
            if _tool_call_id(event):
                paired_ids.add(_tool_call_id(event))
    selected_indices.update(index for index, event in enumerate(events) if _tool_call_id(event) in paired_ids)
    selected_calls = [call for call in calls if str(call["response_id"]) in selected_ids]
    return [(index, events[index]) for index in sorted(selected_indices)], selected_calls


def _call_projection(call: dict, message_limit: int, field_limit: int) -> dict:
    messages = call.get("request_messages") or []
    payload = call.get("request_payload") or {}
    response = call.get("response_message") or {}
    used_tools = {_tool_name(tool) for tool in response.get("tool_calls") or []} - {None}
    tools = payload.get("tools") or []
    projection = {
        "response_id": call["response_id"],
        "source_ref": call.get("source_ref"),
        "request_evidence": call.get("request_evidence"),
        "response_evidence": call.get("response_evidence"),
        "request_messages_count": len(messages),
        "request_messages_shown": min(len(messages), message_limit),
        "request_messages": messages[-message_limit:],
        "response_message": response,
        "available_tool_count": len(tools),
        "used_tool_names": sorted(used_tools),
        "used_tool_schemas": [tool for tool in tools if _tool_name(tool) in used_tools],
    }
    if len(messages) > message_limit:
        projection["request_messages_note"] = "Earlier request messages omitted; use --message-limit to expand only if needed."
    for key in ("instructions", "input"):
        if key in payload:
            value = payload[key]
            projection[f"request_payload.{key}"] = value[-message_limit:] if isinstance(value, list) else value
    for key in ("original_response_message", "generation_settings", "usage", "error"):
        if call.get(key):
            projection[key] = call[key]
    bounded = _bounded(projection, field_limit)
    # Identifiers and artifact paths must remain usable when text previews shrink.
    for key in ("response_id", "source_ref", "request_evidence", "response_evidence"):
        bounded[key] = projection[key]
    return bounded


def render_trace(trace: dict, *, event_indices: list[int] | None = None, response_ids: list[str] | None = None,
                 message_limit: int = 4, field_limit: int = 1200, char_limit: int = 12000) -> str:
    trace = redact_secrets(trace)
    events = trace.get("events") or []
    calls = openhands_model_calls(trace)
    event_response_ids = {event_response_key(event, index) for index, event in enumerate(events)
                          if event.get("kind") == "ActionEvent"
                          or (event.get("kind") == "MessageEvent" and event.get("source") == "agent")}
    header = "OPENHANDS TRACE SUMMARY\n" + json.dumps({
        "events": len(events),
        "model_call_records": len(calls),
        "task_api_calls": openhands_api_call_count(trace),
        "recorded_requests": sum(call.get("request_evidence") == "sdk_completion_log" for call in calls),
        "model_calls_without_events": [call["response_id"] for call in calls if str(call["response_id"]) not in event_response_ids],
        "error": _clip(trace.get("error"), 500) if trace.get("error") else None,
        "eval_output": _clip(trace.get("eval_output"), 500) if trace.get("eval_output") else None,
        "subagent_event_counts": {name: len(data.get("events") or []) for name, data in (trace.get("subagents") or {}).items()},
    }, ensure_ascii=False) + "\n"
    if event_indices or response_ids:
        selected_events, selected_calls = select_evidence(trace, event_indices or [], response_ids or [])

        def render_details(limit: int) -> str:
            lines = [header, f"SELECTED EVENTS (paired actions/results; text fields limited to {limit} chars)"]
            for index, event in selected_events:
                # Shared response reasoning is shown once in the model-call record.
                payload = {key: value for key, value in event.items()
                           if key not in {"thought", "reasoning_content", "llm_response", "log_data"}}
                lines.append(json.dumps({"event_index": index, "source_ref": f"trace.events[{index}]",
                                         "event": _bounded(payload, limit)}, ensure_ascii=False))
            lines.append("SELECTED MODEL CALLS (request_messages uses the plural key)")
            lines.extend(json.dumps(_call_projection(call, message_limit, limit), ensure_ascii=False) for call in selected_calls)
            return "\n".join(lines) + "\n"

        output = render_details(field_limit)
        preview_limit = field_limit
        while char_limit > 0 and len(output) > char_limit and preview_limit > 200:
            preview_limit = max(200, preview_limit * 3 // 4)
            output = render_details(preview_limit)
    else:
        visible = [(index, event) for index, event in enumerate(events) if event.get("kind") not in METADATA_EVENTS]
        prefix = header + "TIMELINE (original event indices; completion/state metadata excluded)\n"
        lines = [_event_line(index, event, 180) for index, event in visible]
        output = prefix + "\n".join(lines) + "\n"
        # Shrink previews before dropping chronology from the summary.
        if char_limit > 0 and len(output) > char_limit:
            bare_lines = [_event_line(index, event, 0) for index, event in visible]
            budget = max(0, char_limit - len(prefix) - sum(len(line) + 1 for line in bare_lines))
            preview_limit = min(180, budget // max(1, len(visible)))
            output = prefix + "\n".join(_event_line(index, event, preview_limit) for index, event in visible) + "\n"
    if char_limit > 0 and len(output) > char_limit:
        marker = "\n... [output truncated; select fewer response IDs/events or raise --char-limit]\n"
        output = output[: max(0, char_limit - len(marker))] + marker
        return output[:char_limit]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--event-index", type=int, action="append", default=[])
    parser.add_argument("--response-id", action="append", default=[])
    parser.add_argument("--message-limit", type=int, default=4)
    parser.add_argument("--field-limit", type=int, default=1200)
    parser.add_argument("--char-limit", type=int, default=12000)
    args = parser.parse_args()
    if args.message_limit < 1 or args.field_limit < 1 or args.char_limit < 1:
        parser.error("message, field, and character limits must be positive")
    data = json.loads(args.trace_path.read_text())
    trace = data if isinstance(data, dict) else {"events": data}
    try:
        output = render_trace(trace, event_indices=args.event_index, response_ids=args.response_id,
                              message_limit=args.message_limit, field_limit=args.field_limit, char_limit=args.char_limit)
    except ValueError as exc:
        parser.error(str(exc))
    print(output, end="")


if __name__ == "__main__":
    main()

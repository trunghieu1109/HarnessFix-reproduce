#!/usr/bin/env python3
"""Probe configured OpenAI-compatible models without printing API keys.

Run from HarnessFix: python scripts/probe_models.py
Reports the actual request parameters and reasoning metadata, without assuming
that a gateway applies temperature or exposes internal reasoning.

For the public Qwen HTTP endpoint, use:
  python scripts/probe_models.py --config configs/qwen.yaml --no-auth \
      --api-base http://151.237.25.16:27410/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import requests
import yaml
from dotenv import dotenv_values


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Calculate (173 * 29) - (81 * 47). Reply with only the final integer."
EXPECTED_ANSWER = "1210"


def reasoning_metadata(message: dict, usage: dict) -> dict:
    fields = ("reasoning_content", "reasoning", "reasoning_details")
    populated = [field for field in fields if message.get(field)]
    content = message.get("content") or ""
    inline = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    token_counts = {}
    for group in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(group) or {}
        if details.get("reasoning_tokens") is not None:
            token_counts[f"{group}.reasoning_tokens"] = details["reasoning_tokens"]
    for field in ("reasoning_tokens", "thoughts_token_count", "thoughtsTokenCount"):
        if usage.get(field) is not None:
            token_counts[field] = usage[field]
    observed = bool(populated or any(text.strip() for text in inline)
                    or any(count > 0 for count in token_counts.values()))
    return {
        "observed": observed,
        "verdict": "observed" if observed else (
            "zero_reported_tokens" if token_counts else "not_exposed"
        ),
        "fields_present": [field for field in fields if field in message],
        "populated_fields": populated,
        "inline_thinking_present": any(text.strip() for text in inline),
        "token_counts": token_counts,
        "note": "No exposed reasoning does not prove that internal reasoning is disabled.",
    }


def error_text(value: str, api_key: str | None) -> str:
    return (value.replace(api_key, "<redacted>") if api_key else value)[:1000]


def temperature_fields(value, path: str = "$") -> dict:
    """Locate temperature fields at any depth, retaining zero and null values."""
    found = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if "temperature" in key.lower():
                found[child_path] = child
            found.update(temperature_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(temperature_fields(child, f"{path}[{index}]"))
    return found


def redact_secret(value, api_key: str | None):
    if isinstance(value, dict):
        return {key: redact_secret(child, api_key) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_secret(child, api_key) for child in value]
    if isinstance(value, str) and api_key:
        return value.replace(api_key, "<redacted>")
    return value


def response_evidence(body: dict, headers: dict, api_key: str | None,
                      include_raw: bool) -> dict:
    body = redact_secret(body, api_key)
    sensitive_headers = {"set-cookie", "authorization", "proxy-authorization",
                         "x-api-key", "x-goog-api-key"}
    headers = {
        key: "<redacted>" if key.lower() in sensitive_headers else redact_secret(value, api_key)
        for key, value in headers.items()
    }
    evidence = {
        "body_keys": list(body),
        "temperature_fields": temperature_fields(body),
        "body_contains_temperature_text": "temperature" in json.dumps(body).lower(),
        "headers": headers,
        "header_temperature_fields": temperature_fields(headers, "headers"),
    }
    if include_raw:
        evidence["raw_json"] = body
    return evidence


def probe(config_path: Path, env: dict, args: argparse.Namespace) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config["model"].startswith("openai/"):
        raise ValueError("This probe requires an openai/ model and an HTTP(S) api_base.")
    api_key = None if args.no_auth else env[config["api_key_env"]]
    if not args.no_auth and not api_key:
        raise ValueError(f"Missing credential: {config['api_key_env']}")
    api_base = (args.api_base or config["api_base"]).rstrip("/")
    payload = {
        "model": config["model"].removeprefix("openai/"),
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": float(config["temperature"]),
        "max_tokens": min(args.max_tokens, int(config["max_output_tokens"])),
        "stream": False,
    }
    if config.get("reasoning_effort") is not None:
        payload["reasoning_effort"] = config["reasoning_effort"]
    # OpenAI extra_body fields belong at the top level of the HTTP JSON body.
    payload.update(config.get("extra_body", {}))
    result = {
        "name": config["name"],
        "config": str(config_path),
        "configured_api_base": config["api_base"],
        "api_base": api_base,
        "authentication": "none" if args.no_auth else "bearer",
        "ngrok_skip_browser_warning": args.ngrok_skip_browser_warning,
        "request": payload,
        "temperature_note": "Request temperature; server application cannot be inferred from answer text.",
    }
    timeout = (min(10, args.timeout), args.timeout)
    with requests.Session() as session:
        if args.no_auth:
            session.trust_env = False  # Also disable implicit .netrc credentials.
        if api_key is not None:
            session.headers.update({"Authorization": f"Bearer {api_key}"})
        if args.ngrok_skip_browser_warning:
            session.headers["ngrok-skip-browser-warning"] = "1"
        if not args.skip_discovery:
            try:
                response = session.get(f"{api_base}/models", timeout=timeout,
                                       allow_redirects=False)
                discovery = {"http_status": response.status_code,
                             "content_type": response.headers.get("Content-Type")}
                if response.ok:
                    body = response.json()
                    ids = [model["id"] for model in body["data"]]
                    discovery.update(model_count=len(ids),
                                     requested_model_present=payload["model"] in ids)
                    discovery["response_evidence"] = response_evidence(
                        body, response.headers, api_key, args.include_raw_response,
                    )
                else:
                    discovery["error"] = error_text(response.text, api_key)
                result["discovery"] = discovery
            except (requests.RequestException, ValueError, KeyError) as exc:
                result["discovery"] = {"error": error_text(str(exc), api_key)}

        started = perf_counter()
        try:
            response = session.post(f"{api_base}/chat/completions", json=payload,
                                    timeout=timeout, allow_redirects=False)
            completion = {"http_status": response.status_code, "ok": False,
                          "content_type": response.headers.get("Content-Type")}
            if response.ok:
                body = response.json()
                choice = body["choices"][0]
                message = choice["message"]
                usage = body.get("usage") or {}
                answer = re.sub(r"<think>.*?</think>", "", message.get("content") or "",
                                flags=re.DOTALL).strip()
                completion.update(
                    ok=bool(answer),
                    response_model=body.get("model"),
                    finish_reason=choice.get("finish_reason"),
                    answer=answer,
                    expected_answer=EXPECTED_ANSWER,
                    answer_correct=answer == EXPECTED_ANSWER,
                    usage=usage,
                    server_reported_temperature=body.get("temperature"),
                    reasoning=reasoning_metadata(message, usage),
                    response_evidence=response_evidence(
                        body, response.headers, api_key, args.include_raw_response,
                    ),
                )
            else:
                completion["error"] = error_text(response.text, api_key)
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            completion = {"ok": False, "error": error_text(str(exc), api_key)}
        completion["elapsed_seconds"] = round(perf_counter() - started, 3)
        result["completion"] = completion
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", default=[
        REPO_ROOT / "configs/qwen.yaml", REPO_ROOT / "configs/gemini.yaml",
    ])
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--api-base", help="Override the endpoint for a single config.")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--include-raw-response", action="store_true",
                        help="Save complete JSON responses and inspect nested temperature fields.")
    parser.add_argument("--ngrok-skip-browser-warning", action="store_true",
                        help="Send ngrok-skip-browser-warning: 1 to bypass the browser interstitial.")
    parser.add_argument("--no-auth", action="store_true",
                        help="Do not read .env or send an Authorization header.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts/model_probe.json")
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_tokens <= 0:
        parser.error("--timeout and --max-tokens must be positive")
    if args.api_base and len(args.config) != 1:
        parser.error("--api-base requires exactly one --config")

    # Match the SLM loader and OpenHands bridge: .env values override the shell.
    env = {} if args.no_auth else dict(os.environ) | dict(dotenv_values(args.env_file))
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "models": []}
    for config_path in args.config:
        result = probe(config_path.resolve(), env, args)
        report["models"].append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Report: {args.output.resolve()}")
    return 0 if all(model["completion"]["ok"] for model in report["models"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Configure native or OpenAI-compatible models for HarnessFix and Better Harness."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import dotenv_values, set_key


REPO_ROOT = Path(__file__).resolve().parents[1]


def configure(repo_root: Path, better_root: Path, config_path: Path, *, analysis_only: bool = False) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    alias = config["name"]
    provider = model.split("/", 1)[0]
    api_base = (config.get("api_base") or "").rstrip("/") or None
    if not model.startswith(("openai/", "gemini/")) or not model.split("/", 1)[1] or "REPLACE_" in model:
        raise ValueError("Set model to openai/<gateway model ID> or gemini/<native Gemini model ID>.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
        raise ValueError("The model alias must contain only letters, digits, _, . or -.")
    if provider == "openai":
        endpoint = urlparse(api_base or "")
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise ValueError("api_base must be an HTTP(S) endpoint including /v1.")
        if "MODEL_SERVER_HOST" in api_base or not endpoint.path.endswith("/v1"):
            raise ValueError("Replace api_base with your gateway/server URL, ending in /v1.")
        if endpoint.username or endpoint.password:
            raise ValueError("Use an API key environment variable, rather than credentials in the URL.")
    elif api_base is not None:
        raise ValueError("Use api_base: null for native Gemini. Use openai/ for a /v1 gateway.")
    input_tokens = int(config["max_input_tokens"])
    output_tokens = int(config["max_output_tokens"])
    if min(input_tokens, output_tokens) <= 0:
        raise ValueError("Token budgets must be positive.")
    if not (better_root / "src" / "collect.py").is_file():
        raise FileNotFoundError(f"Better Harness checkout not found: {better_root}")

    # Prepare all content before updating any files. Preserve existing aliases,
    # credentials and prompt bodies. Better's loader needs a literal model ID.
    key_env = config.get("api_key_env", "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
        raise ValueError("api_key_env must be a valid environment variable name.")
    api_key = (
        os.environ.get(key_env)
        or dotenv_values(repo_root / ".env").get(key_env)
        or dotenv_values(better_root / ".env").get(key_env)
        or ("EMPTY" if provider == "openai" else "")
    )
    if not api_key:
        raise ValueError(f"Set {key_env} before configuring native Gemini.")
    settings = {
        "temperature": float(config["temperature"]),
        "max_tokens": output_tokens,
        "stream": False,
        "timeout": 300,
        "max_retries": 0,
        "drop_params": True,
        "api_base": api_base,
    }
    if config.get("reasoning_effort") is not None:
        settings["reasoning_effort"] = config["reasoning_effort"]
    if "extra_body" in config:
        settings["extra_body"] = config["extra_body"]

    models_path = better_root / "configs" / "models.yaml"
    models = yaml.safe_load(models_path.read_text()) if models_path.exists() else {"models": []}
    entries = [entry for entry in models["models"] if entry["name"] != alias]
    entries.append({
        "name": alias,
        "model": model,
        "api_base": api_base,
        "api_key": "${" + key_env + "}",
        "temperature": settings["temperature"],
        "max_input_tokens": input_tokens,
        "max_output_tokens": output_tokens,
        "reasoning_effort": config.get("reasoning_effort"),
    })
    if "extra_body" in settings:
        entries[-1]["extra_body"] = settings["extra_body"]
    if provider == "openai":
        entries[-1].update(input_cost_per_token=0.0, output_cost_per_token=0.0)
    models["models"] = entries

    registry_path = repo_root / "task_agent" / "model_registry.json"
    registry = json.loads(registry_path.read_text())
    registry[model] = registry.get(model, {}) | {
        "max_tokens": input_tokens + output_tokens,
        "max_input_tokens": input_tokens,
        "max_output_tokens": output_tokens,
        "litellm_provider": provider,
        "mode": "chat",
        "api_base": api_base,
        "api_key_env": key_env,
        "config_name": "swebench_xml.yaml",
        "model_class": "litellm_textbased",
        "model_kwargs_override": settings,
        "cost_tracking": "ignore_errors",
    }
    if provider == "openai":
        registry[model].update(input_cost_per_token=0.0, output_cost_per_token=0.0)

    updates = {}
    for relative in ("failure_analysis/analysis_config_swe.yaml", "enhancement_implementation/config_swe.yaml"):
        path = repo_root / relative
        original = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(original)
        kwargs = dict(parsed["model"]["model_kwargs"])
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("extra_body", None)
        kwargs.update(settings)
        # model_kwargs is the final block in these two configs. Replacing only
        # that block keeps the lengthy agent prompts byte-for-byte intact.
        prefix, count = re.subn(r"(?m)^  model_kwargs:\s*\n[\s\S]*\Z", "", original)
        if count != 1:
            raise ValueError(f"Expected one final model_kwargs block in {path}")
        block = yaml.safe_dump({"model_kwargs": kwargs}, sort_keys=False)
        updates[path] = prefix + "\n".join("  " + line for line in block.splitlines()) + "\n"

    env_values = {
        key_env: api_key,
        "HARNESSFIX_ANALYSIS_MODEL": model,
        "HARNESSFIX_ANALYSIS_MODEL_NAME": alias,
        "BETTER_HARNESS_ROOT": str(better_root),
    }
    if not analysis_only:
        env_values.update(
            HARNESSFIX_TASK_MODEL=model,
            HARNESSFIX_MODEL_NAME=alias,
            HARNESSFIX_MAX_OUTPUT_TOKENS=str(output_tokens),
            HARNESSFIX_MODEL_TEMPERATURE=str(settings["temperature"]),
        )
    if provider == "openai" and not analysis_only:
        env_values.update(VLLM_MODEL=model, OPENAI_API_BASE=api_base, OPENAI_API_KEY=api_key)
    elif provider == "gemini":
        env_values["GEMINI_API_KEY"] = api_key
    for root in (repo_root, better_root):
        env_path = root / ".env"
        if not env_path.exists():
            template = root / ".env.example"
            env_path.write_text(template.read_text() if template.exists() else "", encoding="utf-8")
        env_path.chmod(0o600)
        for key, value in env_values.items():
            if analysis_only and key == key_env and dotenv_values(env_path).get(key):
                continue
            set_key(str(env_path), key, value)

    models_path.parent.mkdir(parents=True, exist_ok=True)
    models_path.write_text(yaml.safe_dump(models, sort_keys=False), encoding="utf-8")
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    for path, contents in updates.items():
        path.write_text(contents, encoding="utf-8")
    print(f"Configured alias {alias}: {model}")
    print(f"Endpoint: {api_base or 'native Gemini API'}")
    print(f"Updated {models_path}, both .env files, SWE registry and analysis/repair settings.")
    if provider == "openai":
        print("Self-hosted pricing is set to zero; reported dollar costs are not measured server costs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "qwen.yaml")
    parser.add_argument("--analysis-config", type=Path, help="Optional separate model for analysis/planning/repair.")
    parser.add_argument("--analysis-only", action="store_true", help="Configure only analysis/planning/repair, preserving task settings and existing credentials.")
    parser.add_argument("--better-root", type=Path, default=REPO_ROOT.parent / "slm-harness-adaptation-reproduce")
    args = parser.parse_args()
    if args.analysis_only and args.analysis_config:
        parser.error("--analysis-only cannot be combined with --analysis-config")
    configure(REPO_ROOT, args.better_root.resolve(), args.config.resolve(), analysis_only=args.analysis_only)
    if args.analysis_config:
        configure(REPO_ROOT, args.better_root.resolve(), args.analysis_config.resolve(), analysis_only=True)
        task = yaml.safe_load(args.config.read_text())
        analysis = yaml.safe_load(args.analysis_config.read_text())
        for root in (REPO_ROOT, args.better_root.resolve()):
            for key, value in {
                "HARNESSFIX_TASK_MODEL": task["model"],
                "HARNESSFIX_MODEL_NAME": task["name"],
            }.items():
                set_key(str(root / ".env"), key, value)
        print(f"Task model: {task['model']}; analysis/repair model: {analysis['model']}")


if __name__ == "__main__":
    main()

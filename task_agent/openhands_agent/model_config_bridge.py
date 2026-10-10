"""Load OpenHands analysis models from Better Harness's model configuration."""

import json
import os
from pathlib import Path

import litellm
import yaml
from dotenv import dotenv_values

MODEL_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "task_agent" / "model_registry.json"

def configured_connection_kwargs(model_name: str, kwargs: dict) -> dict:
    """Apply model-specific settings without replacing task environment globals."""
    result = dict(kwargs)
    path = MODEL_REGISTRY_PATH
    root = path.parent.parent
    entry = json.loads(path.read_text()).get(model_name, {}) if path.is_file() else {}
    result.update(entry.get("model_kwargs_override") or {})
    key_env = entry.get("api_key_env")
    if key_env and not result.get("api_key"):
        key = os.environ.get(key_env) or dotenv_values(root / ".env").get(key_env)
        if key:
            result["api_key"] = key
    return result


def load_model(alias: str) -> dict:
    better_root = Path(os.environ["BETTER_HARNESS_ROOT"])
    models_path = better_root / "configs" / "models.yaml"
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
    model = {entry["name"]: entry for entry in models}[alias]
    dotenv = dotenv_values(better_root / ".env")

    def resolve(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            name = value[2:-1]
            return dotenv[name] if name in dotenv else os.environ[name]
        return value

    resolved = {
        "model": resolve(model["model"]),
        "api_base": resolve(model["api_base"]),
        "api_key": resolve(model["api_key"]),
        "reasoning_effort": resolve(model["reasoning_effort"]),
        "max_input_tokens": int(resolve(model["max_input_tokens"])),
        "max_output_tokens": int(resolve(model["max_output_tokens"])),
    }
    if "temperature" in model:
        resolved["temperature"] = float(resolve(model["temperature"]))
    if "extra_body" in model:
        resolved["extra_body"] = model["extra_body"]
    provider = litellm.get_llm_provider(resolved["model"])[1]
    litellm.register_model({
        resolved["model"]: {
            "max_tokens": resolved["max_input_tokens"],
            "max_input_tokens": resolved["max_input_tokens"],
            "max_output_tokens": resolved["max_output_tokens"],
            "litellm_provider": provider,
            "mode": "chat",
        }
    })
    return resolved


def selected_model_kwargs() -> dict:
    model = load_model(os.environ["HARNESSFIX_MODEL_ALIAS"])
    kwargs = {
        "api_base": model["api_base"],
        "api_key": model["api_key"],
        "reasoning_effort": model["reasoning_effort"],
        "max_tokens": model["max_output_tokens"],
    }
    if "temperature" in model:
        kwargs["temperature"] = model["temperature"]
    if "extra_body" in model:
        kwargs["extra_body"] = model["extra_body"]
    return kwargs

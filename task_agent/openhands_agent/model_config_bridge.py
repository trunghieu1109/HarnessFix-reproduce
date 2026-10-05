"""Load OpenHands analysis models from Better Harness's model configuration."""

import os
from pathlib import Path

import litellm
import yaml
from dotenv import dotenv_values


def load_model(alias: str) -> dict:
    better_root = Path(os.environ["BETTER_HARNESS_ROOT"])
    models_path = better_root / "configs" / "models.yaml"
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
    model = {entry["name"]: entry for entry in models}[alias]
    dotenv = dotenv_values(better_root / ".env")

    def resolve(value: str) -> str:
        if value.startswith("${") and value.endswith("}"):
            name = value[2:-1]
            return dotenv[name] if name in dotenv else os.environ[name]
        return value

    resolved = {
        "model": model["model"],
        "api_base": resolve(model["api_base"]),
        "api_key": resolve(model["api_key"]),
        "reasoning_effort": model["reasoning_effort"],
        "max_input_tokens": model["max_input_tokens"],
        "max_output_tokens": model["max_output_tokens"],
    }
    litellm.register_model({
        resolved["model"]: {
            "max_tokens": resolved["max_input_tokens"],
            "max_input_tokens": resolved["max_input_tokens"],
            "max_output_tokens": resolved["max_output_tokens"],
            "litellm_provider": "openai",
            "mode": "chat",
        }
    })
    return resolved


def selected_model_kwargs() -> dict:
    model = load_model(os.environ["HARNESSFIX_MODEL_ALIAS"])
    return {
        "api_base": model["api_base"],
        "api_key": model["api_key"],
        "reasoning_effort": model["reasoning_effort"],
    }

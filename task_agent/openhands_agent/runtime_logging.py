"""Apply SDK logging and generation fixes inside packed candidate processes.

SDK 1.19 validates the LLM inside RegistryEvent and rebuilds its telemetry.
Remote logging callbacks installed before lazy agent initialization are lost.
This shim runs only in the packed candidate's process and keeps the registry's
intentional separation of metrics for copied LLMs.
"""

from __future__ import annotations

import sys
from functools import wraps


def install_sdk_logging_fix() -> None:
    module = sys.modules.get("openhands.sdk.llm.llm_registry")
    if module is None:
        return
    original_add = module.LLMRegistry.add
    if getattr(original_add, "_harnessfix_logging_fix", False):
        return

    @wraps(original_add)
    def add_with_logging(registry, llm):
        telemetry = llm.telemetry
        completion_callback = telemetry._log_completions_callback
        stats_callback = telemetry._stats_update_callback
        independent = id(llm.metrics) not in registry._metrics_ids
        result = original_add(registry, llm)
        # Shared telemetry belongs to another usage ID and must stay detached.
        if independent:
            current = llm.telemetry
            if completion_callback is not None and current._log_completions_callback is None:
                current.set_log_completions_callback(completion_callback)
            if stats_callback is not None and current._stats_update_callback is None:
                current.set_stats_update_callback(stats_callback)
        return result

    add_with_logging._harnessfix_logging_fix = True
    module.LLMRegistry.add = add_with_logging


def install_sdk_generation_config_fix() -> None:
    """Keep Qwen temperature when its vLLM chat template disables thinking.

    SDK 1.19 removes temperature from all non-Gemini reasoning models, even
    when Qwen's chat template explicitly switches thinking off. Install this
    in the packed launcher so Docker and local candidates use the same settings.
    """
    module = sys.modules.get("openhands.sdk.llm.llm")
    if module is None:
        return
    original_select = module.select_chat_options
    if getattr(original_select, "_harnessfix_generation_fix", False):
        return

    @wraps(original_select)
    def select_with_thinking_settings(llm, user_kwargs, has_tools):
        result = original_select(llm, user_kwargs, has_tools)
        template_kwargs = llm.litellm_extra_body.get("chat_template_kwargs", {})
        if "qwen" in llm.model.lower() and template_kwargs.get("enable_thinking") is False:
            temperature = user_kwargs.get("temperature", llm.temperature)
            if temperature is not None:
                result["temperature"] = temperature
        return result

    select_with_thinking_settings._harnessfix_generation_fix = True
    module.select_chat_options = select_with_thinking_settings

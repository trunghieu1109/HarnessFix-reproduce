"""Keep credentials out of repair-agent observations and persisted trajectories."""

from __future__ import annotations

import os

from failure_analysis.secret_redaction import is_credential_name, redact_secrets
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment


class PromptSafeLocalEnvironment(LocalEnvironment):
    def __init__(self, *, observation_max_chars: int = 0, **kwargs):
        self.observation_max_chars = observation_max_chars
        env = dict(kwargs.pop("env", {}))
        for name in os.environ.keys() | env.keys():
            if is_credential_name(name):
                env[name] = ""
        super().__init__(env=env, **kwargs)

    def _check_finished(self, output: dict):
        cleaned = redact_secrets(output)
        output.clear()
        output.update(cleaned)
        # Redact before the submission marker raises Submitted.
        super()._check_finished(output)
        text = output.get("output", "")
        if self.observation_max_chars > 0 and len(text) > self.observation_max_chars:
            head = self.observation_max_chars // 2
            tail = self.observation_max_chars - head
            output["output"] = text[:head] + "\n[Observation truncated; select a narrower section.]\n" + text[-tail:]

    def get_template_vars(self, **kwargs) -> dict:
        return redact_secrets(super().get_template_vars(**kwargs))


class PromptSafeAgent(DefaultAgent):
    def add_messages(self, *messages: dict) -> list[dict]:
        return super().add_messages(*redact_secrets(list(messages)))

    def serialize(self, *extra_dicts) -> dict:
        return redact_secrets(super().serialize(*extra_dicts))

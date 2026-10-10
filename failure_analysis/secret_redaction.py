"""Redact credentials from model-facing evidence without dropping task semantics."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

REDACTED = "[REDACTED]"
REPO_ROOT = Path(__file__).resolve().parent.parent
CREDENTIAL_SUFFIXES = ("apikey", "accesstoken", "refreshtoken", "authtoken", "clientsecret", "privatekey", "password", "passwd")
CREDENTIAL_KEYS = {"authorization", "proxyauthorization", "token", "secret", "pwd"}
DIRECTORY_ENV_KEYS = {"PWD", "OLDPWD"}
QUOTED_CREDENTIAL = re.compile(
    r"(?i)(\b[\w-]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd)[\"']?\s*[:=]\s*)([\"'])([^\n]*?)\2"
)
BEARER_CREDENTIAL = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_.~+/-]+=*")


def is_credential_name(name: str) -> bool:
    # Preserve shell directory variables before case folding. A lowercase "pwd"
    # field in task data still denotes a password and must remain redacted.
    if name in DIRECTORY_ENV_KEYS:
        return False
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized in CREDENTIAL_KEYS or normalized.endswith(CREDENTIAL_SUFFIXES)


def known_secret_values() -> set[str]:
    better_root = Path(os.environ.get("BETTER_HARNESS_ROOT", REPO_ROOT.parent / "slm-harness-adaptation-reproduce"))
    settings = dict(os.environ)
    values = set()
    for root in (REPO_ROOT, better_root):
        env_path = root / ".env"
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and is_credential_name(key.strip()):
                value = value.strip()
                if value.startswith(("'", '"')):
                    value = value[1:].split(value[0], 1)[0]
                else:
                    value = value.split(" #", 1)[0].strip()
                if value:
                    values.add(value)
    values.update(value for key, value in settings.items() if is_credential_name(key) and value)
    return {value for value in values if len(value) >= 8 and not value.startswith("${") and value != REDACTED}


def redact_secrets(value: Any, *, secrets: set[str] | None = None) -> Any:
    values = known_secret_values() if secrets is None else secrets

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: REDACTED if is_credential_name(str(key)) and child not in (None, "") else clean(child)
                    for key, child in item.items()}
        if isinstance(item, list):
            return [clean(child) for child in item]
        if not isinstance(item, str):
            return item
        text = item
        # Arguments and log_data often embed JSON rather than structured objects.
        if text.lstrip().startswith(("{", "[")):
            try:
                embedded = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                cleaned = clean(embedded)
                if cleaned != embedded:
                    text = json.dumps(cleaned, ensure_ascii=False)
        for secret in sorted(values, key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        text = QUOTED_CREDENTIAL.sub(lambda match: match[1] + match[2] + REDACTED + match[2], text)
        return BEARER_CREDENTIAL.sub("Bearer " + REDACTED, text)

    return clean(value)

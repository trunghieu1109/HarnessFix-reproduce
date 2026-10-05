from __future__ import annotations

import importlib
import json
from pathlib import Path

from openhands.sdk import Agent, AgentContext, Tool
from openhands.sdk.context import Skill
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool


BUNDLE_ROOT = Path(__file__).resolve().parent


def _config() -> dict:
    return json.loads((BUNDLE_ROOT / "config.json").read_text(encoding="utf-8"))


def _optional_text(relative_path: str) -> str | None:
    path = BUNDLE_ROOT / relative_path
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _skills() -> list[Skill]:
    skill_dir = BUNDLE_ROOT / "skills"
    return [
        Skill(name=str(path.relative_to(skill_dir)), content=path.read_text(encoding="utf-8"), trigger=None)
        for path in sorted(skill_dir.rglob("*.md"))
        if path.is_file()
    ]


def _mcp_config(task_id: str, base_dir: str) -> dict:
    module_names = {
        "machine_operating_s2l": "src.task_setups.machine_operating_s2l",
        "woocommerce_stock_alert_s2l": "src.task_setups.woocommerce_stock_alert_s2l",
    }
    module_name = module_names.get(task_id)
    if module_name is None:
        return {}
    module = importlib.import_module(module_name)
    return module.get_mcp_config(base_dir)


def _tools(task_id: str, base_dir: str) -> list[Tool]:
    if task_id == "webarena":
        from openhands.tools.browser_use import BrowserToolSet

        browser_root = Path(base_dir) / ".browser_use"
        return [
            Tool(
                name=BrowserToolSet.name,
                params={
                    "user_data_dir": str(browser_root / "profile"),
                    "downloads_path": str(browser_root / "downloads"),
                },
            )
        ]
    return [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)]


def build_agent(base_dir, llm):
    config = _config()
    task_id = config["task_id"]
    prompt_path = Path(base_dir) / "system_prompt.md"
    prompt_path.write_text((BUNDLE_ROOT / "prompts" / "system.md").read_text(encoding="utf-8"), encoding="utf-8")
    kwargs = {
        "llm": llm,
        "tools": _tools(task_id, str(base_dir)),
        "system_prompt_filename": str(prompt_path),
    }
    skills = _skills()
    system_suffix = _optional_text("context/system_suffix.md")
    user_suffix = _optional_text("context/user_suffix.md")
    if skills or system_suffix or user_suffix:
        kwargs["agent_context"] = AgentContext(
            skills=skills,
            system_message_suffix=system_suffix,
            user_message_suffix=user_suffix,
        )
    mcp_config = _mcp_config(task_id, str(base_dir))
    if mcp_config:
        kwargs["mcp_config"] = mcp_config
    return Agent(**kwargs)


def get_workspace_scripts() -> dict[str, str]:
    scripts_root = BUNDLE_ROOT / "workspace_scripts"
    return {
        str(path.relative_to(scripts_root)).replace("\\", "/"): path.read_text(encoding="utf-8")
        for path in sorted(scripts_root.rglob("*"))
        if path.is_file() and path.name != ".gitkeep"
    }

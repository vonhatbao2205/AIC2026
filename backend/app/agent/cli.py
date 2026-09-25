"""How Codex CLI and Claude Code CLI are launched for one agent run, and how
their JSONL event streams are read back into progress the console can show.

Both are locked down the same way: no shell and no file tools, only the `aic`
MCP bridge, whose calls are pre-approved because nobody is at a terminal to
approve them. Each run is a fresh process, so it authenticates with whichever
account the CLI is logged into at that moment — switching accounts between
searches needs no restart here.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .prompt import SYSTEM_PROMPT

if TYPE_CHECKING:
    from ..config import Settings
    from .runs import AgentState

MCP_SERVER_NAME = "aic"
MCP_SCRIPT = Path(__file__).with_name("mcp_server.py")

#: Codex features that would hand the agent a shell, a browser or other agents.
_CODEX_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "apps", "plugins", "multi_agent",
    "browser_use", "computer_use", "in_app_browser", "image_generation", "hooks",
)

_QUOTA = re.compile(
    r"usage limit|rate.?limit|quota|\b429\b|too many requests|limit reached|credit balance|out of credits",
    re.IGNORECASE,
)
_AUTH = re.compile(
    r"not logged in|please log ?in|\blogin\b|unauthori[sz]ed|\b401\b|authenticat|invalid api key",
    re.IGNORECASE,
)


_OUTDATED = re.compile(
    r"or newer is required|does not support this model|claude update|please update|upgrade codex",
    re.IGNORECASE,
)


def classify_error(text: str) -> str | None:
    """`quota` / `auth` when the failure is about the account, not the search;
    `update` when the installed CLI is too old for the configured model."""
    if _OUTDATED.search(text or ""):
        return "update"
    if _QUOTA.search(text or ""):
        return "quota"
    if _AUTH.search(text or ""):
        return "auth"
    return None


def bridge_env(api_url: str, token: str, agent: str) -> dict[str, str]:
    return {"AIC_AGENT_API": api_url, "AIC_AGENT_TOKEN": token, "AIC_AGENT_NAME": agent}


def agent_process_env() -> dict[str, str]:
    """The CLI's environment: ours, minus every credential this backend holds.

    The backend loads Elastic/Milvus/DRES/DeepSeek keys into `os.environ` from
    `.env`, and a child process inherits all of it by default. The agents reach
    the corpus through the tool bridge only, so none of those keys go with them;
    what stays is what the CLIs need themselves (HOME, PATH, their own auth).
    """
    from ..config import _DOTENV_APPLIED
    from ..config_schema import KEYS_BY_NAME

    blocked = set(_DOTENV_APPLIED) | set(KEYS_BY_NAME)
    return {
        key: value
        for key, value in os.environ.items()
        if key not in blocked and not key.startswith("AIC26_")
    }


def _toml(value: str) -> str:
    # A JSON string literal is a valid TOML basic string for the ASCII paths,
    # URLs and tokens passed here.
    return json.dumps(value)


def codex_argv(settings: "Settings", *, binary: str, workdir: Path, prompt: str, bridge: dict[str, str]) -> list[str]:
    env_table = "{" + ", ".join(f"{key}={_toml(value)}" for key, value in bridge.items()) + "}"
    server = f"mcp_servers.{MCP_SERVER_NAME}"
    argv = [
        binary, "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        # The operator's own config.toml carries other MCP servers, plugins and
        # an xhigh effort; none of it belongs in a timed sidecar. Auth still
        # comes from CODEX_HOME.
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox", "read-only",
        "-C", str(workdir),
        "--model", settings.agent_codex_model,
        "-c", f"model_reasoning_effort={_toml(settings.agent_codex_reasoning_effort)}",
        "-c", 'approval_policy="never"',
        "-c", 'web_search="disabled"',
    ]
    for feature in _CODEX_DISABLED_FEATURES:
        argv += ["--disable", feature]
    argv += [
        "-c", f"{server}.command={_toml(sys.executable)}",
        "-c", f"{server}.args=[{_toml(str(MCP_SCRIPT))}]",
        "-c", f"{server}.env={env_table}",
        # Without this, `approval_policy=never` REJECTS every MCP call.
        "-c", f'{server}.default_tools_approval_mode="approve"',
        "-c", f"{server}.tool_timeout_sec=180",
        "-c", f"{server}.startup_timeout_sec=30",
        prompt,
    ]
    return argv


def claude_argv(settings: "Settings", *, binary: str, prompt: str, bridge: dict[str, str]) -> list[str]:
    mcp_config = {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(MCP_SCRIPT)],
                "env": bridge,
            }
        }
    }
    return [
        # The prompt must follow `-p` directly: several options below are
        # variadic and would swallow a trailing positional prompt.
        binary, "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--mcp-config", json.dumps(mcp_config),
        "--strict-mcp-config",
        "--tools", "",
        "--allowedTools", f"mcp__{MCP_SERVER_NAME}",
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--system-prompt", SYSTEM_PROMPT,
        "--model", settings.agent_claude_model,
        "--effort", settings.agent_claude_effort,
    ]


# ---- event streams --------------------------------------------------------
def _clip(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_call(tool: str, arguments: Any) -> str:
    """One readable line per tool call for the console's progress list."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {}
    args = arguments if isinstance(arguments, dict) else {}
    tool = tool.removeprefix(f"mcp__{MCP_SERVER_NAME}__")
    if tool == "search":
        folders = args.get("folders") or []
        return (
            f"search [{args.get('mode') or 'hybrid'}] “{_clip(args.get('query'), 90)}”"
            + (f" in {','.join(map(str, folders[:6]))}" if folders else "")
        )
    if tool in {"list_videos", "folder_frames"}:
        extra = []
        if args.get("position") is not None:
            extra.append(f"at {args['position']}")
        if args.get("offset"):
            extra.append(f"from #{int(args['offset']) + 1}")
        return f"{tool} {args.get('folder')}" + (f" ({', '.join(extra)})" if extra else "")
    if tool == "video_outline":
        window = ""
        if args.get("start") is not None or args.get("end") is not None:
            window = f" {args.get('start', 0)}–{args.get('end', 'end')}s"
        return f"video_outline {args.get('video_id')}{window}"
    if tool == "video_frames":
        window = ""
        if args.get("start") is not None or args.get("end") is not None:
            window = f" {args.get('start', 0)}–{args.get('end', 'end')}s"
        return f"video_frames {args.get('video_id')}{window}"
    if tool == "view_frames":
        ids = args.get("keyframe_ids") or []
        return f"view_frames {len(ids)}: {', '.join(map(str, ids[:3]))}{' …' if len(ids) > 3 else ''}"
    if tool == "video_text":
        return f"video_text {args.get('video_id')} {args.get('start')}–{args.get('end')}s"
    if tool == "report_candidate":
        where = args.get("keyframe_id") or f"{args.get('video_id')} @ {args.get('time')}s"
        return f"report_candidate {where} ({args.get('confidence')})"
    return f"{tool} {_clip(json.dumps(args, ensure_ascii=False), 120)}"


def read_codex_event(event: dict[str, Any], state: "AgentState") -> None:
    kind = event.get("type")
    item = event.get("item") or {}
    item_type = item.get("type")
    if kind == "item.started" and item_type == "mcp_tool_call":
        state.tool_calls += 1
        state.step("tool", describe_call(str(item.get("tool") or ""), item.get("arguments")))
    elif kind == "item.completed":
        if item_type == "mcp_tool_call" and item.get("error"):
            error = item["error"]
            message = error.get("message") if isinstance(error, dict) else error
            state.step("error", f"{item.get('tool')} failed: {_clip(message)}")
        elif item_type == "agent_message" and item.get("text"):
            state.summary = _clip(item["text"], 1200)
            state.step("message", item["text"])
        elif item_type == "reasoning" and item.get("text"):
            state.step("thought", item["text"])
        elif item_type == "error" and item.get("message"):
            state.step("error", item["message"])
    elif kind == "turn.failed":
        error = event.get("error") or {}
        state.fail_reason = _clip(error.get("message") if isinstance(error, dict) else error, 600)
    elif kind == "error" and event.get("message"):
        state.fail_reason = _clip(event["message"], 600)


def read_claude_event(event: dict[str, Any], state: "AgentState") -> None:
    kind = event.get("type")
    if kind == "system" and event.get("subtype") == "init":
        servers = {s.get("name"): s.get("status") for s in event.get("mcp_servers") or []}
        status = servers.get(MCP_SERVER_NAME)
        if status not in (None, "connected"):
            state.step("error", f"tool bridge {status}")
    elif kind == "assistant":
        for block in (event.get("message") or {}).get("content") or []:
            if block.get("type") == "tool_use":
                state.tool_calls += 1
                state.step("tool", describe_call(str(block.get("name") or ""), block.get("input")))
            elif block.get("type") == "text" and block.get("text", "").strip():
                state.step("message", block["text"])
            elif block.get("type") == "thinking" and block.get("thinking", "").strip():
                state.step("thought", block["thinking"])
    elif kind == "result":
        if event.get("total_cost_usd") is not None:
            state.cost_usd = float(event["total_cost_usd"])
        if event.get("is_error") or event.get("subtype") not in (None, "success"):
            state.fail_reason = _clip(event.get("result") or event.get("subtype") or "error", 600)
        elif event.get("result"):
            state.summary = _clip(event["result"], 1200)

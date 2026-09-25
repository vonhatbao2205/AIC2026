"""Stdio MCP server that Codex CLI / Claude Code CLI spawn for one agent run.

It is a pure bridge: every tool is answered by the backend's
`/api/agent/tools/{name}` endpoint, authorised by the run's token. The bridge
therefore holds no credentials and no retrieval code, and it has to stay
importable by a bare interpreter — stdlib only, no `app.*` imports — because the
CLI launches it as a plain script.

Environment (set by `app.agent.runs`):
  AIC_AGENT_API    backend base URL, e.g. http://127.0.0.1:8000
  AIC_AGENT_TOKEN  per-run secret; the backend rejects calls without it
  AIC_AGENT_NAME   "codex" | "claude", so candidates are attributed correctly
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request

PROTOCOL_VERSION = "2025-06-18"
API = (os.environ.get("AIC_AGENT_API") or "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.environ.get("AIC_AGENT_TOKEN") or ""
AGENT = os.environ.get("AIC_AGENT_NAME") or "agent"
#: A contact sheet downloads up to 48 keyframes, and a hybrid search runs every
#: retrieval channel; both finish well inside this.
HTTP_TIMEOUT = float(os.environ.get("AIC_AGENT_HTTP_TIMEOUT") or 150)

_write_lock = threading.Lock()


def _send(message: dict) -> None:
    line = json.dumps(message, ensure_ascii=False)
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _backend(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-AIC-Agent-Token": TOKEN,
            "X-AIC-Agent-Name": AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _error_result(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _call_tool(params: dict) -> dict:
    name = str(params.get("name") or "")
    arguments = params.get("arguments") or {}
    try:
        return _backend("POST", f"/api/agent/tools/{name}", {"arguments": arguments})
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail")
        except Exception:  # noqa: BLE001 - the status alone is still useful
            detail = None
        return _error_result(f"Tool {name} failed ({exc.code}): {detail or exc.reason}")
    except Exception as exc:  # noqa: BLE001 - a dead backend is a tool error, not a crash
        return _error_result(f"Tool {name} failed: {exc}")


def _handle(message: dict) -> None:
    method = message.get("method")
    msg_id = message.get("id")
    if msg_id is None:
        return  # notification (initialized, cancelled, ...): nothing to answer
    try:
        if method == "initialize":
            requested = (message.get("params") or {}).get("protocolVersion")
            result = {
                "protocolVersion": requested or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "aic26-video-search", "version": "1.0.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": _backend("GET", "/api/agent/tools").get("tools", [])}
        elif method == "tools/call":
            result = _call_tool(message.get("params") or {})
        else:
            _send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Unknown method {method}"}})
            return
    except Exception as exc:  # noqa: BLE001
        _send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": str(exc)[:500]}})
        return
    _send({"jsonrpc": "2.0", "id": msg_id, "result": result})


def main() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            message = json.loads(raw)
        except ValueError:
            continue
        # A thread per request: an agent may issue tool calls in parallel, and a
        # contact sheet must not hold up a quick text search behind it.
        threading.Thread(target=_handle, args=(message,), daemon=True).start()


if __name__ == "__main__":
    main()

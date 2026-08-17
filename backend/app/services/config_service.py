"""Read and rewrite the operator's `.env` from inside the running app.

The packaged app ships with no credentials at all: the person who downloads it
imports the `.env` the team already shares out-of-band. That import has to be
survivable — a wrong file must not leave the instance unable to explain itself —
so writes are atomic and the reply describes exactly what landed on disk.

Secrets go one way only. Values arrive here from an upload and are written to a
0600 file; what goes back to the browser is masked by `config_schema.mask`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .. import paths
from ..config import Settings, parse_dotenv
from ..config_schema import GROUPS, KEYS_BY_NAME, REQUIRED_KEYS, mask

#: Variables that decide where this process reads config, data and the UI from.
#: Honouring them from an uploaded file would let one bad import point the app at
#: a directory it cannot write and lock the operator out of the settings screen.
BLOCKED_KEYS = frozenset(
    {"AIC26_CONFIG_DIR", "AIC26_DATA_DIR", "AIC26_STATIC_DIR", "HF_HOME", "PATH", "PYTHONPATH"}
)

_VALID_KEY = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _is_env_key(key: str) -> bool:
    return bool(key) and not key[0].isdigit() and set(key) <= _VALID_KEY


def file_values() -> dict[str, str]:
    """Whatever the config file currently holds (empty if it does not exist)."""
    path = paths.env_file()
    if not path.exists():
        return {}
    try:
        return parse_dotenv(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def status(settings: Settings) -> dict[str, Any]:
    """Per-key state for the settings screen. Secret values are masked."""
    on_disk = file_values()
    groups: list[dict[str, Any]] = []
    for group in GROUPS:
        keys: list[dict[str, Any]] = []
        for spec in group.keys:
            value = (os.environ.get(spec.key) or "").strip()
            keys.append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "secret": spec.secret,
                    "required": spec.required,
                    "set": bool(value),
                    "preview": mask(spec.key, value),
                    # True when the value comes from the process environment
                    # rather than the file — importing a new file will not
                    # change it, because os.environ wins.
                    "from_process_env": bool(value) and spec.key not in on_disk,
                }
            )
        groups.append({"name": group.name, "summary": group.summary, "keys": keys})

    missing = [k for k in REQUIRED_KEYS if not (os.environ.get(k) or "").strip()]
    return {
        "configured": settings.mock_mode or not missing,
        "mock_mode": settings.mock_mode,
        "missing_required": missing,
        "env_path": str(paths.env_file()),
        "env_exists": paths.env_file().exists(),
        "groups": groups,
    }


def template_text() -> str:
    """The documented `.env.example`, or a generated one if it was not shipped."""
    template = paths.env_template_file()
    if template.exists():
        try:
            return template.read_text(encoding="utf-8")
        except OSError:
            pass
    lines = ["# AIC26 retrieval configuration. Fill in and import via Settings.", ""]
    for group in GROUPS:
        lines.append(f"# --- {group.name} — {group.summary}")
        lines.extend(f"{spec.key}=" for spec in group.keys)
        lines.append("")
    return "\n".join(lines)


def _render(values: dict[str, str]) -> str:
    """Serialise back to `.env`, grouped and commented like the shipped template."""
    remaining = dict(values)
    lines = [
        "# AIC26 retrieval configuration.",
        "# Written by the in-app Settings screen — hand edits are preserved on the",
        "# next import, but comments outside this header are not.",
        "",
    ]
    for group in GROUPS:
        present = [spec.key for spec in group.keys if spec.key in remaining]
        if not present:
            continue
        lines.append(f"# --- {group.name} ---")
        lines.extend(f"{key}={remaining.pop(key)}" for key in present)
        lines.append("")
    if remaining:
        lines.append("# --- Other (not part of the documented schema) ---")
        lines.extend(f"{key}={value}" for key, value in sorted(remaining.items()))
        lines.append("")
    return "\n".join(lines)


def write_values(values: dict[str, str]) -> Path:
    """Atomically replace the config file, readable only by its owner."""
    path = paths.env_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(_render(values), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        # Some bind-mounted filesystems reject chmod; the write itself still stands.
        pass
    os.replace(tmp, path)
    return path


def apply_env_text(text: str, *, replace: bool = False) -> dict[str, Any]:
    """Merge (or replace) the config file with an uploaded `.env`.

    Returns a summary naming every key that was applied, ignored or rejected, so
    the operator can tell a truncated paste from a complete config at a glance.
    """
    parsed = parse_dotenv(text)
    applied: dict[str, str] = {}
    rejected: list[str] = []
    unknown: list[str] = []
    blank: list[str] = []
    for key, value in parsed.items():
        if key in BLOCKED_KEYS or not _is_env_key(key):
            rejected.append(key)
            continue
        if not value.strip():
            # A half-filled template is the common upload. Treat its blanks as
            # "leave this alone" — silently wiping a working DRES password
            # because the new file did not repeat it is not a recoverable
            # surprise. Clearing a key is what `replace` is for.
            blank.append(key)
            continue
        if key not in KEYS_BY_NAME:
            unknown.append(key)
        applied[key] = value

    if not applied:
        raise ValueError(
            "No usable KEY=VALUE lines found — is this the right .env file? "
            f"(rejected: {', '.join(sorted(rejected)) or 'none'}; "
            f"blank: {len(blank)})"
        )

    merged = applied if replace else {**file_values(), **applied}
    path = write_values({k: v for k, v in merged.items() if v.strip()})
    return {
        "env_path": str(path),
        "applied": sorted(applied),
        "unknown": sorted(unknown),
        "rejected": sorted(rejected),
        "ignored_blank": sorted(blank),
        "replaced": replace,
    }

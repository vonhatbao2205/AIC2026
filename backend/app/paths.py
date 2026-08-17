"""Filesystem locations that differ between a source checkout and a packaged app.

In a checkout everything lives next to the code: `backend/.env` holds the config
and `backend/data/` the submit history. In the Docker image the code directory is
disposable — it is replaced wholesale on every `docker pull` — so the operator's
config and history must live on mounted volumes instead. Both are therefore
overridable by environment variable, with the checkout layout as the default.
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


def _dir_from_env(name: str, default: Path) -> Path:
    raw = (os.environ.get(name) or "").strip()
    return Path(raw).expanduser() if raw else default


def config_dir() -> Path:
    """Where `.env` is read from and written to (AIC26_CONFIG_DIR)."""
    return _dir_from_env("AIC26_CONFIG_DIR", BACKEND_ROOT)


def data_dir() -> Path:
    """Where mutable state such as the submit history lives (AIC26_DATA_DIR)."""
    return _dir_from_env("AIC26_DATA_DIR", BACKEND_ROOT / "data")


def env_file() -> Path:
    return config_dir() / ".env"


def secret_dirs() -> tuple[Path, ...]:
    """Directories searched for loose `*.txt` secret files (a dev convenience).

    Overridable by AIC26_SECRET_DIR for the same reason the config dir is: a test
    run must be able to point every credential source somewhere harmless, not
    just the obvious one.
    """
    override = (os.environ.get("AIC26_SECRET_DIR") or "").strip()
    if override:
        return (Path(override).expanduser(),)
    return (REPO_ROOT / "API_KEY", REPO_ROOT)


def env_template_file() -> Path:
    """The documented `.env.example`, shipped alongside the code."""
    return BACKEND_ROOT / ".env.example"


def static_dir() -> Path | None:
    """The built frontend this deployment serves, or None when there isn't one.

    The packaged image copies `frontend/dist` to `backend/static`; a checkout
    running `npm run dev` has no bundled UI and serves the API alone.
    """
    override = (os.environ.get("AIC26_STATIC_DIR") or "").strip()
    candidates = (
        [Path(override).expanduser()]
        if override
        else [BACKEND_ROOT / "static", REPO_ROOT / "frontend" / "dist"]
    )
    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate
    return None

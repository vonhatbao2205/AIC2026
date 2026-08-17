"""The test suite must be unable to reach the real world. Regression guard.

On 2026-08-06 a settings-reload test dropped AIC26_MOCK_MODE, the dotenv loader
re-read the developer's real `backend/.env`, and the submit tests posted 34 wrong
answers to the official DRES evaluation server under the team's account. Mock
mode was doing all the work and it only takes one variable to switch off.

These tests assert the two independent barriers put in afterwards: the suite
cannot see any credential, and the submit path refuses to run under pytest even
if it somehow could.
"""
from __future__ import annotations

import asyncio

import pytest

from app import paths
from app.adapters.dres_client import DresClient, DresError
from app.config import get_settings


def test_the_suite_never_reads_the_developers_credentials():
    settings = get_settings()

    # Neither credential source may resolve to the checkout.
    assert paths.env_file() != paths.BACKEND_ROOT / ".env"
    assert paths.REPO_ROOT / "API_KEY" not in paths.secret_dirs()

    assert not settings.dres_username
    assert not settings.dres_password
    assert not settings.dres_session
    assert not settings.elastic_api_key
    assert not settings.milvus_token
    assert not settings.nvidia_api_key
    assert not settings.deepseek_api_key
    assert settings.has_dres is False


def test_submitting_from_a_test_is_refused_even_when_dres_is_configured(settings):
    """The last line of defence: a live-looking client still must not submit."""
    settings.mock_mode = False
    settings.dres_base_url = "http://dres.invalid"
    settings.dres_session = "pretend-this-is-live"

    with pytest.raises(DresError, match="Refusing to submit from a test run"):
        asyncio.run(DresClient(settings).submit("eval-id", [{"answers": []}]))


def test_the_guard_can_be_lifted_deliberately(settings, allow_test_submit):
    """Opting in reaches the network layer — which is where mock transports sit."""
    settings.mock_mode = False
    settings.dres_base_url = "http://127.0.0.1:9"  # discard port
    settings.dres_session = "s"

    with pytest.raises(DresError) as excinfo:
        asyncio.run(DresClient(settings).submit("eval-id", [{"answers": []}]))
    assert "Refusing to submit" not in str(excinfo.value)

import os
import tempfile
from pathlib import Path

import pytest

# Force mock mode for the whole backend test suite so no live service is hit.
os.environ["AIC26_MOCK_MODE"] = "true"
os.environ["MEDIA_BASE_URL"] = "https://media.test"

# Mock mode alone is not enough. It is one environment variable, and a test that
# legitimately turns it off inherits whatever `backend/.env` holds — which on a
# developer machine is live Elastic/Milvus keys and the team's DRES password.
# That happened: a settings-reload test dropped mock mode, the loader re-read the
# real .env, and the submit tests posted 34 wrong answers to the official
# evaluation server. So point config at a throwaway directory instead: the tests
# never see the developer's credentials at all, whatever else goes wrong.
_SANDBOX = Path(tempfile.mkdtemp(prefix="aic26-tests-"))
os.environ["AIC26_CONFIG_DIR"] = str(_SANDBOX / "config")
os.environ["AIC26_DATA_DIR"] = str(_SANDBOX / "data")
# `.env` is not the only credential source: config also falls back to loose
# API_KEY/*.txt files at the repo root.
os.environ["AIC26_SECRET_DIR"] = str(_SANDBOX / "secrets")

# Belt and braces for a shell that exports DRES settings directly.
for _leaked in ("DRES_USERNAME", "DRES_PASSWORD", "DRES_SESSION"):
    os.environ.pop(_leaked, None)
os.environ["DRES_BASE_URL"] = "http://127.0.0.1:9"  # discard port; nothing listens

from app.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def allow_test_submit(monkeypatch):
    """Lift the DresClient submit guard for a test that drives it through a mock
    transport. Naming it in the signature is the point: a test that posts a
    submission has to say so, and the lift ends when the test does."""
    monkeypatch.setenv("AIC26_ALLOW_TEST_SUBMIT", "1")

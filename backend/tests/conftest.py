import os

import pytest

# Force mock mode for the whole backend test suite so no live service is hit.
os.environ["AIC26_MOCK_MODE"] = "true"
os.environ["MEDIA_BASE_URL"] = "https://media.test"

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

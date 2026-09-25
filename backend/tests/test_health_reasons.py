"""Every health failure must name a cause.

httpx raises its transport errors with no message, so `str(exc)` is `""`. An
adapter formatting that straight into its health payload ships a banner reading
"elastic unreachable: " — the operator is told something broke but not what,
which under a clock is the same as being told nothing. This already bit the
reranker once (see test_reranker.py); these tests hold the line for the rest.
"""
import httpx
import pytest

from app.adapters.dres_client import DresClient
from app.adapters.elastic_client import ElasticClient
from app.adapters.http_pool import failure_reason
from app.adapters.milvus_client import MilvusClient
from app.adapters.nvila_client import NvilaQaClient
from app.adapters.pe_encoder import GlapEncoderClient, PeEncoderClient
from app.adapters.qwen3_vl_encoder import Qwen3VlEncoderClient
from app.adapters.qwen_reranker import QwenRerankerClient

BLANK_TRANSPORT_ERRORS = [
    httpx.ConnectTimeout(""),
    httpx.ConnectError(""),
    httpx.ReadTimeout(""),
    httpx.PoolTimeout(""),
]


@pytest.mark.parametrize("exc", BLANK_TRANSPORT_ERRORS, ids=lambda e: type(e).__name__)
def test_blank_transport_errors_still_name_their_cause(exc):
    assert str(exc) == "", "fixture assumes httpx stringifies these to nothing"
    assert failure_reason(exc) == type(exc).__name__


def test_a_real_message_is_kept_verbatim():
    assert failure_reason(OSError(-2, "Name or service not known")) == (
        "[Errno -2] Name or service not known"
    )
    assert failure_reason(ValueError("  boom  ")) == "boom"


def _client(factory, settings):
    """A live (non-mock) client whose transport always fails blankly."""
    settings.mock_mode = False
    settings.elastic_endpoint = "https://elastic.test"
    settings.elastic_api_key = "key"
    settings.milvus_endpoint = "https://milvus.test"
    settings.milvus_token = "token"
    settings.pe_encoder_url = "https://pe.test"
    settings.glap_encoder_url = "https://glap.test"
    settings.nvila_base_url = "https://nvila.test"
    settings.nvila_token = "token"
    settings.qwen3_vl_encoder_url = "https://qwen.test"
    settings.qwen3_vl_encoder_token = "token"
    settings.qwen_reranker_url = "https://rerank.test"
    settings.qwen_reranker_token = "token"
    settings.qwen_reranker_enabled = True
    # Qwen embeddings exist only for InfoShot++, so its client reports "disabled"
    # on the BTC profile and would never reach the transport being tested here.
    settings.retrieval_database = "infoshotpp"
    # Fake DRES credentials on this settings object only, never the environment:
    # conftest strips the real ones on purpose, and the transport is stubbed
    # below, so nothing can reach the evaluation server from here.
    settings.dres_username = "test-user"
    settings.dres_password = "test-password"
    settings.dres_enabled = True
    client = factory(settings)
    client.mock = False
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [
        ElasticClient,
        NvilaQaClient,
        PeEncoderClient,
        GlapEncoderClient,
        Qwen3VlEncoderClient,
        QwenRerankerClient,
        DresClient,
    ],
    ids=lambda f: f.__name__,
)
async def test_no_adapter_reports_an_unexplained_failure(factory, settings, monkeypatch):
    client = _client(factory, settings)

    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("")

    class DeadClient:
        """Fails at the request, not at client construction: adapters wrap
        transport errors around the call, so raising earlier would test the
        stub rather than the adapter."""

        async def get(self, *args, **kwargs):
            raise httpx.ConnectTimeout("")

        async def post(self, *args, **kwargs):
            raise httpx.ConnectTimeout("")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    for pool in ("_http", "_pool"):
        holder = getattr(client, pool, None)
        if holder is not None:
            monkeypatch.setattr(holder, "get", lambda: DeadClient())
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: DeadClient())
    monkeypatch.setattr(client, "_connect", boom, raising=False)

    health = await client.health()

    if health.get("ok"):
        pytest.skip(f"{factory.__name__} does not reach the network in this config")
    if health.get("mode") == "disabled":
        # "Not configured" is not a failure; main.py keeps it out of the banner.
        pytest.skip(f"{factory.__name__} is disabled in this config")
    reason = str(health.get("error") or health.get("reason") or "")
    assert reason.strip(), f"{factory.__name__} reported a failure with no cause"


@pytest.mark.asyncio
async def test_milvus_image_health_names_its_cause(settings, monkeypatch):
    settings.mock_mode = False
    settings.milvus_endpoint_2 = "https://milvus.test"
    settings.milvus_token_2 = "token"
    client = MilvusClient(settings.for_retrieval_database("infoshotpp"))
    client.mock = False

    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("")

    monkeypatch.setattr(client, "_connect", boom)

    for health in (await client.health(), await client.health_qwen_image()):
        assert health["ok"] is False
        assert str(health.get("error") or health.get("reason") or "").strip()


def test_the_banner_never_ships_a_line_that_explains_nothing():
    """The exact string /api/health builds. `elastic unreachable: ` is the shape
    this whole module exists to prevent."""
    services = {
        "elastic": {"ok": False, "error": failure_reason(httpx.ConnectTimeout(""))},
        "qwen3_vl_encoder": {"ok": False, "mode": "unreachable", "error": "ConnectTimeout"},
        "nvila_qa": {"ok": False, "mode": "unreachable", "error": failure_reason(httpx.ConnectError(""))},
        "dres": {"ok": False, "mode": "disabled", "error": None},
        "milvus": {"ok": True},
    }
    warnings = [
        f"{name} unreachable: {s.get('error')}"
        for name, s in services.items()
        if not s.get("ok") and s.get("mode") != "disabled"
    ]

    assert warnings == [
        "elastic unreachable: ConnectTimeout",
        "qwen3_vl_encoder unreachable: ConnectTimeout",
        "nvila_qa unreachable: ConnectError",
    ]
    for line in warnings:
        assert not line.rstrip().endswith(":"), line
        assert "None" not in line, line


# ---- what the banner is allowed to shout about -------------------------------
#
# These drive the real endpoint. Rebuilding the comprehension inside the test
# would pass just as happily with the production filter deleted.


def _health(monkeypatch, **down):
    """Call /api/health with the named services forced down."""
    from fastapi.testclient import TestClient

    from app import main

    async def dead():
        return {"ok": False, "mode": "unreachable", "error": "ConnectTimeout"}

    service = main.search_services["btc"]
    holders = {
        "elastic": service.elastic,
        "milvus": service.milvus,
        "pe_encoder": service.pe,
        "qa_vision": main.qa_vision,
        "qwen_reranker": service.reranker,
        "web_grounding": main.web_grounding_client,
    }
    for name, is_down in down.items():
        if is_down:
            monkeypatch.setattr(holders[name], "health", dead)
    return TestClient(main.app).get("/api/health").json()


def test_optional_workers_do_not_raise_the_retrieval_banner(monkeypatch):
    """A stopped Colab session for the QA copilot or the reranker leaves search
    working. Calling that "live retrieval degraded" is a false alarm, and a
    banner that cries wolf is one the operator stops reading."""
    body = _health(monkeypatch, qa_vision=True, qwen_reranker=True, web_grounding=True)

    assert body["warnings"] == []
    # Still visible to anyone who looks at the service list or the capabilities;
    # it is only the alarming banner that stays quiet.
    assert body["services"]["qa_vision"]["ok"] is False
    assert body["services"]["qwen_reranker"]["ok"] is False


def test_a_dead_core_channel_still_raises_it(monkeypatch):
    body = _health(monkeypatch, elastic=True, qa_vision=True)

    # `ok` is not asserted: the suite runs in mock mode, where it is always True.
    assert body["warnings"] == ["elastic unreachable: ConnectTimeout"]

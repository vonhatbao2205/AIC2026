"""The in-app configuration screen: status, template and `.env` import.

Every test here redirects AIC26_CONFIG_DIR at a tmp_path first. The packaged app
writes to a mounted volume, but in a checkout the same code path targets the
developer's real `backend/.env` — these tests must never be the thing that
overwrites it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app import paths
from app.services import config_service

client = TestClient(main_module.app)


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC26_CONFIG_DIR", str(tmp_path))
    assert paths.env_file() == tmp_path / ".env"
    yield tmp_path


def _upload(text: str, *, replace: bool = False):
    return client.post(
        "/api/config/import",
        files={"file": (".env", text, "text/plain")},
        data={"replace": str(replace).lower()},
    )


def test_status_lists_groups_and_flags_missing_required(config_dir, monkeypatch):
    for key in ("ELASTIC_ENDPOINT", "ELASTIC_API_KEY", "MILVUS_ENDPOINT_1"):
        monkeypatch.delenv(key, raising=False)
    body = client.get("/api/config").json()

    assert body["env_path"] == str(config_dir / ".env")
    assert body["env_exists"] is False
    assert {g["name"] for g in body["groups"]} >= {"Elastic Cloud", "Milvus / Zilliz", "DRES"}
    assert "ELASTIC_API_KEY" in body["missing_required"]
    # Mock mode is a complete configuration on its own — nothing to import.
    assert body["configured"] is True


def test_secret_values_are_masked_and_plain_ones_are_not(config_dir, monkeypatch):
    monkeypatch.setenv("ELASTIC_API_KEY", "abcdefghijklmnop")
    monkeypatch.setenv("ELASTIC_ENDPOINT", "https://cluster.es.io")
    keys = {
        k["key"]: k
        for group in client.get("/api/config").json()["groups"]
        for k in group["keys"]
    }

    assert keys["ELASTIC_API_KEY"]["preview"] == "••••••••mnop"
    assert "abcdefghij" not in keys["ELASTIC_API_KEY"]["preview"]
    assert keys["ELASTIC_ENDPOINT"]["preview"] == "https://cluster.es.io"


def test_short_secrets_do_not_leak_a_tail(config_dir, monkeypatch):
    monkeypatch.setenv("MILVUS_TOKEN_1", "short")
    keys = {
        k["key"]: k
        for group in client.get("/api/config").json()["groups"]
        for k in group["keys"]
    }
    assert keys["MILVUS_TOKEN_1"]["preview"] == "••••••••"


def test_import_writes_the_file_and_rebuilds_the_running_services(config_dir, monkeypatch):
    # conftest pins mock mode through the real environment, which by design wins
    # over the file; drop it so the imported value is the one under test.
    monkeypatch.delenv("AIC26_MOCK_MODE", raising=False)
    monkeypatch.delenv("MEDIA_BASE_URL", raising=False)
    resp = _upload(
        "# team config\n"
        "ELASTIC_ENDPOINT=https://cluster.es.io\n"
        "ELASTIC_API_KEY=secret-key\n"
        "MEDIA_BASE_URL=https://media.r2.dev\n"
        "AIC26_MOCK_MODE=false\n"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] == [
        "AIC26_MOCK_MODE",
        "ELASTIC_API_KEY",
        "ELASTIC_ENDPOINT",
        "MEDIA_BASE_URL",
    ]
    assert (config_dir / ".env").read_text(encoding="utf-8").count("ELASTIC_API_KEY=secret-key") == 1

    # The imported values are live in the process, not just on disk.
    assert main_module.settings.elastic_endpoint == "https://cluster.es.io"
    assert main_module.settings.mock_mode is False
    assert main_module.media.base_url == "https://media.r2.dev"
    assert main_module.search_service.s is main_module.profile_settings["btc"]


def test_import_merges_by_default_and_replaces_on_request(config_dir):
    _upload("ELASTIC_ENDPOINT=https://one.es.io\nDRES_USERNAME=team01\n")
    _upload("ELASTIC_ENDPOINT=https://two.es.io\n")
    assert config_service.file_values() == {
        "ELASTIC_ENDPOINT": "https://two.es.io",
        "DRES_USERNAME": "team01",
    }

    _upload("ELASTIC_ENDPOINT=https://three.es.io\n", replace=True)
    assert config_service.file_values() == {"ELASTIC_ENDPOINT": "https://three.es.io"}


def test_blank_values_in_a_half_filled_template_keep_the_existing_secret(config_dir):
    _upload("DRES_USERNAME=team01\nDRES_PASSWORD=pw-that-works\n")
    _upload("DRES_USERNAME=team01\nDRES_PASSWORD=\nELASTIC_ENDPOINT=https://one.es.io\n")

    values = config_service.file_values()
    assert values["DRES_PASSWORD"] == "pw-that-works"
    assert values["ELASTIC_ENDPOINT"] == "https://one.es.io"


def test_import_rejects_keys_that_would_relocate_the_config(config_dir):
    body = _upload("ELASTIC_ENDPOINT=https://one.es.io\nAIC26_CONFIG_DIR=/tmp/elsewhere\n").json()

    assert body["rejected"] == ["AIC26_CONFIG_DIR"]
    assert "AIC26_CONFIG_DIR" not in config_service.file_values()
    assert paths.env_file() == config_dir / ".env"


def test_import_keeps_unknown_keys_but_names_them(config_dir):
    body = _upload("ELASTIC_ENDPOINT=https://one.es.io\nTEAM_NOTE=hello\n").json()

    assert body["unknown"] == ["TEAM_NOTE"]
    assert config_service.file_values()["TEAM_NOTE"] == "hello"


def test_a_file_with_nothing_usable_is_refused_before_anything_is_written(config_dir):
    _upload("ELASTIC_ENDPOINT=https://one.es.io\n")
    resp = _upload("just some prose, no assignments\n", replace=True)

    assert resp.status_code == 400
    assert "KEY=VALUE" in resp.json()["detail"]
    assert config_service.file_values() == {"ELASTIC_ENDPOINT": "https://one.es.io"}


def test_binary_upload_is_refused(config_dir):
    resp = client.post(
        "/api/config/import", files={"file": (".env", b"\xff\xfe\x00binary", "text/plain")}
    )
    assert resp.status_code == 400
    assert not (config_dir / ".env").exists()


def test_reload_picks_up_a_file_edited_outside_the_app(config_dir, monkeypatch):
    monkeypatch.delenv("MEDIA_BASE_URL", raising=False)
    (config_dir / ".env").write_text("MEDIA_BASE_URL=https://edited.r2.dev\n", encoding="utf-8")

    assert client.post("/api/config/reload").status_code == 200
    assert main_module.settings.media_base_url == "https://edited.r2.dev"


def test_a_reimport_overrides_the_previous_file_value(config_dir, monkeypatch):
    """os.environ wins over the file, so the loader must drop what it set before."""
    monkeypatch.delenv("MEDIA_BASE_URL", raising=False)
    _upload("MEDIA_BASE_URL=https://first.r2.dev\n")
    assert main_module.settings.media_base_url == "https://first.r2.dev"

    _upload("MEDIA_BASE_URL=https://second.r2.dev\n")
    assert main_module.settings.media_base_url == "https://second.r2.dev"


def test_template_is_downloadable_and_documents_the_required_keys(config_dir):
    resp = client.get("/api/config/template")
    assert resp.status_code == 200
    assert ".env" in resp.headers["content-disposition"]
    for key in ("ELASTIC_ENDPOINT", "MILVUS_TOKEN_1", "MEDIA_BASE_URL", "DRES_USERNAME"):
        assert key in resp.text


@pytest.fixture(autouse=True)
def _restore_runtime():
    """Leave the module globals as the rest of the suite expects them."""
    yield
    main_module.build_runtime()

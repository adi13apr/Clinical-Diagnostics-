"""X-Served-By: which API replica answered (docker-compose.yml sets SERVED_BY_HEADER=1). Off by default."""
import socket

from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from test_api import SECRET, SUPABASE_URL, model_service, store  # noqa: F401  (fixtures)


def app(model_service, store, **settings):
    settings = Settings(supabase_url=SUPABASE_URL, supabase_key="anon", jwt_secret=SECRET, **settings)
    return TestClient(create_app(settings, model_service=model_service, db_factory=store.factory,
                                 llm_factory=lambda: None))


def test_header_names_the_replica_when_enabled(model_service, store):
    client = app(model_service, store, served_by="api-1")
    assert client.get("/health").headers["X-Served-By"] == "api-1"
    assert client.get("/history").headers["X-Served-By"] == "api-1"  # errors too (401 here)


def test_no_header_by_default(model_service, store):
    assert "X-Served-By" not in app(model_service, store).get("/health").headers


def test_env_switch_uses_the_hostname(tmp_path):
    missing = tmp_path / "none.toml"
    assert Settings.from_env({"SERVED_BY_HEADER": "1"}, secrets_path=missing).served_by == socket.gethostname()
    assert Settings.from_env({}, secrets_path=missing).served_by is None

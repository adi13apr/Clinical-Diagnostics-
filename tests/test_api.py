"""FastAPI backend (api/). Supabase, the LLM and the network are faked; the model is a stub."""
import asyncio
import io
import logging
import socket
import threading
import time
import urllib.error
import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import jwt
import numpy as np
import pytest
import torch
import torch.nn as nn
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from PIL import Image

from agent.agent import REPORT_CACHE, MockLLMClient, RedisReportCache, ReportCache
from api.auth import AuthError, AuthUnavailable, SupabaseJWTVerifier
from api.main import check_redis, create_app, daily_volume, shared_state
from api.settings import Settings
from api.state import RedisUploadStore, SerializedModel, UploadStore, connect_redis, redis_location
from services.calibration import REFER_MESSAGE
from services.model_handler import ModelService
from services.tracing import trace_span

SUPABASE_URL = "https://test.supabase.co"
ISSUER = f"{SUPABASE_URL}/auth/v1"
SECRET = "test-jwt-secret-that-is-long-enough-for-hs256"
ALICE, BOB = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"


# ---- Fakes ----

def make_token(sub=ALICE, secret=SECRET, alg="HS256", **overrides):
    now = int(time.time())
    claims = {"sub": sub, "email": f"{sub[:4]}@example.com", "aud": "authenticated", "iss": ISSUER,
              "role": "authenticated", "iat": now, "exp": now + 3600, **overrides}
    return jwt.encode(claims, secret, algorithm=alg)


def auth(sub=ALICE):
    return {"Authorization": f"Bearer {make_token(sub)}"}


class FakeQuery:
    """The slice of the PostgREST query builder the services use."""

    def __init__(self, db, table):
        self.db, self.table = db, table
        self.filters, self.insert_rows, self.order_by, self.window, self.counted = [], None, None, None, False

    def select(self, *columns, count=None):
        self.counted = count is not None
        return self

    def insert(self, data):
        self.insert_rows = data if isinstance(data, list) else [data]
        return self

    def eq(self, column, value):
        self.filters.append(lambda r: r.get(column) == value)
        return self

    def gte(self, column, value):
        self.filters.append(lambda r: r.get(column) >= value)
        return self

    def lt(self, column, value):
        self.filters.append(lambda r: r.get(column) < value)
        return self

    def order(self, column, desc=False):
        self.order_by = (column, desc)
        return self

    def range(self, start, end):
        self.window = (start, end + 1)
        return self

    def limit(self, n):
        self.window = (0, n)
        return self

    def execute(self):
        store = self.db.store
        if self.table in store.failing:
            raise RuntimeError(f"{self.table} unavailable")
        if self.insert_rows is not None:
            rows = [store.insert(self.table, self.db.user_id, row) for row in self.insert_rows]
            return SimpleNamespace(data=rows, count=None)
        if self.table == "prediction_daily_summary":
            rows = store.daily_summary(self.db.user_id)
        else:  # RLS: only the session user's rows
            rows = [r for r in store.tables.get(self.table, []) if r["user_id"] == self.db.user_id]
        rows = [r for r in rows if all(f(r) for f in self.filters)]
        if self.order_by:
            rows.sort(key=lambda r: r[self.order_by[0]], reverse=self.order_by[1])
        total = len(rows)
        if self.window:
            rows = rows[self.window[0]:self.window[1]]
        return SimpleNamespace(data=rows, count=total if self.counted else None)


class FakeStore:
    """Shared fake database; insert fills id, user_id and created_at like the server defaults."""

    def __init__(self):
        self.tables, self.failing, self.clients = {}, set(), []
        self._next_id = 0

    def insert(self, table, user_id, row):
        self._next_id += 1
        created = datetime(2026, 10, 20, 12, tzinfo=timezone.utc) + timedelta(seconds=self._next_id)
        row = {"id": self._next_id, "user_id": user_id, "created_at": created.isoformat(), **row}
        self.tables.setdefault(table, []).append(row)
        return row

    def daily_summary(self, user_id):
        groups = {}
        for r in self.tables.get("predictions", []):
            if r["user_id"] == user_id:
                groups.setdefault((r["created_at"][:10], r["predicted_label"]), []).append(r)
        return [{"day": day, "predicted_label": label, "prediction_count": len(rows),
                 "refer_count": sum(r["status"] == "refer" for r in rows),
                 "avg_inference_ms": sum(r["inference_ms"] for r in rows) / len(rows)}
                for (day, label), rows in groups.items()]

    def factory(self, token):
        user_id = jwt.decode(token, options={"verify_signature": False})["sub"]
        client = FakeDB(self, user_id)
        self.clients.append(client)
        return client


class FakeDB:
    def __init__(self, store, user_id):
        self.store, self.user_id, self.closed = store, user_id, False

    def table(self, name):
        return FakeQuery(self, name)

    def aclose(self):
        self.closed = True


class BrightnessModel(nn.Module):
    """Bright image -> confident Normal; dark image -> uncertain Tuberculosis (refer)."""

    def forward(self, x):
        logits = [8.0, 0.0, 0.0] if x.mean() > 0 else [0.0, 0.2, 0.6]
        return torch.tensor([logits]).expand(x.shape[0], -1), None


def png_bytes(color=255, size=(64, 64), mode="L"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


BRIGHT, DARK = png_bytes(255), png_bytes(0)


def upload(data=BRIGHT, content_type="image/png", name="xray.png"):
    return {"file": (name, data, content_type)}


@pytest.fixture
def model_service(monkeypatch, calibration_file):
    monkeypatch.setattr(ModelService, "_load_model", lambda self, path: BrightnessModel())
    service = ModelService("unused.pth", calibration_path=calibration_file(0.5))
    service.heatmap_calls = 0

    def generate_heatmap(image, prediction_result=None, tracer=None):
        service.heatmap_calls += 1
        with trace_span(tracer, "gradcam", output="figure"):
            return Image.new("RGB", (30, 10), "red")

    def compute_gradcam(image, class_name, tracer=None):
        with trace_span(tracer, "gradcam", output="map", explained_class=class_name):
            return np.full((7, 7), 0.5)

    service.generate_heatmap, service.compute_gradcam = generate_heatmap, compute_gradcam
    return service


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def make_client(model_service, store, tmp_path, monkeypatch, backend, make_redis):
    """Every endpoint test runs twice: uploads and report cache in memory, and in (fake) Redis."""
    monkeypatch.setattr("services.tracing.TRACE_DIR", tmp_path / "traces")

    def make(llm=None, **settings):
        settings = Settings(supabase_url=SUPABASE_URL, supabase_key="anon", jwt_secret=SECRET, **settings)
        if backend == "memory":
            uploads, cache = UploadStore(settings.upload_ttl_s, settings.upload_max_items), ReportCache()
        else:
            uploads = RedisUploadStore(make_redis(), settings.upload_ttl_s, settings.upload_max_items)
            cache = RedisReportCache(make_redis())
        app = create_app(settings, model_service=model_service, db_factory=store.factory,
                         llm_factory=llm or (lambda: None), report_cache=cache, uploads=uploads)
        return TestClient(app)
    return make


@pytest.fixture
def client(make_client):
    return make_client()


def predict(client, data=BRIGHT, user=ALICE):
    res = client.post("/predict", files=upload(data), headers=auth(user))
    assert res.status_code == 200, res.text
    return res.json()


# ---- Health ----

def test_health_needs_no_login(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "calibrated": True, "temperature": 0.5,
                    "auth_configured": True, "llm_configured": False}


# ---- Auth ----

TRACE = str(uuid.uuid4())
PROTECTED = [("post", "/predict"), ("get", f"/gradcam/{TRACE}"), ("get", f"/gradcam/{TRACE}/map"),
             ("post", f"/report/{TRACE}"),
             ("get", "/history"), ("get", "/analytics"), ("get", f"/traces/{TRACE}")]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_every_endpoint_but_health_needs_a_token(client, method, path):
    res = getattr(client, method)(path)
    assert res.status_code == 401
    assert res.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("token", [
    "not-a-jwt",
    make_token(secret="some-other-secret-that-is-also-long-enough"),
    make_token(exp=int(time.time()) - 3600),
    make_token(aud="anon"),
    make_token(iss="https://other.supabase.co/auth/v1"),
    jwt.encode({"aud": "authenticated", "iss": ISSUER, "exp": int(time.time()) + 60}, SECRET),  # no sub
])
def test_bad_tokens_are_rejected(client, token):
    res = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401


class UnreachableJWKS:
    """The JWKS endpoint of a Supabase project whose hostname doesn't resolve (a stale SUPABASE_URL)."""

    def __init__(self):
        self.calls = 0

    def _fail(self):
        self.calls += 1
        try:
            raise urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))
        except urllib.error.URLError as e:
            raise jwt.PyJWKClientConnectionError(f'Fail to fetch data from the url, err: "{e}"') from e

    def get_signing_key_from_jwt(self, token):
        self._fail()

    def get_signing_keys(self):
        self._fail()


OTHER_ISSUER = "https://another-project.supabase.co/auth/v1"


def es256_token(**claims):
    key = ec.generate_private_key(ec.SECP256R1())
    return make_token(secret=key, alg="ES256", **claims)


def test_token_from_another_project_is_401_without_fetching_keys(caplog):
    # Regression: a stale SUPABASE_URL in the shell pointed the API at a dead project, so every login
    # got 503 "Could not fetch Supabase signing keys". A token from another project is a 401 that
    # names both projects, and needs no key download.
    jwks = UnreachableJWKS()
    verifier = SupabaseJWTVerifier(SUPABASE_URL, jwk_client=jwks)
    with pytest.raises(AuthError) as err, caplog.at_level(logging.WARNING, logger="api.auth"):
        verifier.verify(es256_token(iss=OTHER_ISSUER))
    assert OTHER_ISSUER in str(err.value) and ISSUER in str(err.value)
    assert jwks.calls == 0
    assert "Rejected token" in caplog.text


@pytest.mark.parametrize("claims", [
    {"exp": int(time.time()) - 3600},   # expired
    {"aud": "anon"},
    {"sub": ""},
])
def test_bad_tokens_are_401_even_when_signing_keys_are_unreachable(claims):
    jwks = UnreachableJWKS()
    with pytest.raises(AuthError):
        SupabaseJWTVerifier(SUPABASE_URL, jwk_client=jwks).verify(es256_token(**claims))
    assert jwks.calls == 0


def test_unreachable_signing_keys_are_503_and_the_cause_is_logged(caplog):
    verifier = SupabaseJWTVerifier(SUPABASE_URL, jwk_client=UnreachableJWKS())
    with pytest.raises(AuthUnavailable), caplog.at_level(logging.ERROR, logger="api.auth"):
        verifier.verify(es256_token())
    assert f"{ISSUER}/.well-known/jwks.json" in caplog.text
    assert "getaddrinfo failed" in caplog.text
    with caplog.at_level(logging.ERROR, logger="api.auth"):
        assert verifier.check_keys() is False  # the startup check logs the same cause


def test_http_status_for_wrong_project_expired_and_unreachable_keys(model_service, store):
    verifier = SupabaseJWTVerifier(SUPABASE_URL, jwk_client=UnreachableJWKS())
    app = create_app(Settings(supabase_url=SUPABASE_URL, supabase_key="anon"), model_service=model_service,
                     verifier=verifier, db_factory=store.factory, llm_factory=lambda: None,
                     report_cache=ReportCache())
    client = TestClient(app)

    def status(token):
        return client.get("/history", headers={"Authorization": f"Bearer {token}"}).status_code

    assert status(es256_token(iss=OTHER_ISSUER)) == 401
    assert status(es256_token(exp=int(time.time()) - 3600)) == 401
    assert status("not-a-jwt") == 401
    assert status(es256_token()) == 503  # a token that could be valid, but the keys can't be fetched


def test_env_supabase_url_that_differs_from_secrets_is_warned_about(tmp_path, caplog):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[supabase]\nurl = "https://current.supabase.co"\nkey = "anon-current"\n')
    with caplog.at_level(logging.WARNING, logger="api.settings"):
        settings = Settings.from_env({"SUPABASE_URL": "https://stale.supabase.co"}, secrets_path=secrets)
    assert settings.supabase_url == "https://stale.supabase.co"   # the environment still wins (deploys)
    assert settings.supabase_url_source == "environment variable SUPABASE_URL"
    assert "stale.supabase.co" in caplog.text and "current.supabase.co" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="api.settings"):
        settings = Settings.from_env({}, secrets_path=secrets)
    assert settings.supabase_url == "https://current.supabase.co" and caplog.text == ""


def test_asymmetric_tokens_are_checked_against_jwks():
    key = ec.generate_private_key(ec.SECP256R1())
    jwks = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=key.public_key()))
    verifier = SupabaseJWTVerifier(SUPABASE_URL, jwk_client=jwks)
    user = verifier.verify(make_token(secret=key, alg="ES256"))
    assert user.id == ALICE
    other = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(AuthError):
        verifier.verify(make_token(secret=other, alg="ES256"))
    with pytest.raises(AuthError):  # an HS256 token can't pose as the asymmetric key's
        SupabaseJWTVerifier(SUPABASE_URL, jwk_client=jwks, remote=SimpleNamespace(
            get_user=lambda t: None)).verify(make_token(alg="HS256"))


def test_hs256_without_secret_asks_supabase():
    seen = []

    def get_user(token):
        seen.append(token)
        return SimpleNamespace(user=SimpleNamespace(id=BOB, email="bob@example.com"))

    verifier = SupabaseJWTVerifier(SUPABASE_URL, jwk_client=object(), remote=SimpleNamespace(get_user=get_user))
    token = make_token(BOB)
    assert verifier.verify(token).id == BOB and seen == [token]


# ---- /predict ----

def test_predict_returns_calibrated_result_and_logs_it(client, store):
    body = predict(client)
    assert body["predicted_class"] == "Normal" and body["status"] == "confident"
    assert body["confidence"] > 0.99 and abs(sum(body["probabilities"].values()) - 1) < 1e-6
    assert body["referral_message"] is None and body["calibrated"] is True and body["warnings"] == []
    uuid.UUID(body["trace_id"])

    [row] = store.tables["predictions"]
    assert row["user_id"] == ALICE and row["trace_id"] == body["trace_id"] and row["status"] == "confident"
    names = [s["name"] for s in store.tables["traces"]]
    assert names == ["prediction", "preprocess", "classify", "calibrate", "log_prediction"]
    assert all(c.closed for c in store.clients)


def test_uncertain_prediction_is_referred(client):
    body = predict(client, DARK)
    assert body["predicted_class"] == "Tuberculosis" and body["status"] == "refer"
    assert body["referral_message"] == REFER_MESSAGE and body["confidence"] < 0.9


def test_logging_failure_is_a_warning_not_an_error(client, store):
    store.failing.add("predictions")
    body = predict(client)
    assert body["predicted_class"] == "Normal"
    assert body["warnings"] == ["Could not log prediction (RuntimeError)"]


@pytest.mark.parametrize("files,status", [
    (upload(b"hello", "text/plain", "notes.txt"), 415),
    (upload(b"\x89PNG not really", "image/png"), 415),
    (upload(Image.new("RGB", (8, 8)).tobytes(), "image/jpeg", "x.jpg"), 415),
    (upload(b"", "image/png"), 400),
])
def test_upload_type_is_checked(client, files, status):
    assert client.post("/predict", files=files, headers=auth()).status_code == status


def test_gif_disguised_as_png_is_rejected(client):
    buf = io.BytesIO()
    Image.new("L", (8, 8)).save(buf, format="GIF")
    assert client.post("/predict", files=upload(buf.getvalue()), headers=auth()).status_code == 415


def test_upload_size_limit(make_client):
    client = make_client(max_upload_bytes=2000)
    noise = png_bytes(size=(80, 80))
    noisy = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (80, 80), dtype=np.uint8)).save(noisy, "PNG")
    assert len(noise) < 2000 < len(noisy.getvalue()) < 64 * 1024
    assert client.post("/predict", files=upload(noise), headers=auth()).status_code == 200
    # Over the limit but small enough to pass the early header check: caught after reading
    res = client.post("/predict", files=upload(noisy.getvalue()), headers=auth())
    assert res.status_code == 413 and "limit" in res.json()["detail"]
    # Far over: rejected from Content-Length before the body is parsed, still with CORS headers
    res = client.post("/predict", files=upload(b"0" * 200_000), headers={**auth(), "Origin": "http://localhost:5173"})
    assert res.status_code == 413
    assert res.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_prediction_does_not_block_the_event_loop(model_service, store, tmp_path, monkeypatch):
    """While one prediction is stuck in its worker thread, /health still answers."""
    monkeypatch.setattr("services.tracing.TRACE_DIR", tmp_path)
    entered, release, released = threading.Event(), threading.Event(), []
    real_predict = model_service.predict

    def slow_predict(image, tracer=None):
        entered.set()
        # The test releases this only after /health answered; if predict ran on the event loop,
        # /health couldn't answer, so this wait would time out instead
        released.append(release.wait(timeout=3))
        return real_predict(image, tracer=tracer)

    model_service.predict = slow_predict
    app = create_app(Settings(supabase_url=SUPABASE_URL, supabase_key="anon", jwt_secret=SECRET),
                     model_service=model_service, db_factory=store.factory, llm_factory=lambda: None)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            pending = asyncio.create_task(http.post("/predict", files=upload(), headers=auth()))
            await asyncio.to_thread(entered.wait, 3)
            health = await http.get("/health")
            release.set()
            return health.status_code, (await pending).status_code

    assert asyncio.run(scenario()) == (200, 200)
    assert released == [True]


# ---- /gradcam ----

def test_gradcam_returns_png_once_per_upload(client, model_service, store):
    trace_id = predict(client)["trace_id"]
    for _ in range(2):
        res = client.get(f"/gradcam/{trace_id}", headers=auth())
        assert res.status_code == 200 and res.headers["content-type"] == "image/png"
        assert res.content.startswith(b"\x89PNG")
    assert model_service.heatmap_calls == 1
    assert [s["name"] for s in store.tables["traces"] if s["trace_id"] == trace_id].count("gradcam") == 1


def test_gradcam_is_private_to_the_uploader(client):
    trace_id = predict(client)["trace_id"]
    assert client.get(f"/gradcam/{trace_id}", headers=auth(BOB)).status_code == 404
    assert client.get(f"/gradcam/{uuid.uuid4()}", headers=auth()).status_code == 404
    assert client.get("/gradcam/not-a-uuid", headers=auth()).status_code == 422


def test_gradcam_map_returns_the_raw_map_once_per_upload(client, model_service, store):
    trace_id = predict(client)["trace_id"]
    calls = []
    original = model_service.compute_gradcam
    model_service.compute_gradcam = lambda *a, **k: calls.append(a[1]) or original(*a, **k)
    for _ in range(2):
        res = client.get(f"/gradcam/{trace_id}/map", headers=auth())
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["explained_class"] == "Normal" and calls == ["Normal"]
        assert len(body["cam"]) == 7 and all(len(row) == 7 for row in body["cam"])
        assert body["cam"][0][0] == 0.5
        assert body["regions"]["focus"] == "diffuse" and len(body["regions"]["region_shares"]) == 6
    spans = [s for s in store.tables["traces"] if s["trace_id"] == trace_id and s["name"] == "gradcam"]
    assert len(spans) == 1 and spans[0]["attributes"]["output"] == "map"


def test_gradcam_map_is_private_to_the_uploader(client):
    trace_id = predict(client)["trace_id"]
    assert client.get(f"/gradcam/{trace_id}/map", headers=auth(BOB)).status_code == 404
    assert client.get(f"/gradcam/{trace_id}/map").status_code == 401


# ---- /report ----

def report_script(truth):
    report = {
        "predicted_class": truth["predicted_class"],
        "calibrated_probability": round(truth["confidence"], 4),
        "status": truth["status"],
        "summary": f"The model's most likely category is {truth['predicted_class']}. "
                   "This is a screening result, not a diagnosis.",
        "recommendation": "Share this result with a clinician at the next visit.",
        "limitations": ["Evaluated on two public datasets only."],
        "disclaimer": "Not medical advice.",
    }
    return [MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"),
            MockLLMClient.tool("get_condition_info"), MockLLMClient.final(report)]


def test_report_without_llm_key_uses_the_template(client):
    truth = predict(client, DARK)
    res = client.post(f"/report/{truth['trace_id']}", headers=auth())
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "fallback" and body["reason"].startswith("No LLM configured")
    assert body["report"]["predicted_class"] == "Tuberculosis" and body["report"]["status"] == "refer"
    assert "clinician review" in body["report"]["recommendation"].lower()


def test_report_by_llm_is_guarded_cached_and_traced(make_client, store):
    llms = []

    def llm_factory():
        llms.append(MockLLMClient(report_script(truth)))
        return llms[-1]

    client = make_client(llm=llm_factory)
    truth = predict(client)
    first = client.post(f"/report/{truth['trace_id']}", headers=auth()).json()
    assert first["source"] == "llm" and first["cached"] is False and first["model"] == "mock"
    assert first["report"]["calibrated_probability"] == round(truth["confidence"], 4)
    second = client.post(f"/report/{truth['trace_id']}", headers=auth()).json()
    assert second["cached"] is True and second["report"] == first["report"]
    assert len(llms[1].calls) == 0  # the cache hit spent no LLM calls

    spans = client.get(f"/traces/{truth['trace_id']}", headers=auth()).json()["spans"]
    names = [s["name"] for s in spans]
    assert names[0] == "prediction" and {"report", "llm_call", "tool_call", "guard", "report_cache_hit"} <= set(names)


def test_report_guard_rejects_a_wrong_probability(make_client):
    def llm_factory():
        wrong = report_script({**truth, "confidence": 0.5})
        return MockLLMClient(wrong + [wrong[-1]])

    client = make_client(llm=llm_factory)
    truth = predict(client)
    body = client.post(f"/report/{truth['trace_id']}", headers=auth()).json()
    assert body["source"] == "fallback"  # same guard and fallback as the Streamlit app
    assert body["report"]["calibrated_probability"] == round(truth["confidence"], 4)


def test_report_is_private_to_the_uploader(client):
    trace_id = predict(client)["trace_id"]
    assert client.post(f"/report/{trace_id}", headers=auth(BOB)).status_code == 404


# ---- /history ----

def test_history_is_paginated_filtered_and_per_user(client):
    ids = [predict(client, data)["trace_id"] for data in (BRIGHT, DARK, BRIGHT)]
    predict(client, user=BOB)

    page = client.get("/history?limit=2", headers=auth()).json()
    assert page["total"] == 3 and page["limit"] == 2 and page["offset"] == 0
    assert [i["trace_id"] for i in page["items"]] == ids[::-1][:2]  # newest first
    rest = client.get("/history?limit=2&offset=2", headers=auth()).json()
    assert [i["trace_id"] for i in rest["items"]] == [ids[0]]

    refer = client.get("/history?status=refer", headers=auth()).json()
    assert refer["total"] == 1 and refer["items"][0]["predicted_label"] == "Tuberculosis"
    assert client.get("/history?label=Normal", headers=auth()).json()["total"] == 2
    assert client.get("/history?date_from=2026-10-20&date_to=2026-10-20", headers=auth()).json()["total"] == 3
    assert client.get("/history?date_to=2026-10-19", headers=auth()).json()["total"] == 0
    assert client.get("/history", headers=auth(BOB)).json()["total"] == 1


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "label=Cancer", "status=maybe",
                                   "date_from=2026-10-21&date_to=2026-10-20", "page=2"])
def test_history_rejects_bad_queries(client, query):
    assert client.get(f"/history?{query}", headers=auth()).status_code == 422


# ---- /analytics ----

def test_analytics_volume_referral_rate_and_latency(client, monkeypatch):
    monkeypatch.setattr("api.main.datetime", SimpleNamespace(now=lambda tz: datetime(2026, 10, 20, tzinfo=tz)))
    for data in (BRIGHT, DARK):
        predict(client, data)
    predict(client, user=BOB)

    body = client.get("/analytics", headers=auth()).json()
    assert body["total_predictions"] == 2 and body["total_referred"] == 1 and body["referral_rate"] == 0.5
    [day] = body["daily"]
    assert day["day"] == "2026-10-20" and day["by_label"] == {"Normal": 1, "Tuberculosis": 1}
    steps = body["pipeline"]["steps"]
    assert steps["classify"]["count"] == 2 and body["pipeline"]["traces"] == 2
    assert body["warnings"] == []


def test_analytics_survives_a_missing_traces_table(client, store):
    predict(client)
    store.failing.add("traces")
    body = client.get("/analytics", headers=auth()).json()
    assert body["pipeline"] is None and body["warnings"] == ["Could not load traces (RuntimeError)"]


def test_daily_volume_aggregates_labels_within_the_window():
    rows = [
        {"day": "2026-10-20", "predicted_label": "Normal", "prediction_count": 3, "refer_count": 0, "avg_inference_ms": 10},
        {"day": "2026-10-20", "predicted_label": "Pneumonia", "prediction_count": 1, "refer_count": 1, "avg_inference_ms": 50},
        {"day": "2026-10-19", "predicted_label": "Normal", "prediction_count": 2, "refer_count": None, "avg_inference_ms": None},
        {"day": "2026-09-01", "predicted_label": "Normal", "prediction_count": 9, "refer_count": 9, "avg_inference_ms": 1},
    ]
    out = daily_volume(rows, days=7, today=date(2026, 10, 20))
    assert [d.day for d in out] == ["2026-10-20", "2026-10-19"]
    assert (out[0].predictions, out[0].referred, out[0].referral_rate, out[0].avg_inference_ms) == (4, 1, 0.25, 20.0)
    assert out[1].avg_inference_ms is None and out[1].referred == 0


# ---- /traces ----

def test_traces_are_private(client):
    trace_id = predict(client)["trace_id"]
    spans = client.get(f"/traces/{trace_id}", headers=auth()).json()["spans"]
    assert spans[0]["name"] == "prediction" and spans[0]["parent_span_id"] is None
    assert spans[1]["parent_span_id"] == spans[0]["span_id"]
    assert client.get(f"/traces/{trace_id}", headers=auth(BOB)).status_code == 404


# ---- CORS ----

@pytest.mark.parametrize("origin,allowed", [
    ("http://localhost:5173", True), ("http://127.0.0.1:3000", True), ("https://app.example.com", True),
    ("https://evil.example.com", False), ("http://localhost.evil.com", False),
])
def test_cors(make_client, origin, allowed):
    client = make_client(frontend_urls=["https://app.example.com/"])
    res = client.options("/predict", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                              "Access-Control-Request-Headers": "authorization"})
    assert (res.headers.get("access-control-allow-origin") == origin) is allowed


# ---- State and settings ----

def test_upload_store_expires_evicts_and_checks_owner(make_upload_store, redis_clock):
    uploads = make_upload_store(ttl_s=10, max_items=2)
    uploads.put("a", ALICE, b"1", {})
    assert uploads.get("a", ALICE) is not None and uploads.get("a", BOB) is None
    uploads.put("b", ALICE, b"2", {})
    uploads.put("c", ALICE, b"3", {})
    assert uploads.get("a", ALICE) is None and len(uploads) == 2  # oldest evicted
    redis_clock[0] += 11
    assert uploads.get("b", ALICE) is None and len(uploads) == 0


def redis_replicas(model_service, store, make_redis, llm_factory=lambda: None):
    """Two apps (replicas) whose uploads and report cache share one fake Redis."""
    def app():
        settings = Settings(supabase_url=SUPABASE_URL, supabase_key="anon", jwt_secret=SECRET)
        return TestClient(create_app(settings, model_service=model_service, db_factory=store.factory,
                                     llm_factory=llm_factory, uploads=RedisUploadStore(make_redis()),
                                     report_cache=RedisReportCache(make_redis())))
    return app(), app()


def test_replicas_share_uploads_gradcam_and_reports_through_redis(model_service, store, make_redis, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr("services.tracing.TRACE_DIR", tmp_path)
    llms = []

    def llm_factory():
        llms.append(MockLLMClient(report_script(truth)))
        return llms[-1]

    a, b = redis_replicas(model_service, store, make_redis, llm_factory)
    truth = predict(a)
    trace_id = truth["trace_id"]

    assert b.get(f"/gradcam/{trace_id}", headers=auth()).status_code == 200
    assert a.get(f"/gradcam/{trace_id}", headers=auth()).status_code == 200
    assert model_service.heatmap_calls == 1  # A used the PNG that B rendered

    calls = []
    original = model_service.compute_gradcam
    model_service.compute_gradcam = lambda *args, **kw: calls.append(args[1]) or original(*args, **kw)
    first_map = a.get(f"/gradcam/{trace_id}/map", headers=auth()).json()
    assert b.get(f"/gradcam/{trace_id}/map", headers=auth()).json() == first_map and calls == ["Normal"]

    first = a.post(f"/report/{trace_id}", headers=auth()).json()
    second = b.post(f"/report/{trace_id}", headers=auth()).json()
    assert first["source"] == "llm" and not first["cached"]
    assert second["cached"] and second["report"] == first["report"] and len(llms[1].calls) == 0

    for client in (a, b):  # Bob can't reach Alice's upload on either replica
        assert client.get(f"/gradcam/{trace_id}", headers=auth(BOB)).status_code == 404
        assert client.get(f"/gradcam/{trace_id}/map", headers=auth(BOB)).status_code == 404
        assert client.post(f"/report/{trace_id}", headers=auth(BOB)).status_code == 404


def test_redis_down_is_503_not_a_crash(model_service, store, make_redis, redis_server, tmp_path, monkeypatch):
    monkeypatch.setattr("services.tracing.TRACE_DIR", tmp_path)
    a, _ = redis_replicas(model_service, store, make_redis)
    trace_id = predict(a)["trace_id"]
    redis_server.connected = False
    res = a.post("/predict", files=upload(), headers=auth())
    assert res.status_code == 503 and "Shared upload store unavailable" in res.json()["detail"]
    assert a.get(f"/gradcam/{trace_id}", headers=auth()).status_code == 503
    assert a.post(f"/report/{trace_id}", headers=auth()).status_code == 503


def test_backend_comes_from_redis_url():
    memory = Settings(supabase_url=SUPABASE_URL)
    uploads, cache, client = shared_state(memory)
    assert type(uploads) is UploadStore and cache is REPORT_CACHE and client is None

    redis_settings = Settings(supabase_url=SUPABASE_URL, redis_url="redis://:pw@redis.internal:6380/2")
    uploads, cache, client = shared_state(redis_settings)  # redis-py connects lazily: no network here
    assert isinstance(uploads, RedisUploadStore) and isinstance(cache, RedisReportCache)
    assert uploads.client is client and cache.client is client

    injected = UploadStore()  # empty, so falsy: must still be used
    uploads, cache, client = shared_state(redis_settings, uploads=injected, report_cache=ReportCache())
    assert uploads is injected and type(cache) is ReportCache and client is None


def test_redis_url_from_env_or_secrets(tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[supabase]\nurl = "https://s.supabase.co"\n[redis]\nurl = "redis://file:6379/0"\n',
                       encoding="utf-8")
    assert Settings.from_env({}, secrets).redis_url == "redis://file:6379/0"
    assert Settings.from_env({"REDIS_URL": "rediss://env:6380/1"}, secrets).redis_url == "rediss://env:6380/1"
    assert Settings.from_env({}, tmp_path / "none.toml").redis_url is None


def test_redis_startup_check_logs_location_but_never_the_password(redis_server, caplog):
    client = connect_redis("redis://:s3cret-pw@redis.internal:6380/2")
    assert redis_location(client) == "redis.internal:6380/2"
    import fakeredis
    fake = fakeredis.FakeRedis(server=redis_server)
    fake.connection_pool.connection_kwargs.update(host="redis.internal", password="s3cret-pw")
    redis_server.connected = False
    with caplog.at_level(logging.INFO, logger="api.main"):
        assert check_redis(fake) is False
    assert "unreachable" in caplog.text and "s3cret-pw" not in caplog.text


def test_serialized_model_runs_one_call_at_a_time():
    active, overlaps = [0], []

    class Service:
        temperature = 0.5

        def predict(self, image, tracer=None):
            active[0] += 1
            overlaps.append(active[0])
            time.sleep(0.01)
            active[0] -= 1

        compute_gradcam = generate_heatmap = predict

    model = SerializedModel(Service())
    threads = [threading.Thread(target=getattr(model, name), args=(None,))
               for name in ("predict", "compute_gradcam", "generate_heatmap") * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(overlaps) == 1 and model.temperature == 0.5


def test_settings_from_env_and_secrets(tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[supabase]\nurl = "https://s.supabase.co/"\nkey = "anon-from-file"\n', encoding="utf-8")
    s = Settings.from_env({"FRONTEND_URL": "https://a.example, https://b.example/", "MAX_UPLOAD_MB": "2"}, secrets)
    assert (s.supabase_url, s.supabase_key, s.jwt_secret) == ("https://s.supabase.co", "anon-from-file", None)
    assert s.frontend_urls == ["https://a.example", "https://b.example"] and s.max_upload_bytes == 2 * 1024 * 1024
    s = Settings.from_env({"SUPABASE_URL": "https://env.supabase.co", "SUPABASE_KEY": "k"}, tmp_path / "none.toml")
    assert (s.supabase_url, s.supabase_key, s.frontend_urls) == ("https://env.supabase.co", "k", [])


def test_api_draws_figures_without_a_display():
    """Outside Streamlit, matplotlib picked Tk on Windows and Grad-CAM failed in worker threads."""
    import matplotlib
    import api  # noqa: F401
    assert matplotlib.get_backend().lower() == "agg"

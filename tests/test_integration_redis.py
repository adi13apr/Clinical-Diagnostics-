"""Integration tests against a real Redis server (CI job "integration": a redis:7 service container).

Skipped unless INTEGRATION_REDIS_URL is set. That database is FLUSHED before and after every test,
so point it at a scratch DB only, e.g. redis://localhost:6379/15 (CI), or the Compose stack's Redis
on a DB it doesn't use: redis://127.0.0.1:6380/15.

What fakeredis can't vouch for: real Lua (RESERVE/RECORD scripts), real WATCH/MULTI between
connections, real key expiry, and a real RQ worker taking a job off the queue. Groq and Supabase are
still faked (MockLLMClient, FakeStore), so no network beyond Redis.
"""
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from rq import SimpleWorker

from agent.agent import MockLLMClient, RedisReportCache
from agent.rate_limit import RedisModelBudget
from api.jobs import ReportJobs
from api.main import create_app
from api.settings import Settings
from api.state import RedisUploadStore, SerializedModel, connect_redis
from test_api import (  # noqa: F401  (model_service and store are fixtures)
    ALICE, BOB, SECRET, SUPABASE_URL, auth, model_service, predict, report_script, store,
)
from worker import report_worker
from worker.report_worker import WorkerContext

URL = os.environ.get("INTEGRATION_REDIS_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set INTEGRATION_REDIS_URL to a scratch Redis DB (it is flushed)")


class WouldWait(Exception):
    pass


def would_wait(seconds):
    raise WouldWait(seconds)


@pytest.fixture
def real_redis():
    """make() -> a new client on the real server: each one is like a separate replica's connection."""
    clients = []

    def make():
        clients.append(connect_redis(URL))
        return clients[-1]

    make().flushdb()
    yield make
    clients[0].flushdb()
    for c in clients:
        c.close()


@pytest.fixture
def replica(model_service, store, real_redis, tmp_path, monkeypatch):
    """replica(llm_factory, queue) -> TestClient of an API replica with its own Redis connections."""
    monkeypatch.setattr("services.tracing.TRACE_DIR", tmp_path / "traces")

    def make(llm_factory=lambda: None, queue=False):
        settings = Settings(supabase_url=SUPABASE_URL, supabase_key="anon", jwt_secret=SECRET)
        return TestClient(create_app(
            settings, model_service=model_service, db_factory=store.factory, llm_factory=llm_factory,
            uploads=RedisUploadStore(real_redis()), report_cache=RedisReportCache(real_redis()),
            report_jobs=ReportJobs(real_redis()) if queue else None))
    return make


def test_replicas_share_uploads_gradcam_and_the_report_cache(replica, model_service):
    llms = []

    def llm_factory():
        llms.append(MockLLMClient(report_script(truth)))
        return llms[-1]

    a, b = replica(llm_factory), replica(llm_factory)
    truth = predict(a)
    trace_id = truth["trace_id"]

    assert b.get(f"/gradcam/{trace_id}", headers=auth()).status_code == 200
    assert a.get(f"/gradcam/{trace_id}", headers=auth()).status_code == 200
    assert model_service.heatmap_calls == 1  # A served the PNG that B stored
    assert b.get(f"/gradcam/{trace_id}/map", headers=auth()).json() == \
        a.get(f"/gradcam/{trace_id}/map", headers=auth()).json()

    first = a.post(f"/report/{trace_id}", headers=auth()).json()["result"]
    second = b.post(f"/report/{trace_id}", headers=auth()).json()["result"]
    assert first["source"] == "llm" and not first["cached"]
    assert second["cached"] and second["report"] == first["report"]
    for client in (a, b):
        assert client.get(f"/gradcam/{trace_id}", headers=auth(BOB)).status_code == 404


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="RQ workers need SIGALRM (Linux/macOS)")
def test_a_real_worker_writes_a_job_queued_on_one_replica_and_polled_on_another(
        replica, model_service, store, real_redis):
    def llm_factory():
        return MockLLMClient(report_script(truth))

    a, b = replica(queue=True), replica(queue=True)
    truth = predict(a)  # confident Normal: report_script passes the guard (a refer case needs "clinician review")
    trace_id = truth["trace_id"]
    first = a.post(f"/report/{trace_id}", headers=auth())
    second = b.post(f"/report/{trace_id}", headers=auth())
    assert first.status_code == second.status_code == 202
    assert first.json()["status"] == "queued" and second.json()["job_id"] == first.json()["job_id"]
    job_id = first.json()["job_id"]

    previous = report_worker.configure(WorkerContext(
        model=SerializedModel(model_service), uploads=RedisUploadStore(real_redis()),
        cache=RedisReportCache(real_redis()), llm_factory=llm_factory, jobs=ReportJobs(real_redis())))
    try:
        jobs = ReportJobs(real_redis())
        worked = SimpleWorker([jobs.queue], connection=jobs.client, serializer=jobs.queue.serializer).work(
            burst=True)
    finally:
        report_worker.configure(previous)
    assert worked

    body = b.get(f"/report/jobs/{job_id}", headers=auth()).json()
    assert body["status"] == "done" and body["error"] is None, body
    assert body["result"]["source"] == "llm"
    assert body["result"]["report"]["predicted_class"] == truth["predicted_class"]
    rows = [r for r in store.tables.get("traces", []) if r["trace_id"] == trace_id]
    assert {"report", "llm_call", "guard"} <= {r["name"] for r in rows}  # stored by B's poll, as Alice
    assert {r["user_id"] for r in rows} == {ALICE}
    assert a.get(f"/report/jobs/{job_id}", headers=auth(BOB)).status_code == 404


def test_concurrent_submits_on_two_replicas_queue_one_job(replica, real_redis):
    a = replica(queue=True)
    trace_id = predict(a)["trace_id"]
    replicas = [ReportJobs(real_redis()), ReportJobs(real_redis())]
    ids, barrier = [], threading.Barrier(12)

    def submit(i):
        barrier.wait()
        ids.append(replicas[i % 2].submit(trace_id, ALICE)[0])

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ids) == 12 and len(set(ids)) == 1 and replicas[0].queue.count == 1


def test_lua_budget_grants_exactly_the_limit_across_replicas(real_redis):
    budgets = [RedisModelBudget(real_redis(), "it/tpm", tpm=8000, rpm=30, sleep=would_wait) for _ in range(3)]

    def try_reserve(i):
        try:
            return budgets[i % 3].acquire(1000)
        except WouldWait:
            return None

    with ThreadPoolExecutor(40) as pool:
        granted = [r for r in pool.map(try_reserve, range(40)) if r is not None]
    assert len(granted) == 8  # 8 x 1000 = tpm; the other 32 were told to wait
    assert budgets[0].used_last_minute() == (8, 8000)

    budgets[1].record(granted[0], 400)  # the real usage was lower: room for exactly one more
    budgets[2].record(granted[1], 600)
    assert budgets[0].used_last_minute() == (8, 7000)
    assert budgets[0].acquire(1000) is not None
    with pytest.raises(WouldWait):
        budgets[1].acquire(1000)
    assert budgets[2].tokens_today == 8000 and budgets[2].requests_today == 9


def test_lua_budget_request_limit_across_replicas(real_redis):
    budgets = [RedisModelBudget(real_redis(), "it/rpm", rpm=10, sleep=would_wait) for _ in range(4)]

    def try_reserve(i):
        try:
            return budgets[i % 4].acquire(1)
        except WouldWait:
            return None

    with ThreadPoolExecutor(40) as pool:
        granted = [r for r in pool.map(try_reserve, range(40)) if r is not None]
    assert len(granted) == 10
    day_key = budgets[0]._day_keys(granted[0].day)[0]
    assert 0 < real_redis().ttl(day_key) <= 24 * 3600  # EXPIREAT next UTC midnight


def test_uploads_expire_on_the_real_server(real_redis):
    uploads = RedisUploadStore(real_redis(), ttl_s=1)
    other = RedisUploadStore(real_redis(), ttl_s=1)
    uploads.put("t1", ALICE, b"png", {"diagnosis": "Normal"})
    assert other.get("t1", ALICE) is not None and other.get("t1", BOB) is None
    time.sleep(1.3)
    assert other.get("t1", ALICE) is None and len(other) == 0

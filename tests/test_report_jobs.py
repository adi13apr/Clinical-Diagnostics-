"""Report jobs (REDIS_URL set): POST /report queues, a worker writes, GET /report/jobs/{id} polls.

No real worker or Redis: fakeredis, and RQ's synchronous mode (is_async=False) runs the job inside the
POST with worker/report_worker.py's own job function. Tests that need a job to stay queued or running
use the normal async queue, with no worker, and emulate the worker's bookkeeping where needed.
"""
import json
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from rq import SimpleWorker
from rq.job import Job

from agent.agent import MockLLMClient, RedisReportCache
from api.jobs import FAILED, UPLOAD_EXPIRED, WORKER_DIED, ReportJobs
from api.main import create_app
from api.settings import Settings
from api.state import RedisUploadStore, SerializedModel
from services.tracing import SupabaseTraceStore
from test_api import (  # noqa: F401  (model_service and store are fixtures)
    ALICE, BOB, DARK, SECRET, SUPABASE_URL, auth, model_service, predict, report_script, store,
)
from worker import report_worker
from worker.report_worker import WorkerContext


@pytest.fixture
def queue_app(model_service, store, make_redis, tmp_path, monkeypatch):
    """make(llm_factory, is_async) -> (client, jobs): an API in queue mode, plus a worker context on the
    same fake Redis (its own clients, like a separate process)."""
    monkeypatch.setattr("services.tracing.TRACE_DIR", tmp_path / "traces")
    previous = []

    def make(llm_factory=lambda: None, is_async=False):
        redis = make_redis()
        jobs = ReportJobs(redis, is_async=is_async)
        previous.append(report_worker.configure(WorkerContext(
            model=SerializedModel(model_service), uploads=RedisUploadStore(make_redis()),
            cache=RedisReportCache(make_redis()), llm_factory=llm_factory, jobs=ReportJobs(make_redis()))))
        app = create_app(Settings(supabase_url=SUPABASE_URL, supabase_key="anon", jwt_secret=SECRET),
                         model_service=model_service, db_factory=store.factory, llm_factory=llm_factory,
                         uploads=RedisUploadStore(redis), report_cache=RedisReportCache(redis), report_jobs=jobs)
        return TestClient(app), jobs

    yield make
    if previous:
        report_worker.configure(previous[0])


def post_report(client, trace_id, user=ALICE):
    res = client.post(f"/report/{trace_id}", headers=auth(user))
    assert res.status_code == 202, res.text
    return res.json()


def poll(client, job_id, user=ALICE):
    return client.get(f"/report/jobs/{job_id}", headers=auth(user))


def spans_in_db(store, trace_id):
    return [r for r in store.tables.get("traces", []) if r["trace_id"] == trace_id]


def start_like_a_worker(jobs, job_id):
    """What a SimpleWorker records when it picks up a job, without running it (as if it then died)."""
    worker = SimpleWorker([jobs.queue], connection=jobs.client, serializer=jobs.queue.serializer)
    job = Job.fetch(job_id, connection=jobs.client, serializer=jobs.queue.serializer)
    worker.prepare_execution(job)
    worker.prepare_job_execution(job)


# ---- Full flow ----

def test_job_flow_writes_the_report_and_stores_spans_as_the_user(queue_app, store):
    client, jobs = queue_app()
    truth = predict(client, DARK)
    job = post_report(client, truth["trace_id"])
    assert job["job_id"] and job["status"] == "done" and job["result"] is None  # the report comes from GET

    res = poll(client, job["job_id"])
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "done" and body["error"] is None and body["warnings"] == []
    report = body["result"]
    assert report["source"] == "fallback" and report["reason"].startswith("No LLM configured")
    assert report["report"]["predicted_class"] == "Tuberculosis" and report["report"]["status"] == "refer"

    # The worker's spans reached Supabase only through this poll, as Alice (RLS fills user_id)
    rows = spans_in_db(store, truth["trace_id"])
    assert {"report", "fallback_used"} <= {r["name"] for r in rows}
    assert {r["user_id"] for r in rows} == {ALICE}
    assert not jobs.client.exists(jobs.spans_key(job["job_id"]))
    names = [s["name"] for s in client.get(f"/traces/{truth['trace_id']}", headers=auth()).json()["spans"]]
    assert names[0] == "prediction" and "report" in names  # one trace: prediction, then the worker's report

    poll(client, job["job_id"])
    assert len(spans_in_db(store, truth["trace_id"])) == len(rows)  # polling again adds nothing


def test_job_uses_the_llm_guard_and_shared_cache(queue_app):
    llms = []

    def llm_factory():
        llms.append(MockLLMClient(report_script(truth)))
        return llms[-1]

    client, _ = queue_app(llm_factory=llm_factory)
    truth = predict(client)
    first = poll(client, post_report(client, truth["trace_id"])["job_id"]).json()["result"]
    assert first["source"] == "llm" and not first["cached"]
    assert first["report"]["calibrated_probability"] == round(truth["confidence"], 4)

    again = post_report(client, truth["trace_id"])  # a done job isn't reused: a new job, a cache hit
    second = poll(client, again["job_id"]).json()["result"]
    assert second["cached"] and second["report"] == first["report"] and len(llms[-1].calls) == 0


def test_guard_rejection_falls_back_in_the_worker_too(queue_app):
    def llm_factory():
        wrong = report_script({**truth, "confidence": 0.5})
        return MockLLMClient(wrong + [wrong[-1]])

    client, _ = queue_app(llm_factory=llm_factory)
    truth = predict(client)
    result = poll(client, post_report(client, truth["trace_id"])["job_id"]).json()["result"]
    assert result["source"] == "fallback"
    assert result["report"]["calibrated_probability"] == round(truth["confidence"], 4)


def test_job_holds_only_ids_and_short_retention(queue_app):
    client, jobs = queue_app()
    trace_id = predict(client)["trace_id"]
    job_id = post_report(client, trace_id)["job_id"]
    job = Job.fetch(job_id, connection=jobs.client, serializer=jobs.queue.serializer)
    assert job.args == [trace_id, ALICE] and job.kwargs == {}
    assert job.result_ttl == 3600 and job.failure_ttl == 24 * 3600 and job.timeout == 180
    raw = jobs.client.hget(job.key, "data")
    assert b"Bearer" not in raw and b"eyJ" not in raw  # no token in the job
    assert 0 < jobs.client.ttl(job.key) <= 3600


# ---- Idempotency ----

def test_asking_again_while_queued_returns_the_same_job(queue_app):
    client, jobs = queue_app(is_async=True)  # no worker: the job stays queued
    trace_id = predict(client)["trace_id"]
    first = post_report(client, trace_id)
    second = post_report(client, trace_id)
    assert first["status"] == "queued" and second == first
    assert jobs.queue.count == 1
    assert poll(client, first["job_id"]).json()["status"] == "queued"

    start_like_a_worker(jobs, first["job_id"])
    assert post_report(client, trace_id)["job_id"] == first["job_id"]  # running: still the same job
    assert poll(client, first["job_id"]).json()["status"] == "running"


def test_concurrent_requests_queue_one_job(queue_app):
    client, jobs = queue_app(is_async=True)
    trace_id = predict(client)["trace_id"]
    ids, barrier = [], threading.Barrier(12)

    def submit():
        barrier.wait()
        ids.append(jobs.submit(trace_id, ALICE)[0])

    threads = [threading.Thread(target=submit) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ids) == 12 and len(set(ids)) == 1 and jobs.queue.count == 1


def test_a_finished_job_is_not_returned_for_a_new_request(queue_app):
    client, jobs = queue_app(is_async=True)
    trace_id = predict(client)["trace_id"]
    first = post_report(client, trace_id)["job_id"]
    jobs.queue.run_sync(Job.fetch(first, connection=jobs.client, serializer=jobs.queue.serializer))
    assert poll(client, first).json()["status"] == "done"
    assert post_report(client, trace_id)["job_id"] != first


# ---- Ownership ----

def test_only_the_owner_can_read_a_job_and_others_touch_no_spans(queue_app, store):
    client, jobs = queue_app()
    trace_id = predict(client, DARK)["trace_id"]
    job_id = post_report(client, trace_id)["job_id"]

    assert poll(client, job_id, user=BOB).status_code == 404
    assert jobs.client.llen(jobs.spans_key(job_id)) > 0  # Bob's poll read and wrote nothing
    assert "report" not in {r["name"] for r in spans_in_db(store, trace_id)}
    assert client.post(f"/report/{trace_id}", headers=auth(BOB)).status_code == 404  # nor start one on it

    assert poll(client, job_id).json()["status"] == "done"
    assert "report" in {r["name"] for r in spans_in_db(store, trace_id)}


def test_unknown_and_malformed_job_ids(queue_app):
    client, _ = queue_app()
    assert poll(client, uuid.uuid4()).status_code == 404
    assert poll(client, "not-a-uuid").status_code == 422


# ---- Failures ----

def test_a_failing_job_reports_failed_and_a_new_request_retries(queue_app, store):
    fail = [True]

    def llm_factory():
        if fail[0]:
            raise RuntimeError("provider config is broken")
        return None

    client, jobs = queue_app(llm_factory=llm_factory)
    trace_id = predict(client)["trace_id"]
    job = post_report(client, trace_id)
    assert job["status"] == "failed"
    body = poll(client, job["job_id"]).json()
    assert body["status"] == "failed" and body["error"] == FAILED and body["result"] is None
    assert "provider config" not in body["error"]  # the raw exception isn't shown to users
    assert 0 < jobs.client.ttl(f"rq:job:{job['job_id']}") <= 24 * 3600  # not RQ's one-year default

    fail[0] = False
    retry = post_report(client, trace_id)
    assert retry["job_id"] != job["job_id"] and retry["status"] == "done"


def test_upload_expired_before_the_job_ran(queue_app):
    client, jobs = queue_app(is_async=True)
    trace_id = predict(client)["trace_id"]
    job_id = post_report(client, trace_id)["job_id"]
    jobs.client.delete(f"cdp:upload:{trace_id}")  # expired while queued
    jobs.queue.run_sync(Job.fetch(job_id, connection=jobs.client, serializer=jobs.queue.serializer))
    body = poll(client, job_id).json()
    assert body["status"] == "failed" and body["error"] == UPLOAD_EXPIRED


def test_a_job_whose_worker_died_ends_failed_not_running(queue_app, monkeypatch):
    client, jobs = queue_app(is_async=True)
    trace_id = predict(client)["trace_id"]
    job_id = post_report(client, trace_id)["job_id"]
    start_like_a_worker(jobs, job_id)  # ... and the worker process is killed here
    assert poll(client, job_id).json()["status"] == "running"

    # The worker registered the job until start + timeout + 60 s; after that, a poll marks it failed
    later = int(time.time()) + jobs.job_timeout_s + 61
    monkeypatch.setattr("rq.registry.current_timestamp", lambda: later)
    body = poll(client, job_id).json()
    assert body["status"] == "failed" and body["error"] == WORKER_DIED
    assert 0 < jobs.client.ttl(f"rq:job:{job_id}") <= 24 * 3600
    assert post_report(client, trace_id)["job_id"] != job_id  # asking again starts over


def test_failed_span_write_is_retried_on_the_next_poll(queue_app, store):
    client, jobs = queue_app()
    trace_id = predict(client, DARK)["trace_id"]
    job_id = post_report(client, trace_id)["job_id"]
    buffered = {json.loads(r)["span_id"] for r in jobs.client.lrange(jobs.spans_key(job_id), 0, -1)}

    store.failing.add("traces")
    body = poll(client, job_id).json()
    assert body["status"] == "done" and body["warnings"] == ["Trace not saved yet; it is retried on the next poll"]
    assert jobs.client.llen(jobs.spans_key(job_id)) == len(buffered)  # nothing dropped

    store.failing.discard("traces")
    assert poll(client, job_id).json()["warnings"] == []
    assert not jobs.client.exists(jobs.spans_key(job_id))
    stored = [r["span_id"] for r in spans_in_db(store, trace_id) if r["span_id"] in buffered]
    assert sorted(stored) == sorted(buffered)  # each buffered span exactly once


def test_flushing_twice_never_duplicates_spans(queue_app, store):
    client, jobs = queue_app()
    trace_id = predict(client, DARK)["trace_id"]
    job_id = post_report(client, trace_id)["job_id"]
    db = store.factory(auth()["Authorization"].split()[1])
    raw = jobs.client.lrange(jobs.spans_key(job_id), 0, -1)
    SupabaseTraceStore(db).write([json.loads(r) for r in raw])  # a poll that wrote, then died before trimming
    assert jobs.flush_spans(job_id, SupabaseTraceStore(db)) == []  # the next poll writes them again
    ids = [(r["trace_id"], r["span_id"]) for r in spans_in_db(store, trace_id)]
    assert len(ids) == len(set(ids))


def test_redis_down_is_503_for_jobs(queue_app, redis_server):
    client, _ = queue_app()
    trace_id = predict(client)["trace_id"]
    job_id = post_report(client, trace_id)["job_id"]
    redis_server.connected = False
    assert poll(client, job_id).status_code == 503


# ---- Modes ----

def test_redis_url_turns_on_queue_mode():
    settings = Settings(supabase_url=SUPABASE_URL, redis_url="redis://redis.internal:6379/0")
    app = create_app(settings, model_service=object(), verifier=object(), db_factory=lambda token: None)
    assert isinstance(app.state.report_jobs, ReportJobs)
    assert app.state.report_jobs.client is app.state.uploads.client  # one connection for both
    inline = create_app(Settings(supabase_url=SUPABASE_URL), model_service=object(), verifier=object(),
                        db_factory=lambda token: None)
    assert inline.state.report_jobs is None


def test_health_shows_queue_mode(queue_app):
    client, _ = queue_app()
    body = client.get("/health").json()
    assert body["report_mode"] == "queue" and body["report_workers"] == 0


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")  # the re-run is the point
def test_python_m_configures_the_module_rq_imports(monkeypatch):
    """`python -m worker.report_worker` must run main() of the module RQ imports, not of __main__."""
    import runpy
    called = []
    monkeypatch.setattr(report_worker, "main", lambda: called.append(True))
    runpy.run_module("worker.report_worker", run_name="__main__")
    assert called == [True]


def test_worker_refuses_to_run_unconfigured(monkeypatch):
    previous = report_worker.configure(None)
    try:
        with pytest.raises(RuntimeError, match="not configured"):
            report_worker.run_report_job(str(uuid.uuid4()), ALICE)
    finally:
        report_worker.configure(previous)

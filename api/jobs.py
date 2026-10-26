"""Report jobs in Redis (RQ), for when REDIS_URL is set.

POST /report queues a job; a separate worker process (worker/report_worker.py) writes the report with
the same code as the synchronous path (write_report: cache, guard, retry, template fallback). The browser
polls GET /report/jobs/{id}.

Supabase: the worker holds no user token and no service-role key. It keeps the report's spans in a Redis
list; when the job has ended, the API writes them to `traces` with the polling user's own token, so RLS
still fills and checks user_id. The list is trimmed only after the write succeeded, and the write skips
spans already stored (sql/006), so a retry or two polls at once can't lose or duplicate spans.

Keys (prefix "cdp:" by default), next to RQ's own rq:* keys:
  report-job:active:{user_id}:{trace_id}  the newest job for this upload (job id); used for idempotency
  report-job:spans:{job_id}               list of span JSON written by the worker, until the API stores them

The job arguments are (trace_id, user_id) only: no report text, image or token. Job data is JSON
(RQ's JSONSerializer), never pickle.
"""
import json
import logging
import math
import uuid
from contextlib import contextmanager

from agent.agent import generate_report_for_image, report_result_from_json
from agent.rate_limit import WINDOW_S
from api.state import StoreUnavailable
from services.tracing import finish_trace

QUEUE_NAME = "reports"
JOB_FUNC = "worker.report_worker.run_report_job"
# RQ job_timeout: derived per job from the shared LLM budget (report_job_timeout), within these bounds
JOB_TIMEOUT_MIN_S = 180
JOB_TIMEOUT_MAX_S = 900
JOB_BASE_S = 60            # the model, Grad-CAM and the LLM requests themselves (about 10 s), plus slack
REPORT_CALLS = 4           # LLM calls in a normal report: 3 tool turns + the answer (Groq gpt-oss)
REPORT_PROMPT_TOKENS = 6200  # the limiter's prompt estimates for those 4 calls together (1,227-2,019 each, 2026-10-26)
# Multi-agent pipeline (REPORT_PIPELINE=multi), with one revision; the limiter's estimates on real tool outputs
# (2026-10-26): evidence 551 + 691 + 921, writer 1,473, revision 1,946; verifier 1,479 per call
MULTI_REPORT_CALLS = 5           # on the report model: 3 evidence tool turns + a draft + a revision
MULTI_REPORT_PROMPT_TOKENS = 5600
MULTI_VERIFIER_CALLS = 2         # on the verifier model: the draft and the revision
MULTI_VERIFIER_PROMPT_TOKENS = 3000
DEADLINE_MARGIN_S = 30     # the worker stops waiting for the budget this long before the job timeout
QUEUED_TTL_S = 600         # a job no worker started within 10 min is dropped (e.g. no worker running)
RESULT_TTL_S = 3600        # a finished job and its report are kept 1 h
FAILURE_TTL_S = 24 * 3600  # a failed job is kept 24 h (RQ's default is a year)
SUBMIT_ATTEMPTS = 5        # optimistic-lock retries when two POSTs race for the same upload

# RQ status -> API status
STATUSES = {
    "created": "queued", "queued": "queued", "deferred": "queued", "scheduled": "queued",
    "started": "running",
    "finished": "done",
    "failed": "failed", "stopped": "failed", "canceled": "failed",
}
ACTIVE = {"queued", "running"}

UPLOAD_EXPIRED = "The upload has expired. Run the prediction again, then ask for the report."
WORKER_DIED = "The report worker stopped before finishing. Please try again."
FAILED = "The report could not be written. Please try again."

log = logging.getLogger(__name__)


def write_report(model, upload, llm_factory, cache, tracer, deadline=None):
    """The report for one upload: shared by the synchronous API and the worker, so both behave the same.

    deadline (time.monotonic()): LLM calls stop waiting for the rate-limit budget then, and the report
    falls back to the template (the worker sets it before RQ's job timeout would kill the job).
    """
    try:
        llm = llm_factory()
        if llm is not None and deadline is not None:
            llm.deadline = deadline
        return generate_report_for_image(model, upload.image(), prediction=upload.prediction,
                                         llm=llm, cache=cache, tracer=tracer)
    finally:
        finish_trace(tracer)  # spans of a report that raised are still kept


def _windows(workers, calls, prompt_tokens, tpm, rpm, max_completion_tokens, output_tokens=None):
    """60 s windows `workers` reports need on one model's per-minute limits. output_tokens (the calls'
    caps together) replaces calls x max_completion_tokens when the calls have different caps."""
    reserved = prompt_tokens + (calls * max_completion_tokens if output_tokens is None else output_tokens)
    windows = 1
    if tpm:
        windows = max(windows, math.ceil(workers * reserved / tpm))
    if rpm:
        windows = max(windows, math.ceil(workers * calls / rpm))
    return windows


def report_job_timeout(workers, tpm=None, rpm=None, max_completion_tokens=0, pipeline="single", verifier=None,
                       writer_max_completion_tokens=None):
    """RQ job_timeout (s) for one report, derived from the shared LLM budget.

    Every running worker draws on the same per-minute limits, so a report gets about 1/workers of
    them: it needs workers x (one report's reservations) / tpm minutes, rounded up to whole 60 s
    windows, plus JOB_BASE_S for the work itself. A report reserves its prompt estimates plus
    max_completion_tokens per call. Bounded to [JOB_TIMEOUT_MIN_S, JOB_TIMEOUT_MAX_S]; a worse than
    fair share is caught by the deadline (the template report instead of a killed job).

    pipeline="multi": the evidence agent (3 calls at max_completion_tokens) and the writer (2 calls at
    writer_max_completion_tokens) use the report model, and the verifier ({tpm, rpm, max_completion_tokens}
    of its model, a separate budget) adds its own windows, since it runs after the writer.
    """
    workers = max(int(workers or 1), 1)
    if pipeline != "multi":
        windows = _windows(workers, REPORT_CALLS, REPORT_PROMPT_TOKENS, tpm, rpm, max_completion_tokens)
    else:
        writer_cap = max_completion_tokens if writer_max_completion_tokens is None else writer_max_completion_tokens
        output = (MULTI_REPORT_CALLS - 2) * max_completion_tokens + 2 * writer_cap
        windows = _windows(workers, MULTI_REPORT_CALLS, MULTI_REPORT_PROMPT_TOKENS, tpm, rpm, max_completion_tokens,
                           output_tokens=output)
        if verifier:
            windows += _windows(workers, MULTI_VERIFIER_CALLS, MULTI_VERIFIER_PROMPT_TOKENS, **verifier)
    return int(min(max(JOB_BASE_S + WINDOW_S * windows, JOB_TIMEOUT_MIN_S), JOB_TIMEOUT_MAX_S))


def _model_budget(model, config):
    from agent.llm_config import request_options
    limits = config["limits"].get(model, {})
    return {"tpm": limits.get("tpm"), "rpm": limits.get("rpm"),
            "max_completion_tokens": int(request_options(model, config).get("max_completion_tokens", 1000))}


def report_model_budget(config=None):
    """{tpm, rpm, max_completion_tokens} of the configured report model (config/llm.json), plus
    pipeline="multi" and the verifier model's budget when REPORT_PIPELINE is multi."""
    from agent.llm_config import agent_request_options, load_llm_config, pipeline_mode, provider_settings
    config = config or load_llm_config()
    settings = provider_settings(config)[1]
    if pipeline_mode(config) == "multi" and settings["kind"] == "openai_compatible":
        def cap(agent):
            return int(agent_request_options(agent, settings[f"{agent}_model"], config).get("max_completion_tokens", 1000))
        verifier = {**_model_budget(settings["verifier_model"], config), "max_completion_tokens": cap("verifier")}
        return {**_model_budget(settings["evidence_model"], config), "max_completion_tokens": cap("evidence"),
                "pipeline": "multi", "writer_max_completion_tokens": cap("writer"), "verifier": verifier}
    return _model_budget(settings["report_model"], config)


class JobView:
    """What GET /report/jobs/{id} shows: status, plus the ReportResult when done or the reason when failed."""

    def __init__(self, job_id, trace_id, status, result=None, error=None):
        self.job_id, self.trace_id, self.status, self.result, self.error = job_id, trace_id, status, result, error


class RedisSpanBuffer:
    """Trace sink used by the worker: appends spans to a Redis list that the API later writes to Supabase."""

    def __init__(self, client, key, ttl_s):
        self.client, self.key, self.ttl_s = client, key, ttl_s

    def write(self, spans):
        pipe = self.client.pipeline()
        pipe.rpush(self.key, *[json.dumps(s, default=str) for s in spans])
        pipe.expire(self.key, self.ttl_s)
        pipe.execute()


class ReportJobs:
    """Queue report jobs, read their status (owner only) and store their spans as the polling user.

    is_async=False is RQ's synchronous mode: the job runs inside submit(), for tests (no worker needed).
    Every method can raise StoreUnavailable (Redis down -> 503).
    """

    def __init__(self, client, prefix="cdp:", is_async=True, job_timeout_s=None, budget=None,
                 queued_ttl_s=QUEUED_TTL_S, result_ttl_s=RESULT_TTL_S, failure_ttl_s=FAILURE_TTL_S):
        """job_timeout_s fixes the job timeout. None derives it per job from budget (report_model_budget()
        by default) and the number of workers listening (report_job_timeout)."""
        from rq import Queue
        from rq.serializers import JSONSerializer
        self.client, self.prefix = client, prefix
        self.queue = Queue(QUEUE_NAME, connection=client, serializer=JSONSerializer, is_async=is_async)
        self.job_timeout_s, self.queued_ttl_s = job_timeout_s, queued_ttl_s
        self.budget = budget if budget is not None or job_timeout_s is not None else report_model_budget()
        self.result_ttl_s, self.failure_ttl_s = result_ttl_s, failure_ttl_s
        # Longest a job can stay active: queued, then running until the timeout or until it counts as abandoned
        self.active_ttl_s = queued_ttl_s + (job_timeout_s or JOB_TIMEOUT_MAX_S) + 120
        self.spans_ttl_s = max(result_ttl_s, failure_ttl_s)  # as long as anyone can poll the job

    def active_key(self, user_id, trace_id):
        return f"{self.prefix}report-job:active:{user_id}:{trace_id}"

    def spans_key(self, job_id):
        return f"{self.prefix}report-job:spans:{job_id}"

    def span_buffer(self, job_id):
        return RedisSpanBuffer(self.client, self.spans_key(job_id), self.spans_ttl_s)

    def job_timeout(self):
        """Seconds a new job may run: job_timeout_s if fixed, else derived from the budget and the workers now."""
        if self.job_timeout_s is not None:
            return self.job_timeout_s
        return report_job_timeout(self.worker_count(), **self.budget)

    def submit(self, trace_id, user_id):
        """(job_id, status). The queued or running job for this upload if there is one, else a new job.

        A failed, expired or finished job is not reused: asking again queues a new one (a finished LLM
        report then comes back from the report cache). The active key and the new job are written in one
        MULTI under WATCH, so two POSTs at once get the same job, never two.
        """
        import redis
        key = self.active_key(user_id, trace_id)
        with self._errors():
            timeout = None
            for _ in range(SUBMIT_ATTEMPTS):
                current = self.client.get(key)
                if current is not None:
                    job, status = self._job_and_status(current.decode())
                    if status in ACTIVE:
                        return job.id, status
                timeout = timeout or self.job_timeout()
                job = self.queue.create_job(
                    JOB_FUNC, args=(trace_id, user_id), job_id=str(uuid.uuid4()),
                    timeout=timeout, ttl=self.queued_ttl_s,
                    result_ttl=self.result_ttl_s, failure_ttl=self.failure_ttl_s,
                    meta={"user_id": user_id, "trace_id": trace_id})
                with self.client.pipeline() as pipe:
                    try:
                        pipe.watch(key)
                        if pipe.get(key) != current:
                            continue  # another request queued a job meanwhile: look at that one
                        pipe.multi()
                        pipe.set(key, job.id, px=self.active_ttl_s * 1000)
                        job.save(pipeline=pipe)  # status "queued": a racing POST already sees an active job
                        job.cleanup(ttl=self.queued_ttl_s, pipeline=pipe)
                        pipe.execute()
                    except redis.WatchError:
                        continue
                self.queue.enqueue_job(job)  # with is_async=False this runs the job now
                _, status = self._job_and_status(job.id)
                return job.id, status
        raise StoreUnavailable("Could not queue the report (too many concurrent requests); try again")

    def get(self, job_id, user_id):
        """JobView, or None if the job is unknown, expired or another user's (callers answer 404 for all)."""
        with self._errors():
            job = self._fetch(job_id)
            # Ownership first: nothing else about someone else's job is read
            if job is None or job.meta.get("user_id") != user_id or job.func_name != JOB_FUNC:
                return None
            job, status = self._job_and_status(job_id)
            if job is None:
                return None
            view = JobView(job.id, job.meta.get("trace_id"), status)
            if status == "done":
                try:
                    view.result = report_result_from_json(job.return_value())
                except Exception as e:  # result expired or unreadable
                    log.warning("Report job %s finished but its result can't be read (%s)", job_id, type(e).__name__)
                    view.status, view.error = "failed", FAILED
            elif status == "failed":
                view.error = self.failure_reason(job)
            return view

    def flush_spans(self, job_id, trace_store):
        """Write the ended job's spans with trace_store (the polling user's Supabase), then drop them.

        Returns warnings. If the write fails the spans stay in Redis for the next poll; spans already
        stored are skipped by the database (unique (trace_id, span_id), sql/006), so two polls at once
        or a retry after a partial failure never duplicate them.
        """
        key = self.spans_key(job_id)
        with self._errors():
            raw = self.client.lrange(key, 0, -1)
        if not raw:
            return []
        try:
            trace_store.write([json.loads(r) for r in raw])
        except Exception as e:
            log.warning("Could not store the spans of report job %s (%s); kept for the next poll",
                        job_id, type(e).__name__)
            return ["Trace not saved yet; it is retried on the next poll"]
        with self._errors():
            self.client.ltrim(key, len(raw), -1)  # only what was written; an emptied list is deleted
        return []

    def failure_reason(self, job):
        """User-facing reason: the job's own (meta['error']), else from how RQ recorded the failure."""
        if job.meta.get("error"):
            return job.meta["error"]
        result = job.latest_result()
        exc = (result.exc_string or "") if result is not None else ""
        if "AbandonedJobError" in exc:
            return WORKER_DIED
        if "JobTimeoutException" in exc:
            return f"Writing the report took longer than {job.timeout} s. Please try again."
        return FAILED

    def worker_count(self):
        """Workers listening on the report queue, or None if Redis can't be asked."""
        from rq import Worker
        try:
            return Worker.count(queue=self.queue)
        except Exception:
            return None

    def _fetch(self, job_id):
        from rq.exceptions import NoSuchJobError
        from rq.job import Job
        try:
            return Job.fetch(job_id, connection=self.client, serializer=self.queue.serializer)
        except NoSuchJobError:
            return None

    def _job_and_status(self, job_id):
        """(job, API status), or (None, None) if gone. A started job whose worker died becomes failed here.

        RQ moves such a job to failed only when someone cleans the started registry (workers do it
        every 10 min). Doing it on read means the job turns failed within timeout + 60 s of starting.
        """
        job = self._fetch(job_id)
        if job is None:
            return None, None
        status = job.get_status(refresh=False)
        if status == "started":
            self.queue.started_job_registry.cleanup()
            job = self._fetch(job_id)
            if job is None:
                return None, None
            status = job.get_status(refresh=False)
        return job, STATUSES.get(str(status.value if hasattr(status, "value") else status), "queued")

    @contextmanager
    def _errors(self):
        import redis
        try:
            yield
        except redis.RedisError as e:
            raise StoreUnavailable(f"Report queue unavailable ({type(e).__name__})") from e

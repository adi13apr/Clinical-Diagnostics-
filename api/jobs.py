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
import uuid
from contextlib import contextmanager

from agent.agent import generate_report_for_image, report_result_from_json
from api.state import StoreUnavailable
from services.tracing import finish_trace

QUEUE_NAME = "reports"
JOB_FUNC = "worker.report_worker.run_report_job"
JOB_TIMEOUT_S = 180        # the worker stops a report after this (RQ job_timeout); a normal one takes ~60 s
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


def write_report(model, upload, llm_factory, cache, tracer):
    """The report for one upload: shared by the synchronous API and the worker, so both behave the same."""
    try:
        return generate_report_for_image(model, upload.image(), prediction=upload.prediction,
                                         llm=llm_factory(), cache=cache, tracer=tracer)
    finally:
        finish_trace(tracer)  # spans of a report that raised are still kept


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

    def __init__(self, client, prefix="cdp:", is_async=True, job_timeout_s=JOB_TIMEOUT_S,
                 queued_ttl_s=QUEUED_TTL_S, result_ttl_s=RESULT_TTL_S, failure_ttl_s=FAILURE_TTL_S):
        from rq import Queue
        from rq.serializers import JSONSerializer
        self.client, self.prefix = client, prefix
        self.queue = Queue(QUEUE_NAME, connection=client, serializer=JSONSerializer, is_async=is_async)
        self.job_timeout_s, self.queued_ttl_s = job_timeout_s, queued_ttl_s
        self.result_ttl_s, self.failure_ttl_s = result_ttl_s, failure_ttl_s
        # Longest a job can stay active: queued, then running until the timeout or until it counts as abandoned
        self.active_ttl_s = queued_ttl_s + job_timeout_s + 120
        self.spans_ttl_s = max(result_ttl_s, failure_ttl_s)  # as long as anyone can poll the job

    def active_key(self, user_id, trace_id):
        return f"{self.prefix}report-job:active:{user_id}:{trace_id}"

    def spans_key(self, job_id):
        return f"{self.prefix}report-job:spans:{job_id}"

    def span_buffer(self, job_id):
        return RedisSpanBuffer(self.client, self.spans_key(job_id), self.spans_ttl_s)

    def submit(self, trace_id, user_id):
        """(job_id, status). The queued or running job for this upload if there is one, else a new job.

        A failed, expired or finished job is not reused: asking again queues a new one (a finished LLM
        report then comes back from the report cache). The active key and the new job are written in one
        MULTI under WATCH, so two POSTs at once get the same job, never two.
        """
        import redis
        key = self.active_key(user_id, trace_id)
        with self._errors():
            for _ in range(SUBMIT_ATTEMPTS):
                current = self.client.get(key)
                if current is not None:
                    job, status = self._job_and_status(current.decode())
                    if status in ACTIVE:
                        return job.id, status
                job = self.queue.create_job(
                    JOB_FUNC, args=(trace_id, user_id), job_id=str(uuid.uuid4()),
                    timeout=self.job_timeout_s, ttl=self.queued_ttl_s,
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
            return f"Writing the report took longer than {self.job_timeout_s} s. Please try again."
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

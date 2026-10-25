"""Report worker: takes report jobs from Redis (RQ) and writes them, so API replicas never block on the LLM.

Run it next to the API with the same REDIS_URL (Linux or Docker only: RQ workers need fork/SIGALRM,
which Windows doesn't have):

    REDIS_URL=redis://redis:6379/0 GROQ_API_KEY=... python -m worker.report_worker

Each process loads the model once at startup and runs one job at a time; start more processes (or
containers) for more parallel reports. Groq's per-model limits are shared by every worker and API replica
through the Redis rate limiter, so more workers mean more reports waiting, not more tokens per minute.

The worker needs no Supabase settings: it never writes to Supabase (see api/jobs.py for why).
"""
import logging
import signal
import sys
from functools import partial

from rq import SimpleWorker, get_current_job

from agent.agent import RedisReportCache, build_llm_client, report_result_to_json
from agent.rate_limit import BudgetRegistry
from api.jobs import UPLOAD_EXPIRED, ReportJobs, write_report
from api.settings import Settings
from api.state import RedisUploadStore, SerializedModel, connect_redis, redis_location
from services.tracing import make_tracer

log = logging.getLogger("worker.report_worker")


class UploadExpired(Exception):
    """The upload left Redis (1 h TTL, or evicted) before the job ran."""


class WorkerContext:
    """What a job needs, built once per process: the model and the shared Redis stores."""

    def __init__(self, model, uploads, cache, llm_factory, jobs):
        self.model = model                # SerializedModel
        self.uploads = uploads            # RedisUploadStore (the API's uploads)
        self.cache = cache                # RedisReportCache (shared with the API)
        self.llm_factory = llm_factory    # build_llm_client with the Redis rate-limit budgets
        self.jobs = jobs                  # ReportJobs: where the spans go


_context = None


def configure(context):
    """Set the context jobs run with (main() at startup; tests with fakes). Returns the previous one."""
    global _context
    previous, _context = _context, context
    return previous


def build_context(settings, client):
    from services.model_handler import ModelService
    model = SerializedModel(ModelService(settings.model_path))  # ~16 MB checkpoint, once per process
    cache = RedisReportCache(client, on_error=lambda op, e: log.warning(
        "Report cache %s failed (%s); treated as a miss", op, type(e).__name__))
    return WorkerContext(
        model=model,
        uploads=RedisUploadStore(client, settings.upload_ttl_s, settings.upload_max_items),
        cache=cache,
        llm_factory=partial(build_llm_client, budgets=BudgetRegistry(client)),
        jobs=ReportJobs(client))


def run_report_job(trace_id, user_id):
    """The job. Same report path as the synchronous API: report cache, guard, retry, template fallback.

    Returns report_result_to_json(result), which RQ keeps for RESULT_TTL_S. Spans go to a Redis list
    (and the local traces/ JSONL); the API writes them to Supabase as the user when the job is polled.
    """
    if _context is None:
        raise RuntimeError("Report worker not configured (configure() was never called)")
    job = get_current_job()
    upload = _context.uploads.get(trace_id, user_id)
    if upload is None:
        job.meta["error"] = UPLOAD_EXPIRED
        job.save_meta()
        raise UploadExpired(trace_id)
    tracer = make_tracer(_context.jobs.span_buffer(job.id), trace_id=trace_id)  # same trace as the prediction
    result = write_report(_context.model, upload, _context.llm_factory, _context.cache, tracer)
    return report_result_to_json(result)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
    if not hasattr(signal, "SIGALRM"):
        sys.exit("RQ workers need Linux or macOS (job timeouts use SIGALRM). Run the worker in Docker.")
    settings = Settings.from_env()
    if not settings.redis_url:
        sys.exit("REDIS_URL is not set. Without it the API writes reports itself and needs no worker.")
    client = connect_redis(settings.redis_url)
    client.ping()
    configure(build_context(settings, client))
    log.info("Report worker ready: model loaded, Redis at %s, LLM %s", redis_location(client),
             "configured" if _context.llm_factory() is not None else "not configured (template reports)")
    # Its own connection: RQ raises this one's socket timeout for its blocking dequeue
    queue_client = connect_redis(settings.redis_url)
    jobs = ReportJobs(queue_client)
    # SimpleWorker runs jobs in this process: the model stays loaded, and PyTorch never runs in a forked
    # child. The job timeout is still enforced (SIGALRM); a crashed process leaves the job "started" until
    # the API (or another worker) cleans the started registry, which marks it failed.
    SimpleWorker([jobs.queue], connection=queue_client, serializer=jobs.queue.serializer).work()


if __name__ == "__main__":
    # `python -m` runs this file as __main__, but RQ imports worker.report_worker to call the job: a
    # second module with its own _context. Run main() from that module, so configure() sets the one jobs read.
    from worker.report_worker import main as module_main
    module_main()

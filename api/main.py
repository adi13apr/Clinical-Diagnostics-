"""FastAPI backend for the React UI.

Reuses services/ (model, calibration, tracing, Supabase) and agent/ (report, guard, fallback, cache);
the Streamlit app is unchanged. Run locally with:

    uvicorn api.main:create_app --factory --port 8000

Every endpoint except /health needs `Authorization: Bearer <Supabase access token>`. Database calls
send that token, so Supabase row-level security only ever returns the caller's own rows.

Reports: without REDIS_URL, POST /report writes the report during the request (about 60 s, in its own
thread pool). With REDIS_URL, it queues a job for worker/report_worker.py and answers 202 at once; the
client polls GET /report/jobs/{job_id} (see api/jobs.py).
"""
import asyncio
import io
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from datetime import date, datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agent.agent import REPORT_CACHE, RedisReportCache, build_llm_client
from agent.rate_limit import BUDGETS, BudgetRegistry
from api.auth import AuthError, AuthUnavailable, SupabaseJWTVerifier, user_db_factory
from api.jobs import ReportJobs, write_report
from api.schemas import (
    AnalyticsResponse, DailyVolume, GradcamMapResponse, HealthResponse, HistoryItem, HistoryPage, HistoryQuery,
    PipelineSummary, PredictResponse, ReportJobResponse, ReportResponse, SpanOut, TraceResponse,
)
from api.settings import LOCALHOST_ORIGIN_REGEX, Settings
from api.state import (
    RedisUploadStore, SerializedModel, StoreUnavailable, UploadStore, connect_redis, redis_location,
)
from services.calibration import REFER_MESSAGE, REFER_THRESHOLD, STATUS_REFER
from services import image_validation
from services.image_validation import ImageRejected, too_large
from services.prediction_handler import PredictionService
from services.tracing import SupabaseTraceStore, finish_trace, make_tracer, summarize_spans, trace_span

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
REJECTED_STATUS = {image_validation.NOT_AN_IMAGE: 415, image_validation.TOO_LARGE: 413, image_validation.EMPTY: 400}
MULTIPART_OVERHEAD = 64 * 1024      # Content-Length allowance for multipart headers and boundaries
REPORT_WORKERS = 4                  # report threads mostly wait on the LLM; kept off the default pool
TRACE_READ_LIMIT = 5000             # spans read for /analytics

bearer = HTTPBearer(auto_error=False, description="Supabase access token of the signed-in user")
log = logging.getLogger(__name__)


def configure_logging():
    """Show the API's own INFO/WARNING/ERROR lines under uvicorn, which only configures its own loggers.

    Left alone when something (pytest, a host) already configured the root logger.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s:     %(name)s: %(message)s")
        logging.getLogger("api").setLevel(logging.INFO)


def create_app(settings=None, model_service=None, verifier=None, db_factory=None, llm_factory=None,
               report_cache=None, uploads=None, report_jobs=None):
    """Build the app. Tests inject fakes; by default everything comes from Settings.from_env().

    Uploads, the report cache and the LLM rate-limit budgets live in Redis when settings.redis_url is
    set (shared by every replica), else in this process's memory (REPORT_CACHE and BUDGETS are the ones
    the Streamlit app uses too). With Redis, reports are also queued for worker processes (report_jobs);
    without it, the API writes them itself.
    """
    configure_logging()
    settings = settings or Settings.from_env()
    uploads, report_cache, budgets, redis_client = shared_state(settings, uploads, report_cache)
    if report_jobs is None and redis_client is not None:
        report_jobs = ReportJobs(redis_client)

    @asynccontextmanager
    async def lifespan(app):
        if app.state.model is None:  # ~16 MB checkpoint, loaded once per process
            from services.model_handler import ModelService
            service = await asyncio.to_thread(ModelService, settings.model_path)
            app.state.model = SerializedModel(service)
        verifier = app.state.verifier
        if verifier is None:
            log.warning("Auth is not configured: set SUPABASE_URL and SUPABASE_KEY. Every request but /health gets 503.")
        else:
            log.info("Verifying Supabase tokens issued by %s (URL from %s)", getattr(verifier, "issuer", "?"),
                     settings.supabase_url_source or "settings")
            if settings.check_keys_on_startup and hasattr(verifier, "check_keys"):
                await asyncio.to_thread(verifier.check_keys)
        if redis_client is None:
            log.info("Uploads, the report cache and LLM rate limits are kept in this process's memory "
                     "(REDIS_URL not set)")
        else:
            await asyncio.to_thread(check_redis, redis_client)
        if report_jobs is None:
            log.info("Reports are written by this API process (no REDIS_URL, so no report worker)")
        else:
            workers = await asyncio.to_thread(report_jobs.worker_count)
            log.log(logging.WARNING if not workers else logging.INFO,
                    "Reports are queued for report workers; %s listening now%s", workers,
                    "" if workers else ". Start one: python -m worker.report_worker (jobs wait up to 10 min)")
        yield
        app.state.report_pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Clinical Diagnostic Portal API", version="1.0", lifespan=lifespan,
                  description="Chest X-ray screening aid (Normal / Pneumonia / Tuberculosis). Not a diagnosis.")
    app.state.settings = settings
    app.state.model = SerializedModel(model_service) if model_service is not None else None
    app.state.verifier = verifier or (
        SupabaseJWTVerifier(settings.supabase_url, settings.supabase_key, settings.jwt_secret)
        if settings.supabase_url else None)
    app.state.db_factory = db_factory or (
        user_db_factory(settings.supabase_url, settings.supabase_key) if settings.supabase_url else None)
    app.state.llm_factory = llm_factory or partial(build_llm_client, budgets=budgets)
    app.state.report_cache = report_cache
    app.state.uploads = uploads
    app.state.report_jobs = report_jobs
    app.state.report_pool = ThreadPoolExecutor(REPORT_WORKERS, thread_name_prefix="report")

    @app.middleware("http")
    async def reject_large_uploads(request, call_next):
        # Early 413 from the header, before the body is read; read_upload() checks the real size
        if request.url.path == "/predict":
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > settings.max_upload_bytes + MULTIPART_OVERHEAD:
                return JSONResponse({"detail": too_large(settings.max_upload_bytes)}, status_code=413)
        return await call_next(request)

    if settings.served_by:
        @app.middleware("http")
        async def served_by(request, call_next):
            response = await call_next(request)
            response.headers["X-Served-By"] = settings.served_by
            return response

    # Added last so it's outermost: every response, including the early 413, gets CORS headers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_urls,
        allow_origin_regex=LOCALHOST_ORIGIN_REGEX,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ---- Dependencies ----

    async def current_user(request: Request,
                           credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(401, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
        if request.app.state.verifier is None:
            raise HTTPException(503, "Auth is not configured (set SUPABASE_URL and SUPABASE_KEY)")
        try:
            # May fetch signing keys or ask Supabase, so off the event loop
            return await asyncio.to_thread(request.app.state.verifier.verify, credentials.credentials)
        except AuthError as e:
            raise HTTPException(401, str(e), headers={"WWW-Authenticate": "Bearer"})
        except AuthUnavailable as e:
            raise HTTPException(503, str(e))

    def user_db(request: Request, user=Depends(current_user)):
        """Database client acting as this user (sync dependency: FastAPI runs it in a thread)."""
        client = request.app.state.db_factory(user.token)
        try:
            yield client
        finally:
            close = getattr(client, "aclose", None)  # postgrest's sync client names its close aclose()
            if close is not None:
                close()

    def model(request: Request):
        if request.app.state.model is None:
            raise HTTPException(503, "Model not loaded")
        return request.app.state.model

    User = Annotated[object, Depends(current_user)]
    Db = Annotated[object, Depends(user_db)]
    Model = Annotated[object, Depends(model)]

    async def find_upload(request, trace_id, user):
        upload = await use_store(request.app.state.uploads.get, str(trace_id), user.id)
        if upload is None:
            # Same answer for "never existed", "expired" and "someone else's"
            raise HTTPException(404, "Upload not found or expired; run /predict again")
        return upload

    # ---- Endpoints ----

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        state = request.app.state
        if not hasattr(state, "llm_configured"):
            state.llm_configured = state.llm_factory() is not None  # keys don't change while running
        loaded = state.model is not None
        jobs = state.report_jobs
        return HealthResponse(
            status="ok", model_loaded=loaded,
            calibrated=bool(loaded and state.model.calibrated),
            temperature=state.model.temperature if loaded else None,
            auth_configured=state.verifier is not None, llm_configured=state.llm_configured,
            report_mode="inline" if jobs is None else "queue",
            report_workers=None if jobs is None else await asyncio.to_thread(jobs.worker_count))

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, user: User, db: Db, model: Model,
                      file: Annotated[UploadFile, File(description="Chest X-ray, JPEG or PNG")]):
        data = await read_upload(file, settings.max_upload_bytes)
        image = await asyncio.to_thread(decode_image, data)
        trace_id = str(uuid.uuid4())
        tracer = make_tracer(SupabaseTraceStore(db), trace_id=trace_id)
        result, inference_ms, warnings = await asyncio.to_thread(
            predict_and_log, model, PredictionService(db), image, tracer, trace_id)
        await use_store(request.app.state.uploads.put, trace_id, user.id, data, result)
        return PredictResponse(
            trace_id=trace_id, predicted_class=result["diagnosis"], confidence=result["confidence"],
            probabilities=result["probabilities"], status=result["status"],
            referral_threshold=REFER_THRESHOLD,
            referral_message=REFER_MESSAGE if result["status"] == STATUS_REFER else None,
            temperature=result["temperature"], calibrated=model.calibrated,
            inference_ms=inference_ms, warnings=warnings)

    @app.get("/gradcam/{trace_id}", response_class=Response,
             responses={200: {"content": {"image/png": {}}, "description": "Grad-CAM figure (PNG)"}})
    async def gradcam(request: Request, trace_id: uuid.UUID, user: User, db: Db, model: Model):
        upload = await find_upload(request, trace_id, user)
        if upload.heatmap_png is None:
            tracer = make_tracer(SupabaseTraceStore(db), trace_id=str(trace_id))
            upload.heatmap_png = await asyncio.to_thread(render_heatmap, model, upload, tracer)
            await use_store(request.app.state.uploads.save_heatmap, str(trace_id), upload.heatmap_png)
        return Response(upload.heatmap_png, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/gradcam/{trace_id}/map", response_model=GradcamMapResponse)
    async def gradcam_map(request: Request, trace_id: uuid.UUID, user: User, db: Db, model: Model):
        """The raw map, so a client can draw its own overlay (e.g. with an opacity slider)."""
        upload = await find_upload(request, trace_id, user)
        if upload.cam_map is None:
            tracer = make_tracer(SupabaseTraceStore(db), trace_id=str(trace_id))
            upload.cam_map = await asyncio.to_thread(compute_cam_map, model, upload, tracer)
            await use_store(request.app.state.uploads.save_cam_map, str(trace_id), upload.cam_map)
        cam, regions = upload.cam_map
        return GradcamMapResponse(trace_id=str(trace_id), explained_class=upload.prediction["diagnosis"],
                                  cam=cam, regions=regions)

    @app.post("/report/{trace_id}", response_model=ReportJobResponse, status_code=202, responses={
        200: {"model": ReportJobResponse, "description": "Without REDIS_URL: written during the request, status 'done'"}})
    async def report(request: Request, response: Response, trace_id: uuid.UUID, user: User, db: Db, model: Model):
        """Start a report. With REDIS_URL: 202 and a job to poll; asking again while that job is queued or
        running returns the same job. Without: the finished report (200)."""
        upload = await find_upload(request, trace_id, user)
        state = request.app.state
        if state.report_jobs is not None:
            job_id, status = await use_store(state.report_jobs.submit, str(trace_id), user.id)
            return ReportJobResponse(job_id=job_id, trace_id=str(trace_id), status=status)
        tracer = make_tracer(SupabaseTraceStore(db), trace_id=str(trace_id))  # same trace as the prediction
        loop = asyncio.get_running_loop()
        # Up to ~60 s, mostly waiting on the LLM's rate limit: its own pool, so it can't starve predictions
        result = await loop.run_in_executor(
            state.report_pool, write_report, model, upload, state.llm_factory, state.report_cache, tracer)
        response.status_code = 200
        return ReportJobResponse(job_id=None, trace_id=str(trace_id), status="done",
                                 result=report_response(str(trace_id), result), warnings=trace_warnings(tracer))

    @app.get("/report/jobs/{job_id}", response_model=ReportJobResponse)
    async def report_job(request: Request, job_id: uuid.UUID, user: User, db: Db):
        """Status of a report job; the report once it's done. 404 for unknown, expired or other users' jobs."""
        jobs = request.app.state.report_jobs
        view = await use_store(jobs.get, str(job_id), user.id) if jobs is not None else None
        if view is None:
            raise HTTPException(404, "Report job not found or expired; ask for the report again")
        warnings = []
        if view.status in ("done", "failed"):
            # Only now, and only for the owner: the worker's spans go to Supabase with this user's token
            warnings = await use_store(jobs.flush_spans, view.job_id, SupabaseTraceStore(db))
        return ReportJobResponse(
            job_id=view.job_id, trace_id=view.trace_id, status=view.status, error=view.error, warnings=warnings,
            result=report_response(view.trace_id, view.result) if view.result is not None else None)

    @app.get("/history", response_model=HistoryPage)
    async def history(user: User, db: Db, query: Annotated[HistoryQuery, Query()]):
        since = day_start(query.date_from) if query.date_from else None
        until = day_start(query.date_to + timedelta(days=1)) if query.date_to else None
        res = await asyncio.to_thread(
            PredictionService(db).get_history, user.id, limit=query.limit, offset=query.offset,
            label=query.label, status=query.status, since=since, until=until)
        items = [HistoryItem(**{k: row.get(k) for k in HistoryItem.model_fields}) for row in res.data]
        return HistoryPage(items=items, total=res.count if res.count is not None else len(items),
                           limit=query.limit, offset=query.offset)

    @app.get("/analytics", response_model=AnalyticsResponse)
    async def analytics(user: User, db: Db, days: Annotated[int, Query(ge=1, le=365)] = 30):
        summary, spans = await asyncio.gather(
            asyncio.to_thread(lambda: PredictionService(db).get_daily_summary().data),
            asyncio.to_thread(SupabaseTraceStore(db).read, TRACE_READ_LIMIT, user.id),
            return_exceptions=True)
        warnings = []
        if isinstance(summary, Exception):
            warnings.append(f"Could not load the prediction summary ({type(summary).__name__})")
            summary = []
        if isinstance(spans, Exception):
            # No fallback to local traces/: that file holds every user's spans
            warnings.append(f"Could not load traces ({type(spans).__name__})")
            spans = None
        daily = daily_volume(summary, days)
        total = sum(d.predictions for d in daily)
        referred = sum(d.referred for d in daily)
        pipeline = None
        if spans is not None:
            pipeline = PipelineSummary(spans=len(spans), **summarize_spans(spans))
        return AnalyticsResponse(days=days, total_predictions=total, total_referred=referred,
                                 referral_rate=round(referred / total, 4) if total else None,
                                 daily=daily, pipeline=pipeline, warnings=warnings)

    @app.get("/traces/{trace_id}", response_model=TraceResponse)
    async def trace(trace_id: uuid.UUID, user: User, db: Db):
        rows = await asyncio.to_thread(SupabaseTraceStore(db).read_trace, str(trace_id), user.id)
        if not rows:
            raise HTTPException(404, "Trace not found")
        spans = [SpanOut(**{k: row.get(k) for k in SpanOut.model_fields if k != "attributes"},
                         attributes=row.get("attributes") or {}) for row in rows]
        return TraceResponse(trace_id=str(trace_id), spans=spans)

    return app


# ---- Helpers (plain functions; the blocking ones run in worker threads) ----

def shared_state(settings, uploads=None, report_cache=None):
    """(uploads, report_cache, budgets, redis client or None). Injected stores win; else Redis if REDIS_URL is set.

    budgets is the BudgetRegistry for LLM clients: in Redis, every replica shares each model's
    requests/tokens per minute and daily caps instead of each assuming it has the whole quota.
    """
    client = connect_redis(settings.redis_url) if settings.redis_url else None
    if uploads is None:
        uploads = (RedisUploadStore(client, settings.upload_ttl_s, settings.upload_max_items) if client
                   else UploadStore(settings.upload_ttl_s, settings.upload_max_items))
    if report_cache is None:
        report_cache = RedisReportCache(client, on_error=log_cache_error) if client else REPORT_CACHE
    budgets = BudgetRegistry(client) if client else BUDGETS
    return uploads, report_cache, budgets, client


def check_redis(client):
    """One PING at startup. A failure is logged, not fatal: requests get 503 until Redis is back."""
    try:
        client.ping()
        log.info("Uploads, the report cache and LLM rate limits are shared through Redis at %s",
                 redis_location(client))
        return True
    except Exception as e:
        log.error("Redis at %s is unreachable (%s: %s); uploads will answer 503 until it is back",
                  redis_location(client), type(e).__name__, e)
        return False


def log_cache_error(operation, error):
    log.warning("Report cache %s failed (%s); treated as a miss", operation, type(error).__name__)


async def use_store(method, *args):
    """Call an upload store or report job method off the event loop (Redis is network I/O); 503 if Redis is down."""
    try:
        return await asyncio.to_thread(method, *args)
    except StoreUnavailable as e:
        raise HTTPException(503, str(e))


async def read_upload(file, max_bytes):
    """The upload's bytes, after the type and size checks (415 / 413)."""
    if (file.content_type or "").lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, "Only JPEG and PNG images are accepted")
    data = await file.read(max_bytes + 1)
    try:
        image_validation.check_size(data, max_bytes)
    except ImageRejected as e:
        raise HTTPException(REJECTED_STATUS[e.kind], str(e))
    return data


def decode_image(data):
    """Pillow image from validated bytes; 415 if it isn't really a JPEG/PNG, 413 if absurdly large."""
    try:
        return image_validation.decode_image(data)
    except ImageRejected as e:
        raise HTTPException(REJECTED_STATUS[e.kind], str(e))


def predict_and_log(model, prediction_service, image, tracer, trace_id):
    """Same spans as the dashboard's run_prediction: prediction -> preprocess/classify/calibrate, log_prediction."""
    warnings = []
    with trace_span(tracer, "prediction"):
        start = time.perf_counter()
        result = model.predict(image, tracer=tracer)
        inference_ms = round((time.perf_counter() - start) * 1000)
        try:
            with trace_span(tracer, "log_prediction"):
                prediction_service.log_prediction(result, inference_ms, trace_id=trace_id)
        except Exception as e:
            warnings.append(f"Could not log prediction ({type(e).__name__})")
    finish_trace(tracer)
    return result, inference_ms, warnings + trace_warnings(tracer)


def render_heatmap(model, upload, tracer):
    figure = model.generate_heatmap(upload.image(), upload.prediction, tracer=tracer)
    finish_trace(tracer)
    buf = io.BytesIO()
    figure.save(buf, format="PNG")
    return buf.getvalue()


def compute_cam_map(model, upload, tracer):
    from services.heatmap_handler import summarize_cam_regions
    cam = model.compute_gradcam(upload.image(), upload.prediction["diagnosis"], tracer=tracer)
    finish_trace(tracer)
    return [[round(float(v), 4) for v in row] for row in cam], summarize_cam_regions(cam)


def report_response(trace_id, result):
    return ReportResponse(trace_id=trace_id, report=result.report, source=result.source, model=result.model,
                          source_line=result.source_line(), reason=result.reason, cached=result.cached)


def trace_warnings(tracer):
    if tracer is not None and tracer.sink_errors:
        return ["Trace not fully saved; it may be missing from /traces"]
    return []


def day_start(day):
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat()


def daily_volume(summary_rows, days, today=None):
    """prediction_daily_summary rows (one per day and label) -> one DailyVolume per day, newest first."""
    first = (today or datetime.now(timezone.utc).date()) - timedelta(days=days - 1)
    by_day = {}
    for row in summary_rows:
        day = str(row["day"])[:10]
        if date.fromisoformat(day) < first:
            continue
        d = by_day.setdefault(day, {"predictions": 0, "referred": 0, "by_label": {}, "ms": 0.0, "timed": 0})
        count = int(row.get("prediction_count") or 0)
        d["predictions"] += count
        d["referred"] += int(row.get("refer_count") or 0)
        d["by_label"][row["predicted_label"]] = d["by_label"].get(row["predicted_label"], 0) + count
        if row.get("avg_inference_ms") is not None:
            d["ms"] += float(row["avg_inference_ms"]) * count
            d["timed"] += count
    return [DailyVolume(day=day, predictions=d["predictions"], referred=d["referred"],
                        referral_rate=round(d["referred"] / d["predictions"], 4) if d["predictions"] else 0.0,
                        by_label=d["by_label"],
                        avg_inference_ms=round(d["ms"] / d["timed"], 1) if d["timed"] else None)
            for day, d in sorted(by_day.items(), reverse=True)]

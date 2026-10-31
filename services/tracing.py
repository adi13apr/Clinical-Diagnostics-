"""Step-level tracing: one trace per prediction, one span per pipeline step.

App code should only use the fail-safe helpers (trace_span, trace_event, finish_trace).
They swallow and log any tracing error, so tracing can never break a prediction. The
step's own exceptions still propagate.
"""
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO / "traces"
PRICING_PATH = REPO / "config" / "pricing.json"

# Spans that mark a moment rather than time a step (excluded from latency averages)
EVENT_SPANS = {"guard", "fallback_used", "report_cache_hit",
               # multi-agent pipeline (agent/multi_agent.py)
               "final_guard", "handoff", "orchestrator_state", "evidence_reask", "revision_requested"}


@dataclass
class Span:
    trace_id: str
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    start: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: Optional[float] = None
    status: str = "ok"
    error_type: Optional[str] = None
    error: Optional[str] = None
    attributes: dict = field(default_factory=dict)

    def set(self, **attributes):
        self.attributes.update(attributes)

    def fail(self, error, error_type="Error"):
        """Mark the span as failed without raising (e.g. a tool that returned an error)."""
        self.status, self.error, self.error_type = "error", str(error)[:500], error_type

    def to_dict(self):
        return asdict(self)


class _NullSpan:
    """Stand-in when tracing is off or broken; accepts everything and records nothing."""

    def set(self, **attributes):
        pass

    def fail(self, error, error_type="Error"):
        pass


NULL_SPAN = _NullSpan()


class Tracer:
    """Collects spans for one trace and writes them to its sinks on finish()."""

    def __init__(self, trace_id=None, sinks=()):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.sinks = list(sinks)
        self.spans = []
        self.sink_errors = []
        self._stack = []
        self._flushed = 0

    @contextmanager
    def span(self, name, **attributes):
        span = Span(trace_id=self.trace_id, name=name, attributes=dict(attributes),
                    parent_span_id=self._stack[-1].span_id if self._stack else None)
        self.spans.append(span)
        self._stack.append(span)
        start = time.perf_counter()
        try:
            yield span
        except BaseException as e:
            span.fail(e, type(e).__name__)
            raise
        finally:
            span.latency_ms = round((time.perf_counter() - start) * 1000, 3)
            self._stack.pop()

    def event(self, name, status="ok", error=None, **attributes):
        """Zero-duration span, e.g. a guard verdict."""
        span = Span(trace_id=self.trace_id, name=name, latency_ms=0.0, attributes=dict(attributes),
                    parent_span_id=self._stack[-1].span_id if self._stack else None)
        if status != "ok":
            span.fail(error or status, "Rejected" if status == "reject" else "Error")
        self.spans.append(span)
        return span

    def finish(self):
        """Write spans recorded since the last finish() to every sink. Never raises."""
        new = [s.to_dict() for s in self.spans[self._flushed:]]
        self._flushed = len(self.spans)
        if not new:
            return
        for sink in self.sinks:
            try:
                sink.write(new)
            except Exception as e:
                self.sink_errors.append(f"{type(sink).__name__}: {type(e).__name__}: {e}")
                logger.warning("Trace sink %s failed: %s", type(sink).__name__, e)


# ---- Fail-safe helpers (the only API app code should call) ----

@contextmanager
def trace_span(tracer, name, **attributes):
    """Span around a block. Tracing errors are logged and ignored; the block's own errors propagate."""
    cm = None
    span = NULL_SPAN
    if tracer is not None:
        try:
            cm = tracer.span(name, **attributes)
            span = cm.__enter__()
        except Exception as e:
            logger.warning("Tracing failed to open span %r: %s", name, e)
            cm, span = None, NULL_SPAN
    try:
        yield span
    except BaseException as e:
        if cm is not None:
            try:
                cm.__exit__(type(e), e, e.__traceback__)
            except BaseException as exit_error:
                if exit_error is not e:
                    logger.warning("Tracing failed to close span %r: %s", name, exit_error)
        raise
    else:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception as e:
                logger.warning("Tracing failed to close span %r: %s", name, e)


def trace_event(tracer, name, **kwargs):
    if tracer is None:
        return NULL_SPAN
    try:
        return tracer.event(name, **kwargs)
    except Exception as e:
        logger.warning("Tracing failed to record event %r: %s", name, e)
        return NULL_SPAN


def finish_trace(tracer):
    if tracer is None:
        return
    try:
        tracer.finish()
    except Exception as e:
        logger.warning("Tracing failed to finish trace: %s", e)


def safe_set(span, **attributes):
    """Set attributes, ignoring tracing errors."""
    try:
        span.set(**attributes)
    except Exception as e:
        logger.warning("Tracing failed to set attributes: %s", e)


# ---- Sinks ----

_jsonl_lock = threading.Lock()


class JsonlTraceSink:
    """One JSON line per span in traces/YYYY-MM-DD.jsonl (gitignored)."""

    def __init__(self, directory=TRACE_DIR):
        self.directory = Path(directory)

    def write(self, spans):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        lines = "".join(json.dumps(s, default=str) + "\n" for s in spans)
        with _jsonl_lock, open(path, "a", encoding="utf-8") as f:
            f.write(lines)

    def read(self, limit=5000):
        rows = []
        for path in sorted(self.directory.glob("*.jsonl"), reverse=True):
            with open(path, encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
            if len(rows) >= limit:
                break
        return rows[:limit]


class SupabaseTraceStore:
    """Supabase `traces` table (sql/004_traces.sql). Needs the logged-in auth client for RLS."""

    def __init__(self, client):
        self.client = client

    def write(self, spans):
        """Insert spans, skipping any already stored (same trace_id and span_id), so a retry is harmless.

        ON CONFLICT DO NOTHING needs the unique index from sql/006_traces_unique_span.sql.
        """
        rows = [{
            "trace_id": s["trace_id"], "span_id": s["span_id"], "parent_span_id": s["parent_span_id"],
            "name": s["name"], "started_at": s["start"], "latency_ms": s["latency_ms"], "status": s["status"],
            "error_type": s["error_type"], "error": s["error"],
            "attributes": json.loads(json.dumps(s["attributes"], default=str)),
        } for s in spans]
        self.client.table("traces").upsert(rows, on_conflict="trace_id,span_id", ignore_duplicates=True).execute()

    def read(self, limit=5000, user_id=None):
        query = self.client.table("traces").select("*")
        if user_id is not None:
            query = query.eq("user_id", user_id)  # RLS already does this; kept as a second check
        res = query.order("started_at", desc=True).limit(limit).execute()
        return [dict(r, start=r.get("started_at")) for r in res.data]

    def read_trace(self, trace_id, user_id=None):
        """All spans of one trace, oldest first."""
        query = self.client.table("traces").select("*").eq("trace_id", trace_id)
        if user_id is not None:
            query = query.eq("user_id", user_id)
        res = query.order("started_at").execute()
        return [dict(r, start=r.get("started_at")) for r in res.data]


def make_tracer(trace_store=None, trace_id=None, trace_dir=TRACE_DIR):
    """Tracer writing to local JSONL and, if given, the Supabase trace store."""
    try:
        sinks = [JsonlTraceSink(trace_dir)] + ([trace_store] if trace_store is not None else [])
        return Tracer(trace_id=trace_id, sinks=sinks)
    except Exception as e:
        logger.warning("Tracing disabled: %s", e)
        return None


# ---- Cost (nominal) ----

_pricing_cache = {}


def load_pricing(path=PRICING_PATH):
    path = Path(path)
    if path not in _pricing_cache:
        with open(path, encoding="utf-8") as f:
            _pricing_cache[path] = json.load(f)
    return _pricing_cache[path]


def estimate_cost(model, input_tokens, output_tokens, on=None, pricing=None):
    """{'nominal_cost_usd', 'billed_cost_usd'}; nominal uses the list price, billed is 0 on the free tier.

    output_tokens should include thinking tokens (Gemini bills them as output). None if the model is unpriced.
    """
    pricing = pricing or load_pricing()
    entry = pricing["models"].get(model)
    if entry is None or input_tokens is None or output_tokens is None:
        return {"nominal_cost_usd": None, "billed_cost_usd": None}
    day = (on or date.today()).isoformat()
    rate = next(r for r in entry["schedule"] if r["until"] is None or day <= r["until"])
    nominal = input_tokens / 1e6 * rate["input_per_million"] + output_tokens / 1e6 * rate["output_per_million"]
    billed = 0.0 if pricing.get("billing_tier") == "free" else nominal
    return {"nominal_cost_usd": round(nominal, 8), "billed_cost_usd": round(billed, 8)}


# ---- Analytics ----

def summarize_spans(rows):
    """Metrics for the analytics page from span dicts (JSONL or Supabase rows)."""
    timed = [r for r in rows if r["name"] not in EVENT_SPANS and r.get("latency_ms") is not None]
    by_step = {}
    for r in timed:
        s = by_step.setdefault(r["name"], {"count": 0, "total_ms": 0.0, "errors": 0})
        s["count"] += 1
        s["total_ms"] += r["latency_ms"]
        s["errors"] += r["status"] == "error"
    steps = {name: {"count": s["count"], "avg_latency_ms": round(s["total_ms"] / s["count"], 1),
                    "error_rate": round(s["errors"] / s["count"], 3)} for name, s in by_step.items()}

    # Tokens per report: sum the llm_call spans under each report span that made LLM calls. In the
    # multi-agent pipeline they sit one level deeper (report -> writer_agent -> llm_call), so walk up
    # to the enclosing report span; its parent is the call's own parent when there is none.
    by_id = {r.get("span_id"): r for r in rows}

    def report_of(row):
        parent, seen = row.get("parent_span_id"), set()
        while parent in by_id and parent not in seen:
            seen.add(parent)
            if by_id[parent]["name"] == "report":
                return parent
            parent = by_id[parent].get("parent_span_id")
        return row.get("parent_span_id")

    per_report = {}
    for r in rows:
        if r["name"] == "llm_call":
            a = r.get("attributes") or {}
            t = per_report.setdefault(report_of(r), {"input": 0, "output": 0, "cost": 0.0})
            t["input"] += a.get("input_tokens") or 0
            t["output"] += (a.get("output_tokens") or 0) + (a.get("thinking_tokens") or 0)
            t["cost"] += a.get("nominal_cost_usd") or 0.0
    n = len(per_report)

    guards = [r for r in rows if r["name"] == "guard"]
    fallbacks = [r for r in rows if r["name"] == "fallback_used"]
    rejects = sum((r.get("attributes") or {}).get("result") == "reject" for r in guards)
    used = sum(bool((r.get("attributes") or {}).get("used")) for r in fallbacks)
    return {
        "traces": len({r["trace_id"] for r in rows}),
        "steps": steps,
        "reports_with_llm": n,
        "avg_input_tokens_per_report": round(sum(t["input"] for t in per_report.values()) / n, 1) if n else None,
        "avg_output_tokens_per_report": round(sum(t["output"] for t in per_report.values()) / n, 1) if n else None,
        "avg_nominal_cost_per_report_usd": round(sum(t["cost"] for t in per_report.values()) / n, 6) if n else None,
        "guard_checks": len(guards),
        "guard_reject_rate": round(rejects / len(guards), 3) if guards else None,
        "reports": len(fallbacks),
        "fallback_rate": round(used / len(fallbacks), 3) if fallbacks else None,
    }

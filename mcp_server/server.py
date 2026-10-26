"""MCP server: the classifier, Grad-CAM, the guarded report agent, traces and eval results as MCP tools.

Runs in-process on local image files. It reads no Supabase key and writes nothing to Supabase; its spans
go to traces/mcp/ only. Start it with `python <repo>/mcp_server` (stdio). Tests build it with
create_server(...) and fakes, and talk to it through the SDK's in-memory client.

agent/ (and through it torch, cv2, matplotlib) is imported inside the tool bodies, not here: a client
waits for initialize before it can list tools, and those imports took ~8 s on Windows.
"""
import json
import logging
import os
import re
import threading
import time
import tomllib
import uuid
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import get_args

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.types import ToolAnnotations

from agent.llm_config import provider_settings
from agent.schema import ClassName
from mcp_server.paths import REPO, PathRejected, allowed_dirs_from_env, read_limited, resolve_image_path
from mcp_server.schemas import (
    CheckRate, Classification, EvalSummary, GradcamSummary, ReportOut, SpanOut, TraceOut,
)
from services.image_validation import DEFAULT_MAX_UPLOAD_MB, ImageRejected, check_size, decode_image
from services.tracing import finish_trace, make_tracer, trace_span

CLASS_NAMES = list(get_args(ClassName))

MCP_TRACE_DIR = REPO / "traces" / "mcp"   # this server's spans only; never the app's shared traces/
DEFAULT_MODEL_PATH = REPO / "best_student_model_aligned_final.pth"
SECRETS_PATH = REPO / ".streamlit" / "secrets.toml"
RECENT_PREDICTIONS = 100   # images whose latest prediction gradcam_summary/generate_report can reuse
RUN_ID = re.compile(r"^\d{8}T\d{6}Z$")
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

INSTRUCTIONS = """Chest X-ray screening aid (Normal / Pneumonia / Tuberculosis). Not a diagnosis.
Typical use: classify_xray, then gradcam_summary and/or generate_report on the same image_path.
Always pass on the status: 'refer' means clinician review is recommended. Quote probabilities as returned.
Image paths must be JPEG/PNG files inside the server's allowed directories."""

log = logging.getLogger(__name__)


class Clinic:
    """What the tools share: the model (loaded on first use), recent predictions by image, and settings."""

    def __init__(self, model_service=None, llm_factory=None, report_cache=None,
                 allowed_dirs=None, trace_dir=None, max_upload_bytes=None, model_path=DEFAULT_MODEL_PATH):
        self._model = None
        if model_service is not None:
            from api.state import SerializedModel
            self._model = SerializedModel(model_service)
        self._model_lock = threading.Lock()
        self.model_path = Path(model_path)
        self.llm_factory = llm_factory    # None: agent.agent.build_report_llm (REPORT_PIPELINE picks single/multi)
        self.report_cache = report_cache  # None: the process-wide agent.agent.REPORT_CACHE
        self.allowed_dirs = allowed_dirs_from_env() if allowed_dirs is None else [Path(d).resolve() for d in allowed_dirs]
        self.trace_dir = Path(trace_dir or os.environ.get("MCP_TRACE_DIR") or MCP_TRACE_DIR)
        self.max_upload_bytes = int(max_upload_bytes or float(os.environ.get("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))
                                    * 1024 * 1024)
        self._recent = OrderedDict()  # image fingerprint -> (trace_id, prediction); no pixels, no paths
        self._recent_lock = threading.Lock()

    def model(self):
        """The model behind one lock (Grad-CAM hooks the shared network). Loaded on the first tool call,
        so starting the server and listing tools needs no torch work."""
        with self._model_lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise ToolError(f"Model checkpoint not found at {self.model_path}")
                from api.state import SerializedModel
                from services.model_handler import ModelService
                # st.cache_resource outside the app warns on every call from a worker thread
                for name in ("streamlit", "streamlit.runtime.scriptrunner_utils.script_run_context"):
                    logging.getLogger(name).setLevel(logging.ERROR)
                self._model = SerializedModel(ModelService(str(self.model_path)))
            return self._model

    def load_image(self, image_path):
        try:
            path = resolve_image_path(image_path, self.allowed_dirs)
            data = read_limited(path, self.max_upload_bytes)
            check_size(data, self.max_upload_bytes)
            return decode_image(data)
        except (PathRejected, ImageRejected) as e:
            raise ToolError(str(e))
        except OSError as e:
            raise ToolError(f"Could not read the file ({type(e).__name__})")

    def tracer(self, trace_id):
        return make_tracer(None, trace_id=trace_id, trace_dir=self.trace_dir)  # local JSONL only

    def classify(self, image):
        """A new prediction (and trace) for the image, remembered for the image's later tools."""
        from agent.agent import image_fingerprint
        model = self.model()
        trace_id = str(uuid.uuid4())
        tracer = self.tracer(trace_id)
        with trace_span(tracer, "prediction"):
            start = time.perf_counter()
            prediction = model.predict(image, tracer=tracer)
            inference_ms = round((time.perf_counter() - start) * 1000)
        finish_trace(tracer)
        key = image_fingerprint(image)
        with self._recent_lock:
            self._recent[key] = (trace_id, prediction)
            self._recent.move_to_end(key)
            while len(self._recent) > RECENT_PREDICTIONS:
                self._recent.popitem(last=False)
        return trace_id, prediction, inference_ms

    def latest_prediction(self, image):
        """(trace_id, prediction) of this image's latest classify_xray, classifying it now if there's none."""
        from agent.agent import image_fingerprint
        with self._recent_lock:
            seen = self._recent.get(image_fingerprint(image))
        if seen is not None:
            return seen
        trace_id, prediction, _ = self.classify(image)
        return trace_id, prediction

    # ---- Tool bodies (blocking; run in a worker thread) ----

    def classify_xray(self, image_path):
        from agent.tools import ReportTools
        image = self.load_image(image_path)
        trace_id, prediction, inference_ms = self.classify(image)
        facts = ReportTools(self.model(), image, prediction).run_classifier()  # same rounding as the report agent
        return Classification(trace_id=trace_id, calibrated=self.model().calibrated,
                              temperature=prediction["temperature"], inference_ms=inference_ms, **facts)

    def gradcam_summary(self, image_path):
        from agent.tools import DECIMALS, GRADCAM_CAVEAT
        from services.heatmap_handler import summarize_cam_regions
        image = self.load_image(image_path)
        trace_id, prediction = self.latest_prediction(image)
        tracer = self.tracer(trace_id)
        cam = self.model().compute_gradcam(image, prediction["diagnosis"], tracer=tracer)
        finish_trace(tracer)
        regions = summarize_cam_regions(cam)
        return GradcamSummary(
            trace_id=trace_id, explained_class=prediction["diagnosis"], peak_region=regions["peak_region"],
            focus=regions["focus"], region_shares=regions["region_shares"], border_share=regions["border_share"],
            border_attention=regions.get("border_attention"),
            cam=[[round(float(v), DECIMALS) for v in row] for row in cam],
            caveat=GRADCAM_CAVEAT if regions["peak_region"] else f"{regions['note']} {GRADCAM_CAVEAT}")

    def generate_report(self, image_path):
        from agent.agent import REPORT_CACHE, build_report_llm, generate_report_for_image
        image = self.load_image(image_path)
        trace_id, prediction = self.latest_prediction(image)
        tracer = self.tracer(trace_id)  # same trace as the prediction, like the dashboard
        llm = (self.llm_factory or build_report_llm)()
        cache = REPORT_CACHE if self.report_cache is None else self.report_cache  # an empty cache is falsy
        try:
            result = generate_report_for_image(self.model(), image, prediction, llm=llm, cache=cache, tracer=tracer)
        finally:
            finish_trace(tracer)
        llm = result.source == "llm"
        return ReportOut(trace_id=trace_id, source="llm" if llm else "template", written_by=result.source_line(),
                         model=result.model if llm else None, fallback_reason=None if llm else result.reason,
                         cached=result.cached, guard_rejections=len(result.rejected),
                         tool_calls=list(result.tool_calls), pipeline=result.pipeline,
                         pipeline_note=result.pipeline_note, revisions=len(result.revisions), report=result.report)

    def get_trace(self, trace_id):
        try:
            trace_id = str(uuid.UUID(trace_id))
        except (ValueError, TypeError, AttributeError):
            raise ToolError("trace_id must be a UUID, as returned by classify_xray")
        spans = []
        for path in sorted(self.trace_dir.glob("*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if trace_id in line:  # cheap filter before parsing
                        span = json.loads(line)
                        if span.get("trace_id") == trace_id:
                            spans.append(span)
        if not spans:
            raise ToolError("No trace with that id was written by this MCP server")
        spans.sort(key=lambda s: s.get("start") or "")
        return TraceOut(trace_id=trace_id, spans=[SpanOut(**{k: s.get(k) for k in SpanOut.model_fields
                                                              if k != "attributes"},
                                                          attributes=s.get("attributes") or {}) for s in spans])

    def eval_summary(self, run_id):
        from evals import run as evals
        from evals.checks import pass_rates
        from evals.dataset import dataset_hash, load_cases
        from evals.schema import CRITERIA

        if run_id == "latest":
            path = evals.latest_run()
            if path is None:
                raise ToolError("No eval runs in evals/runs/")
        elif isinstance(run_id, str) and RUN_ID.match(run_id):
            path = evals.RUNS_DIR / f"{run_id}.jsonl"
        else:
            raise ToolError("run_id must be 'latest' or a run id like 20261020T075220Z")
        if not path.is_file():
            available = ", ".join(p.stem for p in sorted(evals.RUNS_DIR.glob("*.jsonl")))
            raise ToolError(f"No eval run {run_id}. Available: {available}")

        cases = load_cases()
        data_hash = dataset_hash(cases)
        records = evals.load_run(path)
        if {r["dataset_hash"] for r in records.values()} - {data_hash}:
            raise ToolError(f"Run {path.stem} was recorded against a different version of evals/cases.jsonl")
        case_ids = {c.case_id for c in cases}
        reports = [r for r in records.values() if r.get("report") and r["case_id"] in case_ids]
        judged = [r for r in reports if r.get("judge")]
        baseline = evals.load_baseline()
        rates = pass_rates(reports)
        base_rates = (baseline or {}).get("deterministic_pass_rates", {})
        llm_written = sum(r["report_source"] == "llm" for r in reports)

        def mean_score(criterion):
            scores = [r["judge"]["scores"][criterion]["score"] for r in judged]
            return round(sum(scores) / len(scores), 2) if scores else None

        return EvalSummary(
            run_id=path.stem, is_baseline_run=bool(baseline) and Path(baseline["run"]).stem == path.stem,
            dataset_hash=data_hash, cases=len(cases), reports=len(reports), judged=len(judged),
            llm_written=llm_written, template_fallbacks=len(reports) - llm_written,
            deterministic_checks=[CheckRate(check=name, baseline_rate=base_rates.get(name), **r)
                                  for name, r in rates.items()],
            below_baseline=[name for name, _, _ in evals.compare_to_baseline(rates, baseline)],
            judge_model=judged[0]["judge"]["model"] if judged else None,
            judge_mean_scores={c: mean_score(c) for c in CRITERIA})


async def run_blocking(fn, *args):
    """Run a tool body in a worker thread. Expected problems are ToolErrors with their own message;
    anything else is logged here (stderr) and reaches the client only as its exception type."""
    try:
        return await anyio.to_thread.run_sync(partial(fn, *args))
    except ToolError:
        raise
    except Exception as e:
        log.exception("Tool %s failed", getattr(fn, "__name__", fn))
        raise ToolError(f"Internal error ({type(e).__name__}); details are in the MCP server's log") from None


# ---- Resources ----

def load_condition_info():
    from agent.tools import CONDITION_INFO_PATH
    with open(CONDITION_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)


def latest_results_file():
    files = sorted((REPO / "evals" / "results").glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].md"))
    return files[-1] if files else None


def create_server(**clinic_options):
    """The MCP server. clinic_options go to Clinic (tests pass a fake model_service, llm_factory, dirs)."""
    clinic = Clinic(**clinic_options)
    server = MCPServer("clinical-diagnostics", title="Clinical Diagnostic Portal", version="1.0",
                       instructions=INSTRUCTIONS)
    server.clinic = clinic

    @server.tool(annotations=READ_ONLY)
    async def classify_xray(image_path: str) -> Classification:
        """Classify a chest X-ray as Normal, Pneumonia or Tuberculosis.

        image_path: a .jpg/.jpeg/.png file inside the server's allowed directories (absolute, or relative to
        the first allowed directory). Max 10 MB unless the server sets MAX_UPLOAD_MB.

        Returns the predicted class, temperature-calibrated probabilities for all classes (0-1, 4 dp) and the
        triage status: 'confident' if the top probability is at least 0.9, otherwise 'refer' (clinician
        review recommended). This is a screening aid, not a diagnosis: always report the status, and quote
        probabilities exactly as returned. Each call is a new prediction with a new trace_id.
        """
        return await run_blocking(clinic.classify_xray, image_path)

    @server.tool(annotations=READ_ONLY)
    async def gradcam_summary(image_path: str) -> GradcamSummary:
        """Where in the image the classifier looked (Grad-CAM for the predicted class).

        Returns the most influential zone of a coarse 3x2 grid (patient's right/left, assuming a PA view),
        the share of influence per zone, whether attention is focal or diffuse, how much sits on the image
        border, and the raw 7x7 map. This is not a lung segmentation and shows model attention, not
        confirmed anatomy; describe it with hedged language. Uses the image's latest classify_xray result
        (and trace) if there is one, otherwise classifies it first.
        """
        return await run_blocking(clinic.gradcam_summary, image_path)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=False,
                                             openWorldHint=True))
    async def generate_report(image_path: str, ctx: Context) -> ReportOut:
        """Write a plain-language report on the X-ray's classifier result, with the project's guarded report agent.

        An LLM (Groq, from config/llm.json) gathers the classifier output, the Grad-CAM summary and the
        condition text through tools, and writes the report. A guard rejects any report whose class,
        probability or status differs from the classifier, that claims a confirmed diagnosis, or that
        dismisses follow-up for Pneumonia/TB; after one retry, or without an LLM key or on an LLM error,
        a fixed template report is used. `source` says which: 'llm' or 'template'. Show `written_by` with
        the report. The LLM sees only tool outputs, never the image or file path. Can take up to a minute
        (rate limits). The same image's LLM report is cached in this process.
        """
        await ctx.report_progress(0, 1, "Writing the report (up to a minute on Groq's free tier)")
        result = await run_blocking(clinic.generate_report, image_path)
        await ctx.report_progress(1, 1, "Report ready")
        return result

    @server.tool(annotations=READ_ONLY)
    async def get_trace(trace_id: str) -> TraceOut:
        """Every recorded step of one prediction: preprocess, classify, calibrate, Grad-CAM, and for a report
        the LLM calls (tokens, nominal cost), tool calls, guard verdicts and whether the template was used.

        trace_id comes from classify_xray, gradcam_summary or generate_report. Only traces written by this
        MCP server (traces/mcp/) are visible.
        """
        return await run_blocking(clinic.get_trace, trace_id)

    @server.tool(annotations=READ_ONLY)
    async def eval_summary(run_id: str = "latest") -> EvalSummary:
        """Results of a committed report-agent eval run (evals/runs/): deterministic check pass rates against
        the CI baseline, LLM-written vs template counts, and mean LLM-judge scores per criterion.

        run_id: 'latest' (newest run file) or an id like '20261020T075220Z'. All numbers are on the
        project's in-distribution test split; the judge is known to miss subtle problems.
        """
        return await run_blocking(clinic.eval_summary, run_id)

    @server.resource("clinical://condition-info", name="condition-info", mime_type="application/json",
                     title="Condition background text (DRAFT)",
                     description="Background text for Normal, Pneumonia and Tuberculosis plus the model's known "
                                 "limitations, as given to the report agent. review_status: DRAFT, not yet "
                                 "reviewed by a clinician.")
    def condition_info() -> str:
        return json.dumps(load_condition_info(), indent=2, ensure_ascii=False)

    @server.resource("clinical://condition-info/{condition}", name="condition-info-by-class",
                     mime_type="application/json", title="Condition background text for one class (DRAFT)",
                     description=f"One of {', '.join(CLASS_NAMES)}, with the model's limitations and the "
                                 "review status (DRAFT, not yet reviewed by a clinician).")
    def condition_info_for(condition: str) -> str:
        info = load_condition_info()
        if condition not in CLASS_NAMES:
            raise ResourceNotFoundError(f"Unknown condition {condition!r}; expected one of {CLASS_NAMES}")
        return json.dumps({"condition": condition, **info[condition],
                           "model_limitations": info["_model_limitations"],
                           "review_status": info["_meta"]["review_status"]}, indent=2, ensure_ascii=False)

    @server.resource("clinical://evals/latest", name="eval-results-latest", mime_type="text/markdown",
                     title="Latest eval results",
                     description="The newest evals/results/<date>.md: deterministic checks, judge scores, "
                                 "judge-vs-human agreement and run notes.")
    def eval_results_latest() -> str:
        path = latest_results_file()
        if path is None:
            raise ResourceNotFoundError("No eval results in evals/results/")
        return path.read_text(encoding="utf-8")

    return server


def load_llm_key(secrets_path=SECRETS_PATH, environ=None):
    """Put the LLM provider's key from the repo's secrets.toml into this process's environment, if the env
    doesn't already have it. MCP clients start the server in any directory, where st.secrets wouldn't find
    the file. Only the provider's own section is read."""
    env = os.environ if environ is None else environ
    _, settings = provider_settings()
    if env.get(settings["env_key"]) or not Path(secrets_path).is_file():
        return
    with open(secrets_path, "rb") as f:
        key = tomllib.load(f).get(settings["secret_section"], {}).get("api_key")
    if key:
        env[settings["env_key"]] = key


def main():
    # stdout carries the protocol; logs go to stderr (the MCP client's server log)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("mcp_server").setLevel(logging.INFO)
    load_llm_key()
    server = create_server()
    log.info("Allowed image directories: %s", ", ".join(map(str, server.clinic.allowed_dirs)) or "none")
    server.run("stdio")

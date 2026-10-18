"""Tool-calling report agent.

The LLM gathers facts through tools and writes a Report. The classifier stays the
source of truth: a report whose class, probability or status differs from
run_classifier's output is rejected, and after one retry a deterministic template
report is used instead. The LLM never changes the prediction.

Only tool outputs are sent to the LLM: no image pixels and no patient details.
"""
import dataclasses
import hashlib
import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pydantic import ValidationError

from agent.schema import DISCLAIMER, Report
from agent.tools import TOOL_DECLARATIONS, ReportTools
from services.calibration import REFER_MESSAGE, STATUS_REFER
from services.tracing import estimate_cost, safe_set, trace_event, trace_span

GEMINI_MODEL = "gemini-3.8-flash"  # stable per https://ai.google.dev/gemini-api/docs/models
MAX_STEPS = 5   # LLM calls per report, including the retry
MAX_RETRIES = 1  # re-asks after an invalid or ungrounded report
RATE_LIMIT_MAX_RETRY_DELAY = 10.0  # seconds; on a 429, wait and retry once only if Google asks for less

BANNED_PATTERNS = [r"diagnosis\s+(is\s+|has\s+been\s+)?confirmed", r"confirmed\s+diagnosis"]
# A banned phrase is allowed when its own clause negates it within a few words, e.g. "not a confirmed
# diagnosis" or "until the diagnosis is confirmed". "No doubt"-style phrases are not negations.
NEGATION_CUES = re.compile(r"\b(not|no|never|without|cannot|nor|until|unless|before)\b|n't\b")
NOT_NEGATIONS = re.compile(r"\b(no|without|beyond)\s+(a\s+)?(doubt|question)\b|\bnot\s+only\b")
CLAUSE_BREAK = re.compile(r"[.;:!?,]|\bbut\b|\bhowever\b")
NEGATION_WINDOW_WORDS = 4

SYSTEM_PROMPT = f"""You write short, plain-language reports about an AI chest X-ray classifier's result.

Rules:
1. In your FIRST turn, call all three tools at once (parallel function calls), with no arguments:
   run_classifier, get_gradcam and get_condition_info (it defaults to the predicted class).
   Then write the report. Don't call tools one at a time.
2. run_classifier's output is authoritative. Copy predicted_class, calibrated_probability and status
   into the report EXACTLY as returned. Never change, round or reinterpret them.
3. Use get_gradcam to describe which approximate region influenced the model. Hedge: it is a coarse
   zone, it shows model attention rather than confirmed anatomy, and if focus is "diffuse" say so.
4. Use only get_condition_info's text for medical background. Do not add facts from elsewhere.
   Never say or imply the diagnosis is confirmed. The classifier is a screening aid, not a diagnosis.
5. If status is "refer", the recommendation must begin with "{REFER_MESSAGE.capitalize()}." and say
   that clinician review is needed.
6. summary: 2-3 plain-language sentences for a non-specialist. recommendation: next step for the
   reader. limitations: short bullet strings, including that the model was evaluated on data from
   two public datasets where each disease came from a different source.

When you have the facts, reply with ONLY a JSON object (no markdown) matching this JSON Schema:
{json.dumps(Report.model_json_schema())}"""

USER_PROMPT = "Write the report for the uploaded chest X-ray."


@dataclass
class ToolCall:
    name: str
    args: dict
    id: Optional[str] = None


@dataclass
class LLMTurn:
    """One model response: either tool calls or final text."""
    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider-native message, replayed as-is (keeps Gemini thought signatures)
    usage: Optional[dict] = None          # {"input_tokens", "output_tokens", "thinking_tokens"}
    finish_reason: Optional[str] = None
    rate_limit_retries: int = 0


class LLMClient(ABC):
    """Provider-neutral chat interface.

    messages is a list of {"role": "user", "text": str}, {"role": "assistant", "turn": LLMTurn}
    or {"role": "tool", "results": [(ToolCall, dict), ...]}.
    """
    name = "llm"

    @abstractmethod
    def generate(self, system, messages, tools):
        """Return the next LLMTurn."""


class RateLimitError(Exception):
    """The LLM provider returned 429 and we didn't (or couldn't) wait it out."""

    def __init__(self, retry_delay=None):
        super().__init__(f"rate limited (retry delay {retry_delay})")
        self.retry_delay = retry_delay


def retry_delay_seconds(error):
    """Retry delay Google sent with a 429 (RetryInfo detail, else 'retry in Ns' in the message), or None."""
    body = getattr(error, "details", None)
    details = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
    for d in details:
        if str(d.get("@type", "")).endswith("RetryInfo") and d.get("retryDelay"):
            return float(str(d["retryDelay"]).rstrip("s"))
    m = re.search(r"retry in ([\d.]+)\s*s", str(error))
    return float(m.group(1)) if m else None


def describe_llm_error(error):
    """Short, user-facing reason for an LLM failure (the full error stays in ReportResult.error_detail)."""
    if isinstance(error, RateLimitError):
        wait = f" (retry in {error.retry_delay:.0f} s)" if error.retry_delay else ""
        return f"Gemini rate limit reached{wait}."
    if hasattr(error, "code") and hasattr(error, "status"):  # google.genai.errors.APIError
        return f"Gemini error {error.code} {error.status}."
    return f"LLM call failed ({type(error).__name__})."


class GeminiClient(LLMClient):
    def __init__(self, api_key, model=GEMINI_MODEL, client=None, sleep=time.sleep):
        from google import genai
        from google.genai import errors, types
        self.types = types
        self.errors = errors
        self.model = model
        self.name = model
        self.client = client or genai.Client(api_key=api_key)
        self.sleep = sleep

    def _contents(self, messages):
        t = self.types
        contents = []
        for m in messages:
            if m["role"] == "user":
                contents.append(t.Content(role="user", parts=[t.Part(text=m["text"])]))
            elif m["role"] == "assistant":
                contents.append(m["turn"].raw)
            else:
                # Every FunctionResponse needs the call's id and name
                contents.append(t.Content(role="user", parts=[
                    t.Part(function_response=t.FunctionResponse(id=call.id, name=call.name,
                                                                response={"result": result}))
                    for call, result in m["results"]]))
        return contents

    def generate(self, system, messages, tools):
        t = self.types
        config = t.GenerateContentConfig(
            system_instruction=system,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=d["name"], description=d["description"],
                                      parameters_json_schema=d["parameters"]) for d in tools])],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=t.ThinkingConfig(thinking_level="low"),
        )
        contents = self._contents(messages)
        retries = 0
        for attempt in (1, 2):
            try:
                response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except self.errors.ClientError as e:
                if e.code != 429:
                    raise
                delay = retry_delay_seconds(e)
                # One retry, and only when Google's requested wait is short; otherwise fall back now
                if attempt == 1 and delay is not None and delay < RATE_LIMIT_MAX_RETRY_DELAY:
                    self.sleep(delay)
                    retries += 1
                    continue
                raise RateLimitError(delay) from e
        if not response.candidates or response.candidates[0].content is None:
            raise RuntimeError(f"Gemini returned no content: {response.prompt_feedback}")
        content = response.candidates[0].content
        calls = [ToolCall(name=c.name, args=dict(c.args or {}), id=c.id) for c in (response.function_calls or [])]
        text = None if calls else "".join(p.text for p in content.parts or [] if p.text and not p.thought)
        meta = response.usage_metadata
        usage = None if meta is None else {
            "input_tokens": meta.prompt_token_count,
            "output_tokens": meta.candidates_token_count,
            "thinking_tokens": meta.thoughts_token_count,
        }
        finish = response.candidates[0].finish_reason
        return LLMTurn(text=text, tool_calls=calls, raw=content, usage=usage, rate_limit_retries=retries,
                       finish_reason=getattr(finish, "name", None) or (str(finish) if finish else None))


class MockLLMClient(LLMClient):
    """Scripted client for tests. Each script item is an LLMTurn or a callable(messages) -> LLMTurn."""
    name = "mock"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate(self, system, messages, tools):
        self.calls.append(list(messages))
        if not self.script:
            raise RuntimeError("MockLLMClient script exhausted")
        item = self.script.pop(0)
        return item(messages) if callable(item) else item

    @staticmethod
    def tool(name, **args):
        return LLMTurn(tool_calls=[ToolCall(name=name, args=args, id=f"call-{name}")])

    @staticmethod
    def final(obj, usage=None):
        return LLMTurn(text=obj if isinstance(obj, str) else json.dumps(obj), usage=usage, finish_reason="STOP")


@dataclass
class ReportResult:
    report: Report
    source: str                      # "llm" or "fallback"
    model: Optional[str] = None      # LLM name when source == "llm"
    reason: Optional[str] = None     # why the fallback was used
    steps: int = 0
    problems: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)  # exact LLM outputs the guard rejected, for audit/tests
    error_detail: Optional[str] = None  # full LLM error text (not shown to users)
    cached: bool = False                 # served from REPORT_CACHE, no LLM requests spent

    def source_line(self):
        """One line saying who wrote the report, for the UI and PDF (the fallback reason is UI-only)."""
        if self.source == "llm":
            return (f"Written by {self.model} from the classifier's output; class, probability and "
                    "status were checked against the classifier.")
        return "Standard template report generated from the classifier's output."


def load_gemini_settings():
    """(api_key, model) from st.secrets["gemini"] or env GEMINI_API_KEY / GEMINI_MODEL."""
    secrets = {}
    try:
        import streamlit as st
        secrets = dict(st.secrets.get("gemini", {}))
    except Exception:
        pass  # no secrets file outside Streamlit
    api_key = secrets.get("api_key") or os.environ.get("GEMINI_API_KEY")
    model = secrets.get("model") or os.environ.get("GEMINI_MODEL") or GEMINI_MODEL
    return api_key, model


def build_llm_client():
    """GeminiClient if a key is configured, else None (callers then get the template report)."""
    api_key, model = load_gemini_settings()
    return GeminiClient(api_key, model) if api_key else None


def asserts_confirmed_diagnosis(text):
    """True if the text claims a confirmed diagnosis.

    Negated or conditional mentions in the same clause ("not a confirmed diagnosis", "does not mean the
    diagnosis is confirmed", "until a diagnosis is confirmed") don't count.
    """
    text = text.lower()
    for pattern in BANNED_PATTERNS:
        for match in re.finditer(pattern, text):
            clause = CLAUSE_BREAK.split(text[:match.start()])[-1]
            window = " ".join(clause.split()[-NEGATION_WINDOW_WORDS:])
            if not NEGATION_CUES.search(NOT_NEGATIONS.sub(" ", window)):
                return True
    return False


def check_grounding(report, truth):
    """Problems that make a schema-valid report unacceptable (empty list = grounded)."""
    problems = []
    for key in ("predicted_class", "calibrated_probability", "status"):
        if getattr(report, key) != truth[key]:
            problems.append(f"{key} is {getattr(report, key)!r} but run_classifier returned {truth[key]!r}")
    text = " ".join([report.summary, report.recommendation, *report.limitations, report.disclaimer])
    if asserts_confirmed_diagnosis(text):
        problems.append("the report must never say the diagnosis is confirmed")
    if truth["status"] == STATUS_REFER and "clinician review" not in report.recommendation.lower():
        problems.append("status is 'refer', so the recommendation must say clinician review is recommended")
    return problems


def parse_report(text, truth):
    """(Report, None) if the text is a valid, grounded report, else (None, problem)."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)  # tolerate a markdown fence
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON ({e})"
    try:
        report = Report.model_validate(data)
    except ValidationError as e:
        return None, "schema validation failed: " + "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
    problems = check_grounding(report, truth)
    if problems:
        return None, "grounding check failed: " + "; ".join(problems)
    return report, None


def fallback_report(tools):
    """Deterministic template report built only from tool outputs. Always valid and grounded."""
    truth = tools.run_classifier()
    cls, prob = truth["predicted_class"], truth["calibrated_probability"]
    info = tools.get_condition_info(cls)
    gradcam = tools.call("get_gradcam", {})

    # 2 decimals = the 4-dp tool value, matching the PDF's probability table
    sentences = [f"The AI model's most likely category for this chest X-ray is {cls}, "
                 f"with a calibrated probability of {prob:.2%}."]
    if truth["status"] == STATUS_REFER:
        sentences.append(f"This is below the {truth['referral_threshold']:.0%} confidence threshold, "
                         "so the result is uncertain.")
    else:
        sentences.append(f"This meets the {truth['referral_threshold']:.0%} confidence threshold, "
                         "but it is a screening result, not a diagnosis.")
    if gradcam.get("peak_region"):
        spread = "spread across the image" if gradcam.get("focus") != "focal" else "concentrated"
        sentences.append(f"The model's attention was {spread}, peaking around the {gradcam['peak_region']}.")

    if truth["status"] == STATUS_REFER:
        recommendation = f"{REFER_MESSAGE.capitalize()}. {info['typical_next_steps']}"
    else:
        recommendation = f"{info['typical_next_steps']} A clinician should review the image before acting on this result."

    report = Report(
        predicted_class=cls,
        calibrated_probability=prob,
        status=truth["status"],
        summary=" ".join(sentences),
        recommendation=recommendation,
        limitations=[
            info["what_this_result_means"],
            "The model was evaluated on two public datasets in which each disease came from a different "
            "source, so performance on images from other hospitals or scanners may be lower.",
            "Grad-CAM regions are approximate and show what influenced the model, not confirmed findings.",
        ],
        disclaimer=DISCLAIMER,
    )
    assert not check_grounding(report, truth), check_grounding(report, truth)
    return report


def llm_call_attributes(turn, model):
    """Span attributes for one LLM response: tokens, nominal cost, finish reason, retries."""
    usage = turn.usage or {}
    output = None
    if usage.get("output_tokens") is not None:
        output = usage["output_tokens"] + (usage.get("thinking_tokens") or 0)  # thinking is billed as output
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "thinking_tokens": usage.get("thinking_tokens"),
        **estimate_cost(model, usage.get("input_tokens"), output),
        "finish_reason": turn.finish_reason,
        "tool_calls": len(turn.tool_calls),
        "rate_limit_retries": turn.rate_limit_retries,
    }


def generate_report(tools, llm, max_steps=MAX_STEPS, max_retries=MAX_RETRIES, tracer=None):
    """Run the tool-calling loop and return a ReportResult (LLM report or template fallback).

    Traced as a 'report' span holding llm_call, tool_call, guard and fallback_used spans.
    """
    with trace_span(tracer, "report", llm=getattr(llm, "name", None)) as span:
        result = _generate_report(tools, llm, max_steps, max_retries, tracer)
        trace_event(tracer, "fallback_used", used=result.source == "fallback", reason=result.reason)
        safe_set(span, source=result.source, steps=result.steps, rejected=len(result.rejected))
    return result


def _generate_report(tools, llm, max_steps, max_retries, tracer):
    truth = tools.run_classifier()  # ground truth for the guard, independent of what the LLM calls
    result = ReportResult(report=None, source="fallback")

    def fall_back(reason, detail=None):
        result.report, result.source, result.reason = fallback_report(tools), "fallback", reason
        result.error_detail = detail
        return result

    if llm is None:
        return fall_back("No LLM configured (set st.secrets['gemini']['api_key'] or GEMINI_API_KEY).")

    messages = [{"role": "user", "text": USER_PROMPT}]
    retries_left = max_retries
    for step in range(1, max_steps + 1):
        result.steps = step
        try:
            with trace_span(tracer, "llm_call", model=llm.name, step=step) as span:
                turn = llm.generate(SYSTEM_PROMPT, messages, TOOL_DECLARATIONS)
                safe_set(span, **llm_call_attributes(turn, llm.name))
        except Exception as e:
            detail = f"{type(e).__name__}: {e}" + (f" <- {e.__cause__}" if e.__cause__ else "")
            return fall_back(describe_llm_error(e), detail)
        messages.append({"role": "assistant", "turn": turn})

        if turn.tool_calls:
            results = []
            for call in turn.tool_calls:
                with trace_span(tracer, "tool_call", tool=call.name, args=call.args, step=step) as span:
                    output = tools.call(call.name, call.args)
                    if isinstance(output, dict) and "error" in output:
                        span.fail(output["error"], "ToolError")
                results.append((call, output))
            result.tool_calls += [call.name for call in turn.tool_calls]
            messages.append({"role": "tool", "results": results})
            continue

        report, problem = parse_report(turn.text, truth)
        trace_event(tracer, "guard", status="ok" if report is not None else "reject",
                    error=problem, result="pass" if report is not None else "reject", reason=problem, step=step)
        if report is not None:
            result.report = report.model_copy(update={"disclaimer": DISCLAIMER})
            result.source, result.model = "llm", llm.name
            return result
        result.problems.append(problem)
        result.rejected.append({"step": step, "text": turn.text, "problem": problem})
        if retries_left == 0:
            return fall_back(f"Report rejected after {max_retries} retry: {problem}.")
        retries_left -= 1
        messages.append({"role": "user", "text": f"Your report was rejected: {problem}. "
                                                 "Reply with a corrected JSON report only."})

    return fall_back(f"No valid report within {max_steps} steps.")


class ReportCache:
    """Thread-safe LRU of LLM-written reports (Streamlit serves sessions from threads)."""

    def __init__(self, max_items=256):
        self.max_items = max_items
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key, value):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def __len__(self):
        return len(self._items)


REPORT_CACHE = ReportCache()


def image_fingerprint(image):
    """SHA-256 of the decoded pixels (plus mode and size), so the same scan hashes the same however it was uploaded."""
    h = hashlib.sha256(f"{image.mode}|{image.size}|".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def generate_report_for_image(model_service, image, prediction=None, llm=None, cache=REPORT_CACHE, tracer=None):
    """Dashboard entry point. Reuses a cached LLM report for the same image, temperature and model.

    Only LLM reports are cached: a fallback (e.g. after a rate limit) costs nothing to rebuild, and
    caching it would stop a later click from trying the LLM again.
    """
    key = None
    if llm is not None and cache is not None:
        key = (image_fingerprint(image), float(model_service.temperature), llm.name)
        hit = cache.get(key)
        if hit is not None:
            with trace_span(tracer, "report", llm=llm.name, source=hit.source, cached=True):
                trace_event(tracer, "report_cache_hit")
            return dataclasses.replace(hit, cached=True, steps=0)
    result = generate_report(ReportTools(model_service, image, prediction, tracer=tracer), llm, tracer=tracer)
    if key is not None and result.source == "llm":
        cache.put(key, result)
    return result

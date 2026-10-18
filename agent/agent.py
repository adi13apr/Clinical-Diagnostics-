"""Tool-calling report agent.

The LLM gathers facts through tools and writes a Report. The classifier stays the
source of truth: a report whose class, probability or status differs from
run_classifier's output is rejected, and after one retry a deterministic template
report is used instead. The LLM never changes the prediction.

Only tool outputs are sent to the LLM: no image pixels and no patient details.
"""
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pydantic import ValidationError

from agent.schema import DISCLAIMER, Report
from agent.tools import TOOL_DECLARATIONS, ReportTools
from services.calibration import REFER_MESSAGE, STATUS_REFER

GEMINI_MODEL = "gemini-3.8-flash"  # stable per https://ai.google.dev/gemini-api/docs/models
MAX_STEPS = 5   # LLM calls per report, including the retry
MAX_RETRIES = 1  # re-asks after an invalid or ungrounded report

BANNED_PATTERNS = [r"diagnosis\s+(is\s+|has\s+been\s+)?confirmed", r"confirmed\s+diagnosis"]

SYSTEM_PROMPT = f"""You write short, plain-language reports about an AI chest X-ray classifier's result.

Rules:
1. Call run_classifier first. Its output is authoritative. Copy predicted_class, calibrated_probability
   and status into the report EXACTLY as returned. Never change, round or reinterpret them.
2. Call get_gradcam to describe which approximate region influenced the model. Hedge: it is a coarse
   zone, it shows model attention rather than confirmed anatomy, and if focus is "diffuse" say so.
3. Call get_condition_info for the predicted class. Use only that text for medical background.
   Do not add facts from elsewhere.
4. Never say or imply the diagnosis is confirmed. The classifier is a screening aid, not a diagnosis.
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


class LLMClient(ABC):
    """Provider-neutral chat interface.

    messages is a list of {"role": "user", "text": str}, {"role": "assistant", "turn": LLMTurn}
    or {"role": "tool", "results": [(ToolCall, dict), ...]}.
    """
    name = "llm"

    @abstractmethod
    def generate(self, system, messages, tools):
        """Return the next LLMTurn."""


class GeminiClient(LLMClient):
    def __init__(self, api_key, model=GEMINI_MODEL, client=None):
        from google import genai
        from google.genai import types
        self.types = types
        self.model = model
        self.name = model
        self.client = client or genai.Client(api_key=api_key)

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
        response = self.client.models.generate_content(
            model=self.model, contents=self._contents(messages), config=config)
        if not response.candidates or response.candidates[0].content is None:
            raise RuntimeError(f"Gemini returned no content: {response.prompt_feedback}")
        content = response.candidates[0].content
        calls = [ToolCall(name=c.name, args=dict(c.args or {}), id=c.id) for c in (response.function_calls or [])]
        text = None if calls else "".join(p.text for p in content.parts or [] if p.text and not p.thought)
        return LLMTurn(text=text, tool_calls=calls, raw=content)


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
    def final(obj):
        return LLMTurn(text=obj if isinstance(obj, str) else json.dumps(obj))


@dataclass
class ReportResult:
    report: Report
    source: str                      # "llm" or "fallback"
    model: Optional[str] = None      # LLM name when source == "llm"
    reason: Optional[str] = None     # why the fallback was used
    steps: int = 0
    problems: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)

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


def check_grounding(report, truth):
    """Problems that make a schema-valid report unacceptable (empty list = grounded)."""
    problems = []
    for key in ("predicted_class", "calibrated_probability", "status"):
        if getattr(report, key) != truth[key]:
            problems.append(f"{key} is {getattr(report, key)!r} but run_classifier returned {truth[key]!r}")
    text = " ".join([report.summary, report.recommendation, *report.limitations, report.disclaimer]).lower()
    if any(re.search(p, text) for p in BANNED_PATTERNS):
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
        spread = "spread across the image" if gradcam["focus"] == "diffuse" else "concentrated"
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


def generate_report(tools, llm, max_steps=MAX_STEPS, max_retries=MAX_RETRIES):
    """Run the tool-calling loop and return a ReportResult (LLM report or template fallback)."""
    truth = tools.run_classifier()  # ground truth for the guard, independent of what the LLM calls
    result = ReportResult(report=None, source="fallback")

    def fall_back(reason):
        result.report, result.source, result.reason = fallback_report(tools), "fallback", reason
        return result

    if llm is None:
        return fall_back("No LLM configured (set st.secrets['gemini']['api_key'] or GEMINI_API_KEY)")

    messages = [{"role": "user", "text": USER_PROMPT}]
    retries_left = max_retries
    for step in range(1, max_steps + 1):
        result.steps = step
        try:
            turn = llm.generate(SYSTEM_PROMPT, messages, TOOL_DECLARATIONS)
        except Exception as e:
            return fall_back(f"LLM call failed: {type(e).__name__}: {e}")
        messages.append({"role": "assistant", "turn": turn})

        if turn.tool_calls:
            results = [(call, tools.call(call.name, call.args)) for call in turn.tool_calls]
            result.tool_calls += [call.name for call in turn.tool_calls]
            messages.append({"role": "tool", "results": results})
            continue

        report, problem = parse_report(turn.text, truth)
        if report is not None:
            result.report = report.model_copy(update={"disclaimer": DISCLAIMER})
            result.source, result.model = "llm", llm.name
            return result
        result.problems.append(problem)
        if retries_left == 0:
            return fall_back(f"Report rejected after {max_retries} retry: {problem}")
        retries_left -= 1
        messages.append({"role": "user", "text": f"Your report was rejected: {problem}. "
                                                 "Reply with a corrected JSON report only."})

    return fall_back(f"No valid report within {max_steps} steps")


def generate_report_for_image(model_service, image, prediction=None, llm=None):
    """Convenience wrapper used by the dashboard."""
    return generate_report(ReportTools(model_service, image, prediction), llm)

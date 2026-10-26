"""Multi-agent report pipeline: an orchestrator runs an evidence agent, a writer and a verifier.

    EVIDENCE -> WRITE -> GUARD -> VERIFY -> FINAL_GUARD -> DONE
                  ^        |         |
                  +--------+---------+   REVISE: at most once, for the guard or the verifier
    any state -> FALLBACK (the template report)

- Evidence agent (tool-calling): must call run_classifier, get_gradcam and get_condition_info (for the
  predicted class). If it stops without all three, the orchestrator re-asks once, naming what's missing;
  if they are still missing, the template is used. Its tool outputs (never its own words) become the
  EvidenceBundle.
- Writer (no tools): writes the Report from the EvidenceBundle only, as strict json_schema output.
- Guard: the same deterministic check_grounding as the single agent, on every draft before the verifier
  (a draft it rejects never costs a verifier call), and again on the final report: it has the final say.
- Verifier: checks every claim in the draft against the EvidenceBundle and returns findings (claim,
  supported, source, reason). An unsupported claim sends the draft back once; the revision is verified
  again, and if it still has one, the template is used.

The classifier stays the source of truth, as in agent.agent: only tool outputs reach the LLMs, and the
guard compares the report with run_classifier's output. The single agent (agent.agent._report_loop) is
unchanged and is the baseline this pipeline is compared with.

One loop, two drivers (as in agent.agent): _orchestrate is a generator that yields an AgentRequest and
receives the outcome (an LLMTurn or the exception), so the sync path (app, API, worker) and the async
path (eval runner) make the same decisions.

Each agent stage has a wall-clock limit (config/llm.json "agents", including rate-limit budget waits);
a report job's own deadline applies on top. A stage past its limit falls back to the template.
"""
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.agent import (
    DISCLAIMER, MAX_RETRIES, TRUNCATED_FINISH_REASONS, ReportResult, _close_report_span, check_grounding,
    classify_llm_error, describe_llm_error, fallback_report, llm_call_attributes, parse_report,
)
from agent.schema import ClassName, Report, Status
from agent.tools import CLASS_NAMES, TOOL_DECLARATIONS
from services.calibration import REFER_MESSAGE
from services.tracing import safe_set, trace_event, trace_span

AGENTS = ("evidence", "writer", "verifier")
REQUIRED_TOOLS = ("run_classifier", "get_gradcam", "get_condition_info")
MAX_REVISIONS = 1         # drafts sent back to the writer, for the guard and the verifier together
EVIDENCE_MAX_CALLS = 6    # 3 tool turns, one re-ask, and a retry for a model error
DEFAULT_TIMEOUTS_S = {"evidence": 180.0, "writer": 150.0, "verifier": 120.0}


class State(str, Enum):
    EVIDENCE = "evidence"
    WRITE = "write"
    GUARD = "guard"
    VERIFY = "verify"
    FINAL_GUARD = "final_guard"
    DONE = "done"
    FALLBACK = "fallback"


# ---- Typed messages between the agents ----

class ClassifierFacts(BaseModel):
    """run_classifier's output (4 dp), as the evidence agent received it."""
    model_config = ConfigDict(extra="forbid")

    predicted_class: ClassName
    calibrated_probability: float
    probabilities: dict
    status: Status
    referral_threshold: float
    referral_message: Optional[str] = None


class EvidenceBundle(BaseModel):
    """Evidence agent -> writer and verifier: tool outputs only, never the evidence agent's own text."""
    model_config = ConfigDict(extra="forbid")

    classifier: ClassifierFacts
    gradcam: dict
    condition_info: dict
    dataset_caveat: Optional[str] = Field(
        description="condition_info.model_limitations[0], the dataset/source caveat the report must quote")
    tool_calls: List[str] = Field(description="Tools the evidence agent called, in order")

    def prompt_json(self):
        """What the writer and verifier see."""
        return json.dumps(self.model_dump(exclude={"tool_calls"}), indent=1)


class ClaimFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    supported: bool
    source: Literal["classifier", "gradcam", "condition_info", "none"]
    reason: str


class VerifierVerdict(BaseModel):
    """Verifier -> orchestrator."""
    model_config = ConfigDict(extra="forbid")

    findings: List[ClaimFinding] = Field(min_length=1)

    @property
    def unsupported(self):
        return [f for f in self.findings if not f.supported]


class RevisionRequest(BaseModel):
    """Orchestrator -> writer: why the draft came back."""
    model_config = ConfigDict(extra="forbid")

    by: Literal["guard", "verifier"]
    problems: List[str] = Field(default_factory=list)
    unsupported: List[ClaimFinding] = Field(default_factory=list)

    def prompt(self):
        lines = [f"Your draft was sent back by the {self.by}."]
        lines += [f"- Problem: {p}" for p in self.problems]
        lines += [f"- Unsupported claim: \"{f.claim}\" ({f.reason})" for f in self.unsupported]
        lines.append("Write the corrected report as JSON. Keep what was fine; remove or fix only these, and use "
                     "only the EVIDENCE.")
        return "\n".join(lines)


# ---- Strict json_schema for the writer and the verifier (every object closed and fully required) ----

REPORT_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["predicted_class", "calibrated_probability", "status", "summary", "recommendation",
                 "limitations", "disclaimer"],
    "properties": {
        "predicted_class": {"type": "string", "enum": CLASS_NAMES},
        "calibrated_probability": {"type": "number"},
        "status": {"type": "string", "enum": ["confident", "refer"]},
        "summary": {"type": "string"},
        "recommendation": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "disclaimer": {"type": "string"},
    },
}

VERDICT_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["findings"],
    "properties": {"findings": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["claim", "supported", "source", "reason"],
        "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"},
                       "source": {"type": "string", "enum": ["classifier", "gradcam", "condition_info", "none"]},
                       "reason": {"type": "string"}}}}},
}

# ---- Prompts ----

EVIDENCE_SYSTEM = """You collect evidence for a report about an AI chest X-ray classifier's result. You do not
write the report; another agent does, from the tool outputs only.

Call each of the three tools once, with no arguments: run_classifier, get_gradcam and get_condition_info
(it defaults to the predicted class). If you can make parallel function calls, call all three in your first
turn. When all three have returned, reply with the single word DONE."""

EVIDENCE_USER = "Collect the evidence for the uploaded chest X-ray."

WRITER_SYSTEM = f"""You write short, plain-language reports about an AI chest X-ray classifier's result. You get
EVIDENCE (tool outputs collected by another agent) and nothing else; you have no tools.

Rules:
1. EVIDENCE.classifier is authoritative. Copy predicted_class, calibrated_probability and status into the
   report EXACTLY as given. Never change, round or reinterpret them.
2. Use EVIDENCE.gradcam to describe which approximate region influenced the model. Hedge: it is a coarse
   zone, it shows model attention rather than confirmed anatomy, and if focus is "diffuse" say so. If it
   holds an "error", say the attention map is unavailable.
3. Use only EVIDENCE.condition_info for medical background. Do not add facts from elsewhere.
   Never say or imply the diagnosis is confirmed. The classifier is a screening aid, not a diagnosis.
4. Only if status is "refer": the recommendation must begin with "{REFER_MESSAGE.capitalize()}." and say
   that clinician review is needed. If status is "confident": never call the result uncertain and never
   use that refer sentence. Separately, whatever the status, if predicted_class is Pneumonia or
   Tuberculosis the recommendation must advise clinician follow-up and must never say that no referral,
   action or follow-up is needed.
5. summary: 2-3 plain-language sentences for a non-specialist. recommendation: next step for the reader.
   limitations: condition_info.what_this_result_means and the items of condition_info.model_limitations,
   copied word for word; EVIDENCE.dataset_caveat must be one of them, verbatim. Don't invent others.
6. A verifier checks every claim against the EVIDENCE: write nothing it doesn't support.

Reply with the report as JSON. The disclaimer can be any short text; it is replaced with a fixed one."""

VERIFIER_SYSTEM = """You check a draft report about an AI chest X-ray classifier's result against the EVIDENCE it
was written from (tool outputs). The EVIDENCE is the only ground truth: do not use outside medical knowledge,
except to notice claims the EVIDENCE does not support.

List every factual claim in the draft's summary, recommendation and limitations (not the disclaimer), one
finding per claim, in order. For each:
- supported: true if the EVIDENCE states it or it follows directly from it (copying, paraphrasing, or a
  probability shown with fewer decimals is fine); false if the EVIDENCE contradicts it or says nothing about it.
- source: the part of the EVIDENCE that supports it (classifier, gradcam, condition_info), or none.
- reason: one short sentence.
Advice to involve a clinician, to follow up, or to treat the result as a screening aid rather than a
diagnosis is supported: the report rules require it. Hedged wording about uncertainty is supported when it
matches the classifier's status.

Reply with the findings as JSON."""


def writer_prompt(bundle):
    return f"EVIDENCE:\n{bundle.prompt_json()}\n\nWrite the report."


def verifier_prompt(bundle, report):
    draft = report.model_dump(exclude={"disclaimer"})
    return f"EVIDENCE:\n{bundle.prompt_json()}\n\nDRAFT REPORT:\n{json.dumps(draft, indent=1)}"


# ---- The pipeline object (what build_report_llm returns in multi mode) ----

class AgentTimeoutError(TimeoutError):
    """An agent stage ran past its time limit (classified 'infrastructure', like any timeout)."""

    def __init__(self, agent, limit_s):
        super().__init__(f"{agent} agent ran past its {limit_s:.0f} s limit")
        self.agent, self.limit_s = agent, limit_s


class MultiAgentPipeline:
    """Three LLM clients (one per agent; agents on the same model share its budget) and their time limits.

    Used where an LLMClient is: name (cache key, traces), deadline (set by report jobs; on clock),
    pipeline ("multi", so generate_report_for_image runs the orchestrator).
    """
    pipeline = "multi"
    pipeline_note = None

    def __init__(self, evidence, writer, verifier, timeouts=None, clock=time.monotonic):
        self.clients = {"evidence": evidence, "writer": writer, "verifier": verifier}
        self.timeouts = {**DEFAULT_TIMEOUTS_S, **(timeouts or {})}
        self.clock = clock
        self.deadline = None  # a report job's deadline (clock time), on top of the per-agent limits
        self.name = "+".join(f"{agent}={model}" for agent, model in self.models.items())

    @property
    def models(self):
        return {agent: getattr(client, "name", None) for agent, client in self.clients.items()}


@dataclass
class AgentRequest:
    """One LLM call the orchestrator needs made."""
    agent: str
    step: int                        # LLM call number in this report
    kind: str                        # "tools" (generate) or "json" (generate_json)
    system: str
    messages: list
    deadline: float                  # pipeline.clock time; the call must start and finish before it
    limit_s: float                   # the stage's limit, for the error message
    tools_allowed: bool = True
    schema_name: Optional[str] = None
    schema: Optional[dict] = None


class _Fallback(Exception):
    def __init__(self, reason, detail=None, kind=None):
        super().__init__(reason)
        self.reason, self.detail, self.kind = reason, detail, kind


def _error_detail(e):
    return f"{type(e).__name__}: {e}" + (f" <- {e.__cause__}" if e.__cause__ else "")


class _Orchestrator:
    """State and bookkeeping for one report; _run is the state machine (a generator)."""

    def __init__(self, tools, pipeline, tracer):
        self.tools, self.pipeline, self.tracer = tools, pipeline, tracer
        self.truth = tools.run_classifier()  # ground truth for the guard, independent of what the agents do
        self.result = ReportResult(report=None, source="fallback", pipeline="multi",
                                   agent_models=pipeline.models)
        self.steps = 0
        self.revisions_left = MAX_REVISIONS
        self.state = None
        self._last_draft_text = None  # the writer's latest draft, exactly as returned

    # -- helpers --

    def _transition(self, state, **attributes):
        trace_event(self.tracer, "orchestrator_state", state=state.value,
                    previous=self.state.value if self.state else None, **attributes)
        self.state = state

    def _handoff(self, source, target, message):
        text = message.model_dump_json()
        trace_event(self.tracer, "handoff", from_agent=source, to_agent=target, message=type(message).__name__,
                    chars=len(text))

    def _deadline(self, agent):
        limit = self.pipeline.timeouts[agent]
        deadline = self.pipeline.clock() + limit
        if self.pipeline.deadline is not None:
            deadline = min(deadline, self.pipeline.deadline)
        return deadline, limit

    def _request(self, agent, kind, system, messages, deadline, limit, **kwargs):
        self.steps += 1
        self.result.steps = self.steps
        return AgentRequest(agent, self.steps, kind, system, messages, deadline, limit, **kwargs)

    def _llm_failure(self, agent, e, retries_left):
        """Retry once for a model failure (e.g. an invalid tool call); anything else falls back."""
        kind = classify_llm_error(e)
        problem = f"{agent} agent: {describe_llm_error(e)}"
        if kind == "model" and retries_left > 0:
            self.result.problems.append(problem)
            self.result.rejected.append({"step": self.steps, "agent": agent, "text": None, "problem": problem,
                                         "error_detail": _error_detail(e)[:500]})
            return
        if isinstance(e, AgentTimeoutError):
            problem = f"The {agent} agent ran past its {e.limit_s:.0f} s time limit."
        raise _Fallback(problem, _error_detail(e), kind)

    def _truncated(self, agent, turn, retries_left):
        problem = f"the {agent} agent's reply was cut off at the output-token limit (finish_reason {turn.finish_reason!r})"
        trace_event(self.tracer, "llm_truncated", status="error", error=problem, step=self.steps, agent=agent,
                    finish_reason=turn.finish_reason, output_tokens=(turn.usage or {}).get("output_tokens"))
        self.result.problems.append(problem)
        if agent == "writer":  # a cut-off draft is a report draft; the evidence/verifier replies are not
            self.result.rejected.append({"step": self.steps, "agent": agent, "text": turn.text, "problem": problem,
                                         "finish_reason": turn.finish_reason})
        if retries_left == 0:
            raise _Fallback(f"The {agent} agent's reply was cut off at the output-token limit after "
                            f"{MAX_RETRIES} retry.", problem, "model")

    # -- the state machine --

    def run(self):
        if any(client is None for client in self.pipeline.clients.values()):
            return self._fall_back(_Fallback("No LLM configured for every agent of the multi-agent pipeline."))
        try:
            self._transition(State.EVIDENCE)
            bundle = yield from self._evidence()
            self._handoff("evidence", "writer", bundle)
            revision = None
            draft_messages = None
            while True:
                self._transition(State.WRITE, revision=MAX_REVISIONS - self.revisions_left)
                report, draft_messages = yield from self._write(bundle, draft_messages, revision)
                self._transition(State.GUARD)
                problem = self._guard(report)
                if problem is not None:
                    revision = self._revise(RevisionRequest(by="guard", problems=[problem]), report, problem)
                    continue
                self._transition(State.VERIFY)
                self._handoff("writer", "verifier", report)
                verdict = yield from self._verify(bundle, report)
                if verdict.unsupported:
                    revision = self._revise(RevisionRequest(by="verifier", unsupported=verdict.unsupported),
                                            report, None)
                    continue
                self.result.findings = [f.model_dump() for f in verdict.findings]
                self._transition(State.FINAL_GUARD)
                return self._finish(report)
        except _Fallback as f:
            return self._fall_back(f)

    def _fall_back(self, f):
        self._transition(State.FALLBACK, reason=f.reason)
        r = self.result
        r.report, r.source, r.reason, r.error_detail, r.error_kind = fallback_report(self.tools), "fallback", \
            f.reason, f.detail, f.kind
        return r

    def _evidence(self):
        """Run the evidence agent until all three tools have returned; returns the EvidenceBundle."""
        deadline, limit = self._deadline("evidence")
        messages = [{"role": "user", "text": EVIDENCE_USER}]
        gathered, called = {}, []
        reasked, retries_left = False, MAX_RETRIES
        predicted = self.truth["predicted_class"]
        with trace_span(self.tracer, "evidence_agent", agent="evidence", model=self.pipeline.models["evidence"],
                        step=self.steps + 1) as span:
            usage = _Usage()
            for _ in range(EVIDENCE_MAX_CALLS):
                outcome = yield self._request("evidence", "tools", EVIDENCE_SYSTEM, messages, deadline, limit)
                if isinstance(outcome, Exception):
                    self._llm_failure("evidence", outcome, retries_left)
                    retries_left -= 1
                    continue
                turn = outcome
                usage.add(turn)
                if turn.finish_reason in TRUNCATED_FINISH_REASONS:
                    self._truncated("evidence", turn, retries_left)
                    retries_left -= 1
                    messages.append({"role": "user", "text": "Your previous reply was cut off and discarded. "
                                                             "Call the tools again."})
                    continue
                messages.append({"role": "assistant", "turn": turn})
                if turn.tool_calls:
                    results = []
                    for call in turn.tool_calls:
                        with trace_span(self.tracer, "tool_call", tool=call.name, args=call.args, step=self.steps,
                                        agent="evidence") as tool_span:
                            output = self.tools.call(call.name, call.args)
                            if isinstance(output, dict) and "error" in output:
                                tool_span.fail(output["error"], "ToolError")
                        results.append((call, output))
                        called.append(call.name)
                        # Condition text only counts for the predicted class (the default)
                        if call.name in REQUIRED_TOOLS and (call.name != "get_condition_info"
                                                            or (call.args or {}).get("condition") in (None, predicted)):
                            gathered.setdefault(call.name, output)
                    messages.append({"role": "tool", "results": results})
                    if all(name in gathered for name in REQUIRED_TOOLS):
                        break
                    continue
                # A text reply before every tool returned: re-ask once, naming what's missing
                missing = [name for name in REQUIRED_TOOLS if name not in gathered]
                if reasked:
                    self._set_stage(span, usage, verdict="incomplete", missing=missing)
                    raise _Fallback(f"The evidence agent did not call {', '.join(missing)} (asked twice).",
                                    kind="model")
                reasked = True
                self.result.problems.append(f"evidence agent stopped without calling {', '.join(missing)}")
                trace_event(self.tracer, "evidence_reask", missing=missing, step=self.steps)
                messages.append({"role": "user", "text": f"You have not called {', '.join(missing)} yet. "
                                                         "Call the missing tool(s) now, with no arguments."})
            missing = [name for name in REQUIRED_TOOLS if name not in gathered]
            self.result.tool_calls = called
            if missing:
                self._set_stage(span, usage, verdict="incomplete", missing=missing)
                raise _Fallback(f"The evidence agent did not call {', '.join(missing)} within "
                                f"{EVIDENCE_MAX_CALLS} calls.", kind="model")
            classifier = gathered["run_classifier"]
            if "error" in classifier:
                raise _Fallback("The classifier tool failed.", classifier["error"], "infrastructure")
            info = gathered["get_condition_info"]
            caveats = (info.get("model_limitations") or []) if isinstance(info, dict) else []
            bundle = EvidenceBundle(classifier=ClassifierFacts(**classifier), gradcam=gathered["get_gradcam"],
                                    condition_info=info, dataset_caveat=caveats[0] if caveats else None,
                                    tool_calls=called)
            self._set_stage(span, usage, verdict="complete", reasked=reasked)
        return bundle

    def _write(self, bundle, messages, revision):
        """One writer invocation (a first draft, or a revision): returns (Report, messages)."""
        deadline, limit = self._deadline("writer")
        if messages is None:
            messages = [{"role": "user", "text": writer_prompt(bundle)}]
        else:
            self._handoff("orchestrator", "writer", revision)
            messages = messages + [{"role": "user", "text": revision.prompt()}]
        retries_left = MAX_RETRIES
        with trace_span(self.tracer, "writer_agent", agent="writer", model=self.pipeline.models["writer"],
                        step=self.steps + 1, revision=revision is not None) as span:
            usage = _Usage()
            while True:
                outcome = yield self._request("writer", "json", WRITER_SYSTEM, messages, deadline, limit,
                                              tools_allowed=False, schema_name="report", schema=REPORT_JSON_SCHEMA)
                if isinstance(outcome, Exception):
                    self._llm_failure("writer", outcome, retries_left)
                    retries_left -= 1
                    continue
                turn = outcome
                usage.add(turn)
                if turn.finish_reason in TRUNCATED_FINISH_REASONS:
                    self._truncated("writer", turn, retries_left)
                    retries_left -= 1
                    continue
                report, problem = _parse_draft(turn.text)
                if report is None:
                    # Not a Report at all (bad JSON, schema): a guard rejection like the single agent's
                    self.result.problems.append(problem)
                    self.result.rejected.append({"step": self.steps, "agent": "writer", "text": turn.text,
                                                 "problem": problem})
                    trace_event(self.tracer, "guard", status="reject", error=problem, result="reject",
                                reason=problem, step=self.steps, agent="writer")
                    if retries_left == 0:
                        self._set_stage(span, usage, verdict="invalid")
                        raise _Fallback(f"Report rejected after {MAX_RETRIES} retry: {problem}.", kind="model")
                    retries_left -= 1
                    messages = messages + [{"role": "assistant", "turn": turn},
                                           {"role": "user", "text": f"Your report was rejected: {problem}. "
                                                                    "Reply with a corrected JSON report only."}]
                    continue
                self._set_stage(span, usage, verdict="draft")
                self._last_draft_text = turn.text
                return report, messages + [{"role": "assistant", "turn": turn}]

    def _guard(self, report):
        """check_grounding on a draft (before the verifier). Returns the problem, or None if it passed."""
        problems = check_grounding(report, self.truth)
        problem = "grounding check failed: " + "; ".join(problems) if problems else None
        trace_event(self.tracer, "guard", status="ok" if problem is None else "reject", error=problem,
                    result="pass" if problem is None else "reject", reason=problem, step=self.steps, agent="writer")
        if problem is not None:
            self.result.problems.append(problem)
            self.result.rejected.append({"step": self.steps, "agent": "writer", "text": self._last_draft_text,
                                         "problem": problem})
        return problem

    def _revise(self, request, report, problem):
        """Send the draft back once; with no revision left, the template is used."""
        self.result.revisions.append({"step": self.steps, "by": request.by, "text": self._last_draft_text,
                                      "problems": request.problems,
                                      "unsupported": [f.model_dump() for f in request.unsupported]})
        if self.revisions_left == 0:
            if request.by == "guard":
                raise _Fallback(f"Report rejected after {MAX_REVISIONS} revision: {problem}.", kind="model")
            claims = "; ".join(f"\"{f.claim}\"" for f in request.unsupported)[:300]
            raise _Fallback(f"The verifier still found unsupported claims after {MAX_REVISIONS} revision: {claims}.",
                            kind="model")
        self.revisions_left -= 1
        trace_event(self.tracer, "revision_requested", by=request.by, problems=len(request.problems),
                    unsupported=len(request.unsupported))
        return request

    def _verify(self, bundle, report):
        deadline, limit = self._deadline("verifier")
        messages = [{"role": "user", "text": verifier_prompt(bundle, report)}]
        retries_left = MAX_RETRIES
        with trace_span(self.tracer, "verifier_agent", agent="verifier", model=self.pipeline.models["verifier"],
                        step=self.steps + 1) as span:
            usage = _Usage()
            while True:
                outcome = yield self._request("verifier", "json", VERIFIER_SYSTEM, messages, deadline, limit,
                                              tools_allowed=False, schema_name="verdict", schema=VERDICT_JSON_SCHEMA)
                if isinstance(outcome, Exception):
                    self._llm_failure("verifier", outcome, retries_left)
                    retries_left -= 1
                    continue
                turn = outcome
                usage.add(turn)
                if turn.finish_reason in TRUNCATED_FINISH_REASONS:
                    self._truncated("verifier", turn, retries_left)
                    retries_left -= 1
                    continue
                try:
                    verdict = VerifierVerdict.model_validate_json(_strip_fence(turn.text))
                except ValidationError as e:
                    problem = f"verifier agent: invalid verdict ({e.error_count()} errors)"
                    self.result.problems.append(problem)
                    if retries_left == 0:
                        self._set_stage(span, usage, verdict="invalid")
                        raise _Fallback(f"The verifier's reply was invalid after {MAX_RETRIES} retry.",
                                        problem, "model")
                    retries_left -= 1
                    continue
                self._set_stage(span, usage, verdict="supported" if not verdict.unsupported else "unsupported",
                                claims=len(verdict.findings), unsupported=len(verdict.unsupported))
                return verdict

    def _finish(self, report):
        """The guard has the final say on exactly what is returned (the fixed disclaimer included)."""
        final = report.model_copy(update={"disclaimer": DISCLAIMER})
        problems = check_grounding(final, self.truth)
        trace_event(self.tracer, "final_guard", status="ok" if not problems else "reject",
                    error="; ".join(problems) or None, result="pass" if not problems else "reject")
        if problems:
            raise _Fallback("The final guard rejected the verified report: " + "; ".join(problems) + ".", kind="model")
        self._transition(State.DONE)
        self.result.report, self.result.source = final, "llm"
        self.result.model = self.pipeline.models["writer"]
        return self.result

    def _set_stage(self, span, usage, **attributes):
        safe_set(span, calls=usage.calls, input_tokens=usage.input, output_tokens=usage.output, **attributes)


class _Usage:
    """Tokens of one agent stage, for its span."""

    def __init__(self):
        self.calls, self.input, self.output = 0, 0, 0

    def add(self, turn):
        u = turn.usage or {}
        self.calls += 1
        self.input += u.get("input_tokens") or 0
        self.output += (u.get("output_tokens") or 0) + (u.get("thinking_tokens") or 0)


def _strip_fence(text):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())


def _parse_draft(text):
    """(Report, None) if the text is a schema-valid report, else (None, problem). Grounding is _guard's job."""
    try:
        return Report.model_validate_json(_strip_fence(text)), None
    except ValidationError as e:
        if any(err["type"] == "json_invalid" for err in e.errors()):
            return None, "invalid JSON"
        return None, "schema validation failed: " + "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())


# ---- Drivers ----

def _arm(client, request, clock):
    """Give the client this request's limits: the budget may wait, and the request may run, only until the deadline."""
    left = request.deadline - clock()
    if left <= 0:
        raise AgentTimeoutError(request.agent, request.limit_s)
    client.deadline = time.monotonic() + left
    client.call_timeout_s = left


def _llm_span(tracer, client, request):
    return trace_span(tracer, "llm_call", model=client.name, step=request.step, agent=request.agent,
                      tools_allowed=request.tools_allowed)


def _drive_sync(loop, pipeline, tracer):
    try:
        request = next(loop)
        while True:
            client = pipeline.clients[request.agent]
            try:
                _arm(client, request, pipeline.clock)
                with _llm_span(tracer, client, request) as span:
                    if request.kind == "tools":
                        turn = client.generate(request.system, request.messages, TOOL_DECLARATIONS,
                                               tools_allowed=request.tools_allowed)
                    else:
                        turn = client.generate_json(request.system, request.messages, request.schema_name,
                                                    request.schema)
                    safe_set(span, **llm_call_attributes(turn, client.name))
                outcome = turn
            except Exception as e:
                outcome = e
            request = loop.send(outcome)
    except StopIteration as stop:
        return stop.value


async def _drive_async(loop, pipeline, tracer):
    """As _drive_sync, awaiting agenerate/agenerate_json (a client without them is called inline)."""
    try:
        request = next(loop)
        while True:
            client = pipeline.clients[request.agent]
            try:
                _arm(client, request, pipeline.clock)
                with _llm_span(tracer, client, request) as span:
                    if request.kind == "tools":
                        args = (request.system, request.messages, TOOL_DECLARATIONS)
                        turn = (await client.agenerate(*args, tools_allowed=request.tools_allowed)
                                if hasattr(client, "agenerate") else
                                client.generate(*args, tools_allowed=request.tools_allowed))
                    else:
                        args = (request.system, request.messages, request.schema_name, request.schema)
                        turn = (await client.agenerate_json(*args) if hasattr(client, "agenerate_json")
                                else client.generate_json(*args))
                    safe_set(span, **llm_call_attributes(turn, client.name))
                outcome = turn
            except Exception as e:
                outcome = e
            request = loop.send(outcome)
    except StopIteration as stop:
        return stop.value


def generate_multi_agent_report(tools, pipeline, tracer=None):
    """The multi-agent report (or the template). Traced as a 'report' span (pipeline=multi) holding
    evidence_agent / writer_agent / verifier_agent spans, their llm_call and tool_call spans, and
    handoff, guard, final_guard and fallback_used events."""
    with trace_span(tracer, "report", llm=pipeline.name, pipeline="multi") as span:
        result = _drive_sync(_Orchestrator(tools, pipeline, tracer).run(), pipeline, tracer)
        _close_report_span(span, result, tracer)
    return result


async def agenerate_multi_agent_report(tools, pipeline, tracer=None):
    """Async generate_multi_agent_report (eval runner): identical decisions, LLM calls awaited."""
    with trace_span(tracer, "report", llm=pipeline.name, pipeline="multi") as span:
        result = await _drive_async(_Orchestrator(tools, pipeline, tracer).run(), pipeline, tracer)
        _close_report_span(span, result, tracer)
    return result

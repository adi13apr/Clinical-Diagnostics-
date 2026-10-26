"""Tool-calling report agent.

The LLM gathers facts through tools and writes a Report. The classifier stays the
source of truth: a report whose class, probability or status differs from
run_classifier's output is rejected, and after one retry a deterministic template
report is used instead. The LLM never changes the prediction.

Only tool outputs are sent to the LLM: no image pixels and no patient details.
"""
import asyncio
import dataclasses
import hashlib
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pydantic import ValidationError

from agent.llm_config import load_llm_config, provider_settings, request_options
from agent.rate_limit import BUDGETS, BudgetWaitTooLong, DailyCapReached, estimate_tokens
from agent.schema import DISCLAIMER, Report
from agent.tools import TOOL_DECLARATIONS, ReportTools
from services.calibration import REFER_MESSAGE, STATUS_REFER
from services.tracing import estimate_cost, safe_set, trace_event, trace_span

GEMINI_MODEL = "gemini-3.8-flash"  # stable per https://ai.google.dev/gemini-api/docs/models
MAX_STEPS = 5   # LLM calls per report, including the retry
MAX_RETRIES = 1  # re-asks after an invalid or ungrounded report
RATE_LIMIT_MAX_RETRY_DELAY = 10.0  # seconds; on a 429, wait and retry once only if Google asks for less
DEFAULT_REQUEST_TIMEOUT_S = 120.0  # a request gives up after this; counts as infrastructure
# finish_reason of a reply cut off at the output-token limit (OpenAI-compatible, Gemini)
TRUNCATED_FINISH_REASONS = {"length", "MAX_TOKENS"}

BANNED_PATTERNS = [r"diagnosis\s+(is\s+|has\s+been\s+)?confirmed", r"confirmed\s+diagnosis"]
# A banned phrase is allowed when its own clause negates it within a few words, e.g. "not a confirmed
# diagnosis" or "until the diagnosis is confirmed". "No doubt"-style phrases are not negations.
NEGATION_CUES = re.compile(r"\b(not|no|never|without|cannot|nor|until|unless|before)\b|n't\b")
NOT_NEGATIONS = re.compile(r"\b(no|without|beyond)\s+(a\s+)?(doubt|question)\b|\bnot\s+only\b")
CLAUSE_BREAK = re.compile(r"[.;:!?,]|\bbut\b|\bhowever\b")
NEGATION_WINDOW_WORDS = 4

# "No (immediate) referral/action/follow-up is required/needed": never acceptable for a Pneumonia/TB
# prediction (human review flagged c32; the LLM judge did not). Tolerates any hyphen in "follow-up"
# and one extra word ("no immediate imaging follow-up is required").
DISMISSIVE_NEXT_STEP = re.compile(
    r"\bno\s+(immediate\s+)?(\w+\s+)?(referral|action|follow[\s\-‐-―]?up|further\s+(action|evaluation))\b"
    r"[^.]{0,40}?\b(is\s+|are\s+)?(required|needed|necessary)\b", re.IGNORECASE)
CLINICIAN_FOLLOW_UP = "Arrange clinician follow-up for this result."

SYSTEM_PROMPT = f"""You write short, plain-language reports about an AI chest X-ray classifier's result.

Rules:
1. Call each of the three tools once, with no arguments: run_classifier, get_gradcam and
   get_condition_info (it defaults to the predicted class). If you can make parallel function calls,
   call all three in your first turn. Then write the report.
2. run_classifier's output is authoritative. Copy predicted_class, calibrated_probability and status
   into the report EXACTLY as returned. Never change, round or reinterpret them.
3. Use get_gradcam to describe which approximate region influenced the model. Hedge: it is a coarse
   zone, it shows model attention rather than confirmed anatomy, and if focus is "diffuse" say so.
4. Use only get_condition_info's text for medical background. Do not add facts from elsewhere.
   Never say or imply the diagnosis is confirmed. The classifier is a screening aid, not a diagnosis.
5. Only if status is "refer": the recommendation must begin with "{REFER_MESSAGE.capitalize()}." and say
   that clinician review is needed. If status is "confident": never call the result uncertain and never
   use that refer sentence. Separately, whatever the status, if predicted_class is Pneumonia or
   Tuberculosis the recommendation must advise clinician follow-up and must never say that no referral,
   action or follow-up is needed.
6. summary: 2-3 plain-language sentences for a non-specialist. recommendation: next step for the
   reader. limitations: short bullet strings taken from get_condition_info (what_this_result_means
   and model_limitations); don't invent others.

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
    # time.monotonic() after which a call must not wait for the rate-limit budget (set for report jobs,
    # so they fall back to the template before RQ's job timeout); None = wait as long as needed
    deadline = None
    # Seconds the next request may take on the network (the multi-agent pipeline's per-agent timeouts);
    # None = the client's own request timeout
    call_timeout_s = None
    pipeline = "single"   # agent.multi_agent.MultiAgentPipeline says "multi"
    pipeline_note = None  # why a requested multi pipeline runs as single (build_report_llm)

    @abstractmethod
    def generate(self, system, messages, tools, tools_allowed=True):
        """Return the next LLMTurn. tools_allowed=False forbids tool calls (final-answer turn)."""

    def budget_wait_left(self):
        """Seconds this call may wait for the budget (max_wait), or None without a deadline."""
        return None if self.deadline is None else max(self.deadline - time.monotonic(), 0.0)


class RateLimitError(Exception):
    """The LLM provider returned 429 (or a client-side cap was hit) and we didn't wait it out."""

    def __init__(self, retry_delay=None, quota_id=None, quota_limit=None, provider="Gemini",
                 resets="midnight Pacific time"):
        super().__init__(f"rate limited (quota {quota_id or 'unknown'}, limit {quota_limit}, "
                         f"retry delay {retry_delay})")
        self.retry_delay, self.quota_id, self.quota_limit = retry_delay, quota_id, quota_limit
        self.provider, self.resets = provider, resets

    @property
    def daily(self):
        q = self.quota_id or ""
        return "PerDay" in q or "per day" in q.lower()

    @classmethod
    def from_daily_cap(cls, cap, provider):
        """Client-side daily cap from config/llm.json (agent/rate_limit.py)."""
        return cls(None, f"client daily {cap.what} cap for {cap.model} (per day)", f"{cap.cap} {cap.what}s"
                   if cap.what == "token" else cap.cap, provider=provider, resets="midnight UTC (client cap)")


class BudgetDeadlineError(RateLimitError):
    """The shared rate-limit budget had no room before the report's deadline, so the LLM wasn't called."""

    def __init__(self, wait, provider):
        super().__init__(wait, "client rate-limit budget (report deadline)", None, provider=provider,
                         resets="the next minute")


def quota_info(error):
    """(quota_id, limit) from a 429's QuotaFailure detail, e.g. ('GenerateRequestsPerDay...-FreeTier', '20')."""
    body = getattr(error, "details", None)
    details = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
    for d in details:
        if str(d.get("@type", "")).endswith("QuotaFailure"):
            for v in d.get("violations", []):
                return v.get("quotaId"), v.get("quotaValue")
    return None, None


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
    if isinstance(error, BudgetDeadlineError):
        return f"{error.provider} rate limit: no room in the shared token budget before the report's time limit."
    if isinstance(error, RateLimitError):
        if error.daily:
            limit = error.quota_limit
            unit = "tokens" if "token" in (error.quota_id or "").lower() else "requests"
            limit = "" if not limit else f" of {limit}" if not str(limit).isdigit() else f" of {limit} {unit}"
            return f"{error.provider} daily quota{limit} reached (resets at {error.resets})."
        wait = f" (retry in {error.retry_delay:.0f} s)" if error.retry_delay else ""
        return f"{error.provider} rate limit reached{wait}."
    if hasattr(error, "code") and hasattr(error, "status") and not hasattr(error, "status_code"):
        return f"Gemini error {error.code} {error.status}."  # google.genai.errors.APIError
    if hasattr(error, "status_code"):  # openai.APIStatusError (Groq and other OpenAI-compatible APIs)
        if _is_tool_use_failed(error):
            return "The model produced an invalid tool call."
        if _is_bad_generation(error, "json_validate_failed"):
            return "The model's JSON output didn't match the schema (or was cut off)."
        return f"LLM API error {error.status_code}."
    return f"LLM call failed ({type(error).__name__})."


def _is_tool_use_failed(error):
    return _is_bad_generation(error, "tool_use_failed")


def _is_bad_generation(error, code_name):
    """A 400 with this error code: the model's output was malformed (not the request or the service)."""
    body = getattr(error, "body", None)
    code = (body.get("error", body) if isinstance(body, dict) else {}).get("code") if isinstance(body, dict) else None
    return getattr(error, "status_code", None) == 400 and (code == code_name or code_name in str(error))


def classify_llm_error(error):
    """'rate_limit', 'infrastructure' (network, auth, server, unknown) or 'model' (the model misbehaved).

    Eval runs stop on rate_limit/infrastructure (retry later) but record 'model' failures as results.
    """
    if isinstance(error, RateLimitError):
        return "rate_limit"
    # json_validate_failed: strict json_schema output that didn't validate, e.g. cut off at
    # max_completion_tokens (the multi-agent writer/verifier; the single agent never asks for it)
    if (_is_tool_use_failed(error) or _is_bad_generation(error, "json_validate_failed")
            or isinstance(error, json.JSONDecodeError)):
        return "model"
    return "infrastructure"


class GeminiClient(LLMClient):
    def __init__(self, api_key, model=GEMINI_MODEL, client=None, sleep=time.sleep, budget=None):
        from google import genai
        from google.genai import errors, types
        self.types = types
        self.errors = errors
        self.model = model
        self.name = model
        self.client = client or genai.Client(api_key=api_key)
        self.sleep = sleep
        self.budget = budget  # agent.rate_limit.ModelBudget shared per model, or None

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

    def generate(self, system, messages, tools, tools_allowed=True):
        t = self.types
        config = t.GenerateContentConfig(
            system_instruction=system,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=d["name"], description=d["description"],
                                      parameters_json_schema=d["parameters"]) for d in tools])],
            tool_config=None if tools_allowed else t.ToolConfig(
                function_calling_config=t.FunctionCallingConfig(mode="NONE")),
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=t.ThinkingConfig(thinking_level="low"),
        )
        contents = self._contents(messages)
        retries = 0
        for attempt in (1, 2):
            reservation = self._reserve(system, messages, tools)
            try:
                response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                meta = response.usage_metadata
                if self.budget is not None and meta is not None:
                    self.budget.record(reservation, meta.total_token_count)
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
                raise RateLimitError(delay, *quota_info(e)) from e
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

    def _reserve(self, system, messages, tools):
        if self.budget is None:
            return None
        try:
            return self.budget.acquire(estimate_tokens(system, [m.get("text") for m in messages], tools) + 1000,
                                       max_wait=self.budget_wait_left())
        except DailyCapReached as cap:
            raise RateLimitError.from_daily_cap(cap, "Gemini") from cap
        except BudgetWaitTooLong as e:
            raise BudgetDeadlineError(e.wait, "Gemini") from e


class OpenAICompatibleClient(LLMClient):
    """Chat Completions API with tool calling (Groq, or any OpenAI-compatible endpoint).

    generate() is the sync path (the app). agenerate() is the async path (the eval runner): same
    request, same 429 rule, same parsing, but it awaits AsyncOpenAI, waits with asyncio.sleep, holds
    the model's asyncio.Semaphore slot while in flight, and gives up after timeout_s.
    """

    def __init__(self, api_key, model, base_url, client=None, sleep=time.sleep, budget=None, options=None,
                 provider="Groq", async_client=None, async_sleep=asyncio.sleep, timeout_s=DEFAULT_REQUEST_TIMEOUT_S):
        if client is None:
            from openai import OpenAI
            # retries are handled here; the timeout stops a hung request (report jobs wait longer than it)
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout_s)
        self.client, self.model, self.name = client, model, model
        self.sleep, self.budget, self.provider = sleep, budget, provider
        self.options = dict(options or {})
        self._api_key, self._base_url = api_key, base_url
        self._async_client, self.async_sleep, self.timeout_s = async_client, async_sleep, timeout_s

    def _messages(self, system, messages):
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                out.append(m["turn"].raw)
            else:
                out += [{"role": "tool", "tool_call_id": call.id, "name": call.name,
                         "content": json.dumps(result, default=str)} for call, result in m["results"]]
        return out

    def _prepare(self, system, messages, tools, tools_allowed):
        payload = self._messages(system, messages)
        tool_specs = [{"type": "function", "function": {"name": d["name"], "description": d["description"],
                                                        "parameters": d["parameters"]}} for d in tools]
        options = dict(self.options, **({} if tools_allowed else {"tool_choice": "none"}))
        return payload, tool_specs, options

    def _rate_limit_error(self, error, delay):
        return RateLimitError(delay, *openai_quota_info(error), provider=self.provider,
                              resets="the provider's daily window")

    def generate(self, system, messages, tools, tools_allowed=True):
        payload, tool_specs, options = self._prepare(system, messages, tools, tools_allowed)
        return self._complete(payload, tool_specs, {"tools": tool_specs, **split_request_options(options)})

    async def agenerate(self, system, messages, tools, tools_allowed=True):
        payload, tool_specs, options = self._prepare(system, messages, tools, tools_allowed)
        return await self._acomplete(payload, tool_specs, {"tools": tool_specs, **split_request_options(options)})

    def generate_json(self, system, messages, schema_name, schema):
        """A turn whose text is JSON matching schema (strict json_schema), with no tools: the multi-agent
        pipeline's writer and verifier. Same budget, 429 rule and parsing as generate()."""
        payload, kwargs = self._prepare_json(system, messages, schema_name, schema)
        return self._complete(payload, [], kwargs)

    async def agenerate_json(self, system, messages, schema_name, schema):
        payload, kwargs = self._prepare_json(system, messages, schema_name, schema)
        return await self._acomplete(payload, [], kwargs)

    def _prepare_json(self, system, messages, schema_name, schema):
        response_format = {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        # The schema is part of the request, so it counts towards the prompt estimate (passed as "tool specs")
        return self._messages(system, messages), {"response_format": response_format,
                                                  **split_request_options(self.options)}

    def _complete(self, payload, tool_specs, kwargs):
        import openai
        retries = 0
        if self.call_timeout_s is not None:
            kwargs = dict(kwargs, timeout=min(self.call_timeout_s, self.timeout_s))
        size_parts = tool_specs or kwargs.get("response_format") or []
        for attempt in (1, 2):
            reservation = self._reserve(payload, size_parts)
            try:
                response = self.client.chat.completions.create(model=self.model, messages=payload, **kwargs)
                break
            except openai.RateLimitError as e:
                delay = openai_retry_after(e)
                if attempt == 1 and delay is not None and delay < RATE_LIMIT_MAX_RETRY_DELAY:
                    self.sleep(delay)
                    retries += 1
                    continue
                raise self._rate_limit_error(e, delay) from e
        if self.budget is not None and response.usage is not None:
            self.budget.record(reservation, response.usage.total_tokens, response.usage.completion_tokens)
        return self._parse(response, retries)

    async def _acomplete(self, payload, tool_specs, kwargs):
        import openai
        retries = 0
        timeout = self.timeout_s if self.call_timeout_s is None else min(self.call_timeout_s, self.timeout_s)
        size_parts = tool_specs or kwargs.get("response_format") or []
        for attempt in (1, 2):
            reservation = await self._areserve(payload, size_parts)
            try:
                async with (self.budget.slot() if self.budget is not None else nullcontext()):
                    response = await asyncio.wait_for(
                        self._async().chat.completions.create(model=self.model, messages=payload, **kwargs),
                        timeout)
                break
            except openai.RateLimitError as e:
                delay = openai_retry_after(e)
                if attempt == 1 and delay is not None and delay < RATE_LIMIT_MAX_RETRY_DELAY:
                    await self.async_sleep(delay)
                    retries += 1
                    continue
                raise self._rate_limit_error(e, delay) from e
        if self.budget is not None and response.usage is not None:
            await self.budget.arecord(reservation, response.usage.total_tokens, response.usage.completion_tokens)
        return self._parse(response, retries)

    def _async(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url, max_retries=0)
        return self._async_client

    def _parse(self, response, retries):
        usage = response.usage
        choice = response.choices[0]
        message = choice.message
        calls = []
        for tc in message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}  # malformed arguments: our tools take none, so let the call through
            calls.append(ToolCall(name=tc.function.name, args=args if isinstance(args, dict) else {}, id=tc.id))
        raw = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            raw["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                 for tc in message.tool_calls]
        return LLMTurn(text=None if calls else (message.content or ""), tool_calls=calls, raw=raw,
                       usage=openai_usage(usage), finish_reason=choice.finish_reason, rate_limit_retries=retries)

    def _reservation_size(self, payload, tool_specs):
        max_out = int(self.options.get("max_completion_tokens", 1000))
        return estimate_tokens(payload, tool_specs) + max_out, max_out

    def _reserve(self, payload, tool_specs):
        if self.budget is None:
            return None
        try:
            return self.budget.acquire(*self._reservation_size(payload, tool_specs), max_wait=self.budget_wait_left())
        except DailyCapReached as cap:
            raise RateLimitError.from_daily_cap(cap, self.provider) from cap
        except BudgetWaitTooLong as e:
            raise BudgetDeadlineError(e.wait, self.provider) from e

    async def _areserve(self, payload, tool_specs):
        if self.budget is None:
            return None
        try:
            return await self.budget.aacquire(*self._reservation_size(payload, tool_specs),
                                              max_wait=self.budget_wait_left())
        except DailyCapReached as cap:
            raise RateLimitError.from_daily_cap(cap, self.provider) from cap
        except BudgetWaitTooLong as e:
            raise BudgetDeadlineError(e.wait, self.provider) from e


# Parameters the openai SDK accepts as keyword arguments; provider-specific ones (Groq's
# include_reasoning, reasoning_format, ...) must go through extra_body.
OPENAI_STANDARD_PARAMS = {"max_completion_tokens", "max_tokens", "reasoning_effort", "temperature", "top_p",
                          "seed", "stop", "parallel_tool_calls", "tool_choice"}


def split_request_options(options):
    """Request options as create() kwargs, with non-standard keys moved into extra_body."""
    kwargs = {k: v for k, v in options.items() if k in OPENAI_STANDARD_PARAMS}
    extra = {k: v for k, v in options.items() if k not in OPENAI_STANDARD_PARAMS}
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def openai_usage(usage):
    """{input, output (excluding reasoning), thinking} so output + thinking = billed completion tokens."""
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
    completion = usage.completion_tokens
    return {"input_tokens": usage.prompt_tokens,
            "output_tokens": completion - reasoning if reasoning is not None else completion,
            "thinking_tokens": reasoning}


def openai_retry_after(error):
    """Seconds from the 429's retry-after header, else 'try again in 1m2.5s' in the message, or None."""
    response = getattr(error, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(error))
    return (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else None


def openai_quota_info(error):
    """(quota description, limit) from Groq's 429 message, e.g. ('tokens per day (TPD)', '200000')
    or ('output tokens per minute (OTPM)', '1000')."""
    m = re.search(r"on ([a-z ]*?(?:tokens|requests)) per (minute|day) \((\w+)\): Limit (\d+)", str(error))
    return (f"{m.group(1)} per {m.group(2)} ({m.group(3)})", m.group(4)) if m else (None, None)


class MockLLMClient(LLMClient):
    """Scripted client for tests. Each script item is an LLMTurn or a callable(messages) -> LLMTurn."""
    name = "mock"

    def __init__(self, script, delay_s=0.0, name="mock"):
        self.script = list(script)
        self.calls = []
        self.tools_allowed = []
        self.systems = []   # system prompt of each call
        self.schemas = []   # schema name of each generate_json call (None for generate)
        self.delay_s = delay_s  # simulated network latency (benchmarks/tests)
        self.name = name

    def generate(self, system, messages, tools, tools_allowed=True):
        if self.delay_s:
            time.sleep(self.delay_s)
        return self._next(system, messages, tools_allowed)

    async def agenerate(self, system, messages, tools, tools_allowed=True):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self._next(system, messages, tools_allowed)

    def generate_json(self, system, messages, schema_name, schema):
        if self.delay_s:
            time.sleep(self.delay_s)
        return self._next(system, messages, False, schema_name)

    async def agenerate_json(self, system, messages, schema_name, schema):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self._next(system, messages, False, schema_name)

    def _next(self, system, messages, tools_allowed, schema_name=None):
        self.calls.append(list(messages))
        self.tools_allowed.append(tools_allowed)
        self.systems.append(system)
        self.schemas.append(schema_name)
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
    error_kind: Optional[str] = None    # classify_llm_error(): rate_limit / infrastructure / model
    cached: bool = False                 # served from the report cache, no LLM requests spent
    pipeline: str = "single"             # the pipeline that actually ran: "single" or "multi"
    pipeline_note: Optional[str] = None  # why a requested multi pipeline ran as single
    # Multi-agent pipeline only:
    agent_models: dict = field(default_factory=dict)   # agent -> model ({"evidence", "writer", "verifier"})
    revisions: List[dict] = field(default_factory=list)  # drafts sent back to the writer, with why (guard/verifier)
    findings: List[dict] = field(default_factory=list)   # the verifier's findings on the final draft

    def source_line(self):
        """One line saying who wrote the report, for the UI and PDF (the fallback reason is UI-only)."""
        if self.source == "llm" and self.pipeline == "multi":
            return (f"Written by {self.model} from the classifier's output in the multi-agent pipeline; its "
                    f"claims were checked by {self.agent_models.get('verifier', 'a verifier')}, and class, "
                    "probability and status were checked against the classifier.")
        if self.source == "llm":
            return (f"Written by {self.model} from the classifier's output; class, probability and "
                    "status were checked against the classifier.")
        return "Standard template report generated from the classifier's output."


def build_llm_client(role="report", model=None, config=None, budgets=None):
    """Client for the configured provider (config/llm.json, env LLM_PROVIDER), or None without a key.

    role picks report_model or judge_model; model overrides it. Each model gets its shared budget from
    budgets (agent.rate_limit.BudgetRegistry): this process's in-memory BUDGETS by default, a Redis
    registry in the API when REDIS_URL is set.
    """
    config = config or load_llm_config()
    name, settings = provider_settings(config)
    if not settings["api_key"]:
        return None
    model = model or settings[f"{role}_model"]
    budget = (BUDGETS if budgets is None else budgets).get(model, config["limits"].get(model, {}))
    if settings["kind"] == "gemini":
        return GeminiClient(settings["api_key"], model, budget=budget)
    return OpenAICompatibleClient(settings["api_key"], model, settings["base_url"], budget=budget,
                                  options=request_options(model, config), provider=name.capitalize(),
                                  timeout_s=config.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S))


def build_report_llm(pipeline=None, config=None, budgets=None):
    """What writes reports: the single agent's LLMClient, or a MultiAgentPipeline. None without a key.

    pipeline is "single" or "multi" (default: env REPORT_PIPELINE, else config/llm.json report_pipeline).
    Multi needs an OpenAI-compatible provider (strict json_schema for the writer and verifier); with
    another one (gemini) the single agent runs, and pipeline_note says why, so traces and eval records
    show the pipeline that actually ran. Each agent gets its own client; agents on the same model share
    that model's budget from budgets (as build_llm_client).
    """
    from agent.llm_config import AGENT_ROLES, agent_request_options, agent_timeout_s, pipeline_mode
    config = config or load_llm_config()
    mode = pipeline or pipeline_mode(config)
    if mode == "single":
        return build_llm_client("report", config=config, budgets=budgets)
    name, settings = provider_settings(config)
    if settings["kind"] != "openai_compatible":
        llm = build_llm_client("report", config=config, budgets=budgets)
        if llm is not None:
            llm.pipeline_note = (f"REPORT_PIPELINE=multi needs an OpenAI-compatible provider; {name} ran the "
                                 "single agent.")
        return llm
    clients = {agent: build_llm_client(model=settings[f"{agent}_model"], config=config, budgets=budgets)
               for agent in AGENT_ROLES}
    if any(client is None for client in clients.values()):
        return None
    for agent, client in clients.items():  # e.g. the writer's larger max_completion_tokens
        client.options = agent_request_options(agent, client.model, config)
    from agent.multi_agent import MultiAgentPipeline
    return MultiAgentPipeline(**clients, timeouts={agent: agent_timeout_s(agent, config) for agent in AGENT_ROLES})


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


def check_grounding(report, truth, caveat=None):
    """Problems that make a schema-valid report unacceptable (empty list = grounded).

    caveat: the dataset/source-confound caveat (get_condition_info's first model_limitations item);
    when given, the limitations must contain it word for word.
    """
    problems = []
    for key in ("predicted_class", "calibrated_probability", "status"):
        if getattr(report, key) != truth[key]:
            problems.append(f"{key} is {getattr(report, key)!r} but run_classifier returned {truth[key]!r}")
    text = " ".join([report.summary, report.recommendation, *report.limitations, report.disclaimer])
    if asserts_confirmed_diagnosis(text):
        problems.append("the report must never say the diagnosis is confirmed")
    if truth["status"] == STATUS_REFER and "clinician review" not in report.recommendation.lower():
        problems.append("status is 'refer', so the recommendation must say clinician review is recommended")
    if truth["status"] != STATUS_REFER and labels_confident_as_refer(report):
        problems.append(f"status is '{truth['status']}', so the report must not call the result uncertain or use "
                        f"the refer sentence \"{REFER_MESSAGE.capitalize()}\"")
    if dismisses_abnormal_result(truth["predicted_class"], report.recommendation):
        problems.append(f"predicted_class is {truth['predicted_class']}, so the recommendation must not say "
                        "no referral, action or follow-up is needed; advise clinician follow-up")
    if caveat and not includes_caveat(report.limitations, caveat):
        problems.append("the limitations must include this caveat from get_condition_info's model_limitations, "
                        f"copied word for word: \"{caveat}\"")
    return problems


HYPHENS = str.maketrans(dict.fromkeys("‐‑‒–—", "-"))


def _normalize(text):
    return " ".join(text.translate(HYPHENS).split()).lower()


def includes_caveat(limitations, caveat):
    """True if one limitation contains the caveat verbatim (ignoring case, whitespace and hyphen variants).

    Added after the 20261020T075220Z run: c07 skipped get_condition_info and invented its own
    limitations, dropping the dataset/source-confound caveat; the guard accepted it and the judge
    scored its faithfulness 5. The eval check dataset_caveat_present uses the same rule.
    """
    return any(_normalize(caveat) in _normalize(item) for item in limitations)


def labels_confident_as_refer(report):
    """True if a report uses the refer sentence ("Refer: uncertain…"). Only refer cases may.

    Added after the 2026-10-20 run: a prompt change made 5 of 8 confident Pneumonia/TB reports open with
    the refer sentence (calling a 99.99% result uncertain); the judge caught it, the guard didn't.
    """
    text = f"{report.summary} {report.recommendation}".lower()
    return REFER_MESSAGE.lower() in text or report.recommendation.lower().lstrip().startswith("refer:")


def dismisses_abnormal_result(predicted_class, recommendation):
    """True if a Pneumonia/TB recommendation says no referral/action/follow-up is needed (Normal is exempt)."""
    return predicted_class != "Normal" and bool(DISMISSIVE_NEXT_STEP.search(recommendation))


def parse_report(text, truth, caveat=None):
    """(Report, None) if the text is a valid, grounded report, else (None, problem). caveat: see check_grounding."""
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
    problems = check_grounding(report, truth, caveat)
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
    if cls != "Normal":
        recommendation += f" {CLINICIAN_FOLLOW_UP}"  # Pneumonia/TB always get explicit clinician follow-up

    report = Report(
        predicted_class=cls,
        calibrated_probability=prob,
        status=truth["status"],
        summary=" ".join(sentences),
        recommendation=recommendation,
        limitations=[info["what_this_result_means"], *info["model_limitations"]],  # all from the tool output
        disclaimer=DISCLAIMER,
    )
    problems = check_grounding(report, truth, info["model_limitations"][0])
    assert not problems, problems
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
        result = _drive_sync(_report_loop(tools, llm, max_steps, max_retries, tracer), llm, tracer)
        result.pipeline_note = getattr(llm, "pipeline_note", None)
        _close_report_span(span, result, tracer)
    return result


async def agenerate_report(tools, llm, max_steps=MAX_STEPS, max_retries=MAX_RETRIES, tracer=None):
    """Async generate_report (eval runner): identical decisions, LLM calls awaited."""
    with trace_span(tracer, "report", llm=getattr(llm, "name", None)) as span:
        result = await _drive_async(_report_loop(tools, llm, max_steps, max_retries, tracer), llm, tracer)
        result.pipeline_note = getattr(llm, "pipeline_note", None)
        _close_report_span(span, result, tracer)
    return result


def run_report(tools, llm, tracer=None):
    """The report from whichever pipeline llm is: a MultiAgentPipeline runs the multi-agent orchestrator,
    anything else (an LLMClient or None) the single tool-calling agent."""
    if getattr(llm, "pipeline", "single") == "multi":
        from agent.multi_agent import generate_multi_agent_report
        return generate_multi_agent_report(tools, llm, tracer=tracer)
    return generate_report(tools, llm, tracer=tracer)


async def arun_report(tools, llm, tracer=None):
    """Async run_report (eval runner)."""
    if getattr(llm, "pipeline", "single") == "multi":
        from agent.multi_agent import agenerate_multi_agent_report
        return await agenerate_multi_agent_report(tools, llm, tracer=tracer)
    return await agenerate_report(tools, llm, tracer=tracer)


def _close_report_span(span, result, tracer):
    trace_event(tracer, "fallback_used", used=result.source == "fallback", reason=result.reason)
    safe_set(span, source=result.source, steps=result.steps, rejected=len(result.rejected),
             pipeline=result.pipeline, pipeline_note=result.pipeline_note)


@dataclass
class LLMRequest:
    """One LLM call the report loop needs made."""
    step: int
    messages: list
    tools_allowed: bool


def _drive_sync(loop, llm, tracer):
    """Run the report loop, making each requested LLM call synchronously."""
    try:
        request = next(loop)
        while True:
            try:
                with _llm_span(tracer, llm, request) as span:
                    turn = llm.generate(SYSTEM_PROMPT, request.messages, TOOL_DECLARATIONS,
                                        tools_allowed=request.tools_allowed)
                    safe_set(span, **llm_call_attributes(turn, llm.name))
                outcome = turn
            except Exception as e:
                outcome = e
            request = loop.send(outcome)
    except StopIteration as stop:
        return stop.value


async def _drive_async(loop, llm, tracer):
    """Run the report loop, awaiting each LLM call (agenerate) so other coroutines run meanwhile.

    A client without agenerate (e.g. GeminiClient) is called synchronously, inline: no threads.
    """
    try:
        request = next(loop)
        while True:
            try:
                with _llm_span(tracer, llm, request) as span:
                    if hasattr(llm, "agenerate"):
                        turn = await llm.agenerate(SYSTEM_PROMPT, request.messages, TOOL_DECLARATIONS,
                                                   tools_allowed=request.tools_allowed)
                    else:
                        turn = llm.generate(SYSTEM_PROMPT, request.messages, TOOL_DECLARATIONS,
                                            tools_allowed=request.tools_allowed)
                    safe_set(span, **llm_call_attributes(turn, llm.name))
                outcome = turn
            except Exception as e:
                outcome = e
            request = loop.send(outcome)
    except StopIteration as stop:
        return stop.value


def _llm_span(tracer, llm, request):
    return trace_span(tracer, "llm_call", model=llm.name, step=request.step, tools_allowed=request.tools_allowed)


def _report_loop(tools, llm, max_steps, max_retries, tracer):
    """The report agent's decisions, shared by the sync and async drivers.

    A generator: it yields an LLMRequest whenever it needs the model, receives the outcome (an LLMTurn
    or the exception the call raised) via send(), and returns the ReportResult. All retry, guard and
    fallback logic lives here, so both drivers behave identically.
    """
    truth = tools.run_classifier()  # ground truth for the guard, independent of what the LLM calls
    # Every report must carry the dataset/source-confound caveat, even if the LLM skips get_condition_info
    caveat = tools.get_condition_info()["model_limitations"][0]
    result = ReportResult(report=None, source="fallback")

    def fall_back(reason, detail=None, kind=None):
        result.report, result.source, result.reason = fallback_report(tools), "fallback", reason
        result.error_detail, result.error_kind = detail, kind
        return result

    if llm is None:
        return fall_back("No LLM configured (set an API key for the provider in config/llm.json, "
                         "e.g. st.secrets['groq']['api_key'] or GROQ_API_KEY).")

    messages = [{"role": "user", "text": USER_PROMPT}]
    retries_left = max_retries
    all_tools = {d["name"] for d in TOOL_DECLARATIONS}
    for step in range(1, max_steps + 1):
        result.steps = step
        # Once every tool has been called the model has all the facts: forbid further tool calls so the
        # turn can only be the final answer (avoids malformed tool calls such as Groq's tool_use_failed)
        tools_allowed = not all_tools <= set(result.tool_calls)
        outcome = yield LLMRequest(step, messages, tools_allowed)
        if isinstance(outcome, Exception):
            e = outcome
            detail = f"{type(e).__name__}: {e}" + (f" <- {e.__cause__}" if e.__cause__ else "")
            kind = classify_llm_error(e)
            if kind == "model" and retries_left > 0:
                # The model misbehaved at the API level (e.g. invalid tool call): retry once, like a guard rejection
                retries_left -= 1
                result.problems.append(describe_llm_error(e))
                result.rejected.append({"step": step, "text": None, "problem": describe_llm_error(e),
                                        "error_detail": detail[:500]})
                continue
            return fall_back(describe_llm_error(e), detail, kind)
        turn = outcome
        if turn.finish_reason in TRUNCATED_FINISH_REASONS:
            # Cut off at max_completion_tokens: a partial report or tool call. Never replay it; retry once.
            problem = f"the reply was cut off at the output-token limit (finish_reason {turn.finish_reason!r})"
            trace_event(tracer, "llm_truncated", status="error", error=problem, step=step,
                        finish_reason=turn.finish_reason, output_tokens=(turn.usage or {}).get("output_tokens"))
            result.problems.append(problem)
            result.rejected.append({"step": step, "text": turn.text, "problem": problem,
                                    "finish_reason": turn.finish_reason})
            if retries_left == 0:
                return fall_back(f"Report cut off at the output-token limit after {max_retries} retry.",
                                 problem, "model")
            retries_left -= 1
            messages.append({"role": "user", "text": "Your previous reply was cut off at the length limit and "
                                                     "was discarded. Try again, keeping it brief."})
            continue
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

        report, problem = parse_report(turn.text, truth, caveat)
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
REDIS_REPORT_TTL_S = 24 * 3600


def report_result_to_json(result):
    """ReportResult as JSON text: every dataclass field, the Report via its Pydantic dump (no pickle)."""
    data = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    data["report"] = result.report.model_dump(mode="json")
    return json.dumps(data)


def report_result_from_json(text):
    """Inverse of report_result_to_json; the Report is validated against the schema again."""
    data = json.loads(text)
    data["report"] = Report.model_validate(data["report"])
    return ReportResult(**data)


class RedisReportCache:
    """ReportCache shared between API replicas through Redis: same get/put, entries expire after ttl_s.

    Key: {prefix}report:v2:{pipeline}:{image fingerprint}:{temperature}:{model}; value:
    report_result_to_json(). A value that no longer parses (old format, corrupt) is a miss, and so is a
    Redis error: the cache only saves LLM requests, so the report is written anyway. on_error(operation,
    exc) hears about both.
    """

    VERSION = "v2"  # bump when the stored format changes (v2: pipeline in the key, multi-agent fields)

    def __init__(self, client, ttl_s=REDIS_REPORT_TTL_S, prefix="cdp:", on_error=None):
        self.client, self.ttl_s, self.prefix, self.on_error = client, ttl_s, prefix, on_error

    def key(self, key):
        fingerprint, temperature, model, pipeline = key
        return f"{self.prefix}report:{self.VERSION}:{pipeline}:{fingerprint}:{float(temperature)!r}:{model}"

    def get(self, key):
        try:
            raw = self.client.get(self.key(key))
            return None if raw is None else report_result_from_json(raw)
        except Exception as e:  # redis.RedisError, or a value that doesn't parse (agent/ doesn't import redis)
            self._failed("get", e)
            return None

    def put(self, key, value):
        try:
            self.client.set(self.key(key), report_result_to_json(value), px=int(self.ttl_s * 1000))
        except Exception as e:
            self._failed("put", e)

    def _failed(self, operation, error):
        if self.on_error is not None:
            self.on_error(operation, error)

    def __len__(self):
        return sum(1 for _ in self.client.scan_iter(match=f"{self.prefix}report:{self.VERSION}:*"))


def image_fingerprint(image):
    """SHA-256 of the decoded pixels (plus mode and size), so the same scan hashes the same however it was uploaded."""
    h = hashlib.sha256(f"{image.mode}|{image.size}|".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def generate_report_for_image(model_service, image, prediction=None, llm=None, cache=REPORT_CACHE, tracer=None):
    """Dashboard entry point. Reuses a cached LLM report for the same image, temperature, model and pipeline.

    llm is an LLMClient (the single agent) or a MultiAgentPipeline (build_report_llm). Only LLM reports
    are cached: a fallback (e.g. after a rate limit) costs nothing to rebuild, and caching it would stop
    a later click from trying the LLM again.
    """
    key = None
    if llm is not None and cache is not None:
        pipeline = getattr(llm, "pipeline", "single")
        key = (image_fingerprint(image), float(model_service.temperature), llm.name, pipeline)
        hit = cache.get(key)
        if hit is not None:
            with trace_span(tracer, "report", llm=llm.name, source=hit.source, cached=True, pipeline=hit.pipeline):
                trace_event(tracer, "report_cache_hit")
            return dataclasses.replace(hit, cached=True, steps=0)
    result = run_report(ReportTools(model_service, image, prediction, tracer=tracer), llm, tracer=tracer)
    if key is not None and result.source == "llm":
        cache.put(key, result)
    return result

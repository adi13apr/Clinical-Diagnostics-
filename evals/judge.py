"""LLM judge: scores a report against the tool outputs with the rubric in evals/rubric.md.

Structured output on either provider: Groq/OpenAI-compatible strict json_schema, or Gemini
response_schema. The judge never sees the true label: it judges the report against what
the tools returned.
"""
import asyncio
import json
import time
from contextlib import nullcontext
from pathlib import Path

from agent.agent import (
    DEFAULT_REQUEST_TIMEOUT_S, RATE_LIMIT_MAX_RETRY_DELAY, RateLimitError, openai_quota_info, openai_retry_after,
    openai_usage, quota_info, retry_delay_seconds, split_request_options,
)
from agent.llm_config import load_llm_config, provider_settings, request_options
from agent.rate_limit import DailyCapReached, budget_for, estimate_tokens
from evals.schema import CRITERIA, JudgeResult

RUBRIC_PATH = Path(__file__).resolve().parent / "rubric.md"

JUDGE_SYSTEM = """You are a strict evaluator of AI-written chest X-ray screening reports.
Score the REPORT against the TOOL OUTPUTS using the rubric. The tool outputs are the only
ground truth: do not use outside medical knowledge to reward or penalise content, except to
notice claims the tools do not support. Give whole-number scores 1-5 and one short line of
reason per criterion. Do not reward length."""


def judge_prompt(tool_outputs, report, rubric):
    return (f"RUBRIC:\n{rubric}\n\nTOOL OUTPUTS (ground truth):\n{json.dumps(tool_outputs, indent=1)}\n\n"
            f"REPORT TO SCORE:\n{json.dumps(report, indent=1)}")


# Strict-mode schema: every object closed and fully required; scores as an enum (min/max isn't
# documented for Groq strict mode). Pydantic still validates the parsed result.
_CRITERION_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["score", "reason"],
                     "properties": {"score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                                    "reason": {"type": "string"}}}
JUDGE_JSON_SCHEMA = {"type": "object", "additionalProperties": False, "required": CRITERIA,
                     "properties": {c: _CRITERION_SCHEMA for c in CRITERIA}}


class OpenAICompatibleJudge:
    """Judge on a Chat Completions endpoint (Groq) with strict json_schema output.

    judge() is sync; ajudge() is the async path used by the eval runner (AsyncOpenAI, asyncio.sleep,
    the model's semaphore slot, timeout_s). Same request, same 429 rule, same parsing.
    """

    def __init__(self, api_key, model, base_url=None, client=None, sleep=time.sleep, budget=None, options=None,
                 provider="Groq", rubric_path=RUBRIC_PATH, async_client=None, async_sleep=asyncio.sleep,
                 timeout_s=DEFAULT_REQUEST_TIMEOUT_S):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.client, self.model, self.sleep, self.budget, self.provider = client, model, sleep, budget, provider
        self.options = dict(options or {})
        self.rubric = Path(rubric_path).read_text(encoding="utf-8")
        self._api_key, self._base_url = api_key, base_url
        self._async_client, self.async_sleep, self.timeout_s = async_client, async_sleep, timeout_s

    def _request(self, tool_outputs, report):
        messages = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_prompt(tool_outputs, report, self.rubric)}]
        response_format = {"type": "json_schema",
                           "json_schema": {"name": "judge_result", "strict": True, "schema": JUDGE_JSON_SCHEMA}}
        max_out = int(self.options.get("max_completion_tokens", 1000))
        return messages, response_format, (estimate_tokens(messages) + max_out, max_out)

    def _rate_limit_error(self, error, delay):
        return RateLimitError(delay, *openai_quota_info(error), provider=self.provider,
                              resets="the provider's daily window")

    def _parse(self, response):
        return JudgeResult.model_validate_json(response.choices[0].message.content), openai_usage(response.usage)

    def judge(self, tool_outputs, report):
        import openai
        messages, response_format, size = self._request(tool_outputs, report)
        for attempt in (1, 2):
            reservation = None
            if self.budget is not None:
                try:
                    reservation = self.budget.acquire(*size)
                except DailyCapReached as cap:
                    raise RateLimitError.from_daily_cap(cap, self.provider) from cap
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, response_format=response_format,
                    **split_request_options(self.options))
                break
            except openai.RateLimitError as e:
                delay = openai_retry_after(e)
                if attempt == 1 and delay is not None and delay < RATE_LIMIT_MAX_RETRY_DELAY:
                    self.sleep(delay)
                    continue
                raise self._rate_limit_error(e, delay) from e
        if self.budget is not None and response.usage is not None:
            self.budget.record(reservation, response.usage.total_tokens, response.usage.completion_tokens)
        return self._parse(response)

    async def ajudge(self, tool_outputs, report):
        import openai
        messages, response_format, size = self._request(tool_outputs, report)
        for attempt in (1, 2):
            reservation = None
            if self.budget is not None:
                try:
                    reservation = await self.budget.aacquire(*size)
                except DailyCapReached as cap:
                    raise RateLimitError.from_daily_cap(cap, self.provider) from cap
            try:
                async with (self.budget.slot() if self.budget is not None else nullcontext()):
                    response = await asyncio.wait_for(
                        self._async().chat.completions.create(
                            model=self.model, messages=messages, response_format=response_format,
                            **split_request_options(self.options)),
                        self.timeout_s)
                break
            except openai.RateLimitError as e:
                delay = openai_retry_after(e)
                if attempt == 1 and delay is not None and delay < RATE_LIMIT_MAX_RETRY_DELAY:
                    await self.async_sleep(delay)
                    continue
                raise self._rate_limit_error(e, delay) from e
        if self.budget is not None and response.usage is not None:
            await self.budget.arecord(reservation, response.usage.total_tokens, response.usage.completion_tokens)
        return self._parse(response)

    def _async(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url, max_retries=0)
        return self._async_client


class GeminiJudge:
    def __init__(self, api_key, model, client=None, sleep=time.sleep, rubric_path=RUBRIC_PATH, budget=None):
        from google import genai
        from google.genai import errors, types
        self.types, self.errors = types, errors
        self.model = model
        self.client = client or genai.Client(api_key=api_key)
        self.sleep = sleep
        self.budget = budget
        self.rubric = Path(rubric_path).read_text(encoding="utf-8")

    def judge(self, tool_outputs, report):
        config = self.types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM,
            response_mime_type="application/json",
            response_schema=JudgeResult,
            thinking_config=self.types.ThinkingConfig(thinking_level="low"),
        )
        contents = judge_prompt(tool_outputs, report, self.rubric)
        for attempt in (1, 2):
            if self.budget is not None:
                try:
                    self.budget.acquire(estimate_tokens(JUDGE_SYSTEM, contents) + 1000)
                except DailyCapReached as cap:
                    raise RateLimitError.from_daily_cap(cap, "Gemini") from cap
            try:
                response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except self.errors.ClientError as e:
                if e.code != 429:
                    raise
                delay = retry_delay_seconds(e)
                if attempt == 1 and delay is not None and delay < RATE_LIMIT_MAX_RETRY_DELAY:
                    self.sleep(delay)
                    continue
                raise RateLimitError(delay, *quota_info(e)) from e
        parsed = response.parsed
        if not isinstance(parsed, JudgeResult):  # SDK couldn't parse: validate the raw text (raises if invalid)
            parsed = JudgeResult.model_validate_json(response.text)
        meta = response.usage_metadata
        usage = None if meta is None else {"input_tokens": meta.prompt_token_count,
                                           "output_tokens": meta.candidates_token_count,
                                           "thinking_tokens": meta.thoughts_token_count}
        return parsed, usage


def build_judge(model=None, config=None):
    """Judge for the configured provider (config/llm.json judge_model), or None without a key."""
    config = config or load_llm_config()
    name, settings = provider_settings(config)
    if not settings["api_key"]:
        return None
    model = model or settings["judge_model"]
    budget = budget_for(model, config["limits"].get(model, {}))
    if settings["kind"] == "gemini":
        return GeminiJudge(settings["api_key"], model, budget=budget)
    return OpenAICompatibleJudge(settings["api_key"], model, settings["base_url"], budget=budget,
                                 options=request_options(model, config), provider=name.capitalize(),
                                 timeout_s=config.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S))


class MockJudge:
    """Returns fixed scores (tests)."""
    model = "mock-judge"

    def __init__(self, scores=None, delay_s=0.0):
        self.scores = scores or {"faithfulness": 5, "uncertainty_language": 4, "actionability": 4,
                                 "plain_language": 3}
        self.calls = 0
        self.delay_s = delay_s  # simulated latency (benchmarks/tests)

    def judge(self, tool_outputs, report):
        if self.delay_s:
            time.sleep(self.delay_s)
        return self._result()

    async def ajudge(self, tool_outputs, report):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self._result()

    def _result(self):
        self.calls += 1
        return JudgeResult.model_validate(
            {k: {"score": v, "reason": f"mock {k}"} for k, v in self.scores.items()}), None

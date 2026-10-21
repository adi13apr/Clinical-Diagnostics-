"""Groq / OpenAI-compatible provider, token-aware budgets, config selection, grounded caveat."""
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from PIL import Image

from agent.agent import (
    SYSTEM_PROMPT, GeminiClient, MockLLMClient, OpenAICompatibleClient, RateLimitError, build_llm_client,
    classify_llm_error, describe_llm_error, fallback_report, generate_report,
)
from agent.llm_config import load_llm_config, provider_settings
from agent.rate_limit import DailyCapReached, RedisModelBudget
from agent.tools import TOOL_DECLARATIONS, ReportTools
from evals.judge import JUDGE_JSON_SCHEMA, OpenAICompatibleJudge, build_judge
from evals.schema import CRITERIA

CONFIDENT = ([2.0, 0.0, 0.0], 0.1318)


# ---- Fakes for the openai SDK ----

def completion(content=None, tool_calls=None, prompt=900, completion_tokens=120, reasoning=40, finish="stop"):
    calls = [SimpleNamespace(id=f"call_{i}", type="function",
                             function=SimpleNamespace(name=name, arguments=args))
             for i, (name, args) in enumerate(tool_calls or [])]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or None),
                                 finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion_tokens,
                              total_tokens=prompt + completion_tokens,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning)))


class FakeChat:
    def __init__(self, items):
        self.items, self.requests = list(items), []
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_openai(items):
    chat = FakeChat(items)
    return SimpleNamespace(chat=chat), chat


def rate_limited(retry_after, message="Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute "
                                      "(TPM): Limit 8000, Used 7900, Requested 900. Please try again in 6.2s."):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return openai.RateLimitError(message, response=httpx.Response(429, headers=headers, request=request), body=None)


def tool_use_failed():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "Failed to call a function.", "type": "invalid_request_error",
                      "code": "tool_use_failed"}}
    return openai.BadRequestError("Error code: 400 - tool_use_failed",
                                  response=httpx.Response(400, request=request), body=body)


def client_for(items, **kwargs):
    fake, chat = fake_openai(items)
    return OpenAICompatibleClient("unused", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1",
                                  client=fake, sleep=kwargs.pop("sleep", lambda s: None), **kwargs), chat


@pytest.fixture
def tools(make_service, calibration_file, monkeypatch):
    monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: {"peak_region": "right upper zone",
                                                                 "focus": "diffuse"})
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
    return ReportTools(service, Image.new("L", (64, 64), 128))


def valid_report(truth):
    return {"predicted_class": truth["predicted_class"], "calibrated_probability": truth["calibrated_probability"],
            "status": truth["status"], "summary": "The model suggests Normal. This is a screening result.",
            "recommendation": "Share with a clinician.", "limitations": ["Two public datasets only."],
            "disclaimer": "x"}


# ---- OpenAICompatibleClient ----

def test_sequential_tool_calls_round_trip(tools):
    """gpt-oss-120b has no parallel tool calls: one tool per turn, results sent back as role=tool."""
    truth = tools.run_classifier()
    llm, chat = client_for([
        completion(tool_calls=[("run_classifier", "{}")]),
        completion(tool_calls=[("get_gradcam", "")]),
        completion(tool_calls=[("get_condition_info", '{"condition": "Normal"}')]),
        completion(content=json.dumps(valid_report(truth))),
    ])
    result = generate_report(tools, llm)
    assert result.source == "llm" and result.steps == 4
    assert result.tool_calls == ["run_classifier", "get_gradcam", "get_condition_info"]
    last = chat.requests[-1]["messages"]
    assert last[0]["role"] == "system" and last[1] == {"role": "user", "content": last[1]["content"]}
    assistant, tool_msg = last[2], last[3]
    assert assistant["tool_calls"][0]["function"]["name"] == "run_classifier"
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_0"
    assert json.loads(tool_msg["content"])["predicted_class"] == truth["predicted_class"]
    assert chat.requests[0]["tools"][0]["type"] == "function"
    assert {t["function"]["name"] for t in chat.requests[0]["tools"]} == {d["name"] for d in TOOL_DECLARATIONS}


def test_request_options_and_usage_split():
    llm, chat = client_for([completion(content="{}", prompt=1000, completion_tokens=300, reasoning=120)],
                           options={"reasoning_effort": "low", "include_reasoning": False,
                                    "max_completion_tokens": 2000})
    turn = llm.generate("sys", [{"role": "user", "text": "hi"}], TOOL_DECLARATIONS)
    # Standard parameters go as kwargs; Groq-specific ones in extra_body (the openai SDK rejects unknown kwargs)
    assert chat.requests[0]["reasoning_effort"] == "low" and chat.requests[0]["max_completion_tokens"] == 2000
    assert chat.requests[0]["extra_body"] == {"include_reasoning": False}
    assert turn.usage == {"input_tokens": 1000, "output_tokens": 180, "thinking_tokens": 120}
    assert turn.finish_reason == "stop"


def test_malformed_tool_arguments_become_empty():
    llm, _ = client_for([completion(tool_calls=[("get_gradcam", "{not json")])])
    turn = llm.generate("sys", [{"role": "user", "text": "hi"}], TOOL_DECLARATIONS)
    assert turn.tool_calls[0].name == "get_gradcam" and turn.tool_calls[0].args == {}


def test_short_429_is_retried_once():
    sleeps = []
    llm, chat = client_for([rate_limited(3), completion(content="{}")], sleep=sleeps.append)
    turn = llm.generate("sys", [{"role": "user", "text": "hi"}], [])
    assert sleeps == [3.0] and turn.rate_limit_retries == 1 and len(chat.requests) == 2


def test_daily_429_raises_with_groq_wording(tools):
    msg = ("Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, "
           "Used 199500, Requested 2000. Please try again in 12m3.4s.")
    llm, _ = client_for([rate_limited(None, msg)])
    result = generate_report(tools, llm)
    assert result.source == "fallback" and result.error_kind == "rate_limit"
    assert result.reason == "Groq daily quota of 200000 tokens reached (resets at the provider's daily window)."


def test_retry_after_parsed_from_message_when_header_missing():
    from agent.agent import openai_retry_after
    assert openai_retry_after(rate_limited(None)) == 6.2
    assert openai_retry_after(rate_limited(None, "Please try again in 1m2.5s.")) == 62.5


def test_tool_use_failed_twice_is_a_model_failure(tools):
    llm, _ = client_for([tool_use_failed(), tool_use_failed()])
    result = generate_report(tools, llm)
    assert result.source == "fallback" and result.error_kind == "model"
    assert result.reason == "The model produced an invalid tool call."
    assert classify_llm_error(ConnectionError()) == "infrastructure"


def test_tool_use_failed_once_is_retried(tools):
    truth = tools.run_classifier()
    llm, chat = client_for([
        completion(tool_calls=[("run_classifier", "{}")]),
        tool_use_failed(),                                   # transient: retried within the single retry
        completion(tool_calls=[("get_gradcam", "{}")]),
        completion(tool_calls=[("get_condition_info", "{}")]),
        completion(content=json.dumps(valid_report(truth))),
    ])
    result = generate_report(tools, llm)
    assert result.source == "llm" and result.problems == ["The model produced an invalid tool call."]
    assert result.rejected[0]["text"] is None and "tool_use_failed" in result.rejected[0]["error_detail"]


def test_final_turn_forbids_tool_calls(tools):
    truth = tools.run_classifier()
    llm, chat = client_for([
        completion(tool_calls=[("run_classifier", "{}")]),
        completion(tool_calls=[("get_gradcam", "{}")]),
        completion(tool_calls=[("get_condition_info", "{}")]),
        completion(content=json.dumps(valid_report(truth))),
    ])
    generate_report(tools, llm)
    assert [r.get("tool_choice") for r in chat.requests] == [None, None, None, "none"]
    mock = MockLLMClient([MockLLMClient.tool("run_classifier"), MockLLMClient.tool("get_gradcam"),
                          MockLLMClient.tool("get_condition_info"), MockLLMClient.final(valid_report(truth))])
    generate_report(tools, mock)
    assert mock.tools_allowed == [True, True, True, False]


# ---- Token-aware budget (every test runs on both backends: in memory and in fake Redis) ----

def test_budget_waits_for_token_window(make_budget, budget_clock):
    budget = make_budget(rpm=30, tpm=8000)
    first = budget.acquire(5000)
    budget.record(first, 6000)          # real usage was higher than estimated
    budget_clock.now += 10
    budget.acquire(2500)                # 6000 + 2500 > 8000 -> wait until the first request leaves the window
    assert budget_clock.sleeps == [50.0]
    assert budget.used_last_minute() == (1, 2500)


def test_budget_respects_requests_per_minute(make_budget, budget_clock):
    budget = make_budget(rpm=2, tpm=None)
    for _ in range(3):
        budget.acquire(10)
    assert budget_clock.sleeps == [60.0]


def test_budget_daily_caps_and_rollover(make_budget, budget_clock):
    budget = make_budget(rpd=2, tpd=None)
    budget.acquire(1)
    budget_clock.now += 61
    budget.acquire(1)
    with pytest.raises(DailyCapReached, match="request"):
        budget.acquire(1)
    budget_clock.now += 86400           # new (UTC) day resets the counters
    budget.acquire(1)
    tokens = make_budget("m2", tpd=1000)
    with pytest.raises(DailyCapReached, match="token"):
        tokens.acquire(1500)


def test_budget_output_tokens_per_minute(make_budget, budget_clock):
    """Groq checks max_completion_tokens against OTPM up front (qwen/qwen3.8-27b: 1000 on the free tier)."""
    budget = make_budget("qwen/qwen3.8-27b", tpm=8000, otpm=1000)
    first = budget.acquire(2000, 600)
    budget.record(first, 1900, 210)       # actual output was far below the reservation
    budget.acquire(2000, 600)             # 210 + 600 <= 1000: no wait
    budget_clock.now += 5
    budget.acquire(2000, 600)             # 810 + 600 > 1000: wait for the first request to leave the window
    assert budget_clock.sleeps == [55.0]


def test_otpm_429_is_parsed_and_comment_options_are_not_sent():
    from agent.agent import openai_quota_info
    from agent.llm_config import request_options
    msg = ("Request too large for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on "
           "output tokens per minute (OTPM): Limit 1000, Requested 1263.")
    assert openai_quota_info(rate_limited(None, msg)) == ("output tokens per minute (OTPM)", "1000")
    options = request_options("qwen/qwen3.8-27b")
    assert "_why" not in options and options["max_completion_tokens"] <= 1000
    assert load_llm_config()["limits"]["qwen/qwen3.8-27b"]["otpm"] == 1000


def test_client_daily_cap_becomes_rate_limit_fallback(tools, make_budget):
    budget = make_budget("openai/gpt-oss-120b", rpd=0)
    llm, chat = client_for([completion(content="{}")], budget=budget)
    result = generate_report(tools, llm)
    assert result.source == "fallback" and result.error_kind == "rate_limit" and chat.requests == []
    assert "daily quota" in result.reason


def test_budget_records_actual_tokens(make_budget):
    budget = make_budget("openai/gpt-oss-120b", tpm=8000)
    llm, _ = client_for([completion(content="{}", prompt=1000, completion_tokens=200)], budget=budget,
                        options={"max_completion_tokens": 2000})
    llm.generate("sys", [{"role": "user", "text": "hi"}], [])
    assert budget.used_last_minute() == (1, 1200) and budget.tokens_today == 1200


def test_unreachable_shared_budget_falls_back_to_the_template(tools, make_redis, redis_server):
    """With REDIS_URL, a Redis outage means no LLM call (its limits can't be checked), not a failed report."""
    budget = RedisModelBudget(make_redis(), "openai/gpt-oss-120b", rpm=30)
    llm, chat = client_for([completion(content="{}")], budget=budget)
    redis_server.connected = False
    result = generate_report(tools, llm)
    assert result.source == "fallback" and result.error_kind == "infrastructure" and chat.requests == []


# ---- Config ----

def test_config_defaults_to_groq_models():
    config = load_llm_config()
    assert config["provider"] == "groq"
    groq = config["providers"]["groq"]
    assert groq["report_model"] == "openai/gpt-oss-120b" and groq["judge_model"] == "qwen/qwen3.8-27b"
    assert config["limits"]["openai/gpt-oss-120b"]["tpm"] == 8000
    assert config["limits"]["openai/gpt-oss-120b"]["rpd"] == 1000


def test_provider_selected_by_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    monkeypatch.setenv("GEMINI_API_KEY", "gm")
    monkeypatch.setattr("agent.llm_config._secret", lambda section, key: None)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert isinstance(build_llm_client(), OpenAICompatibleClient)
    assert build_llm_client().name == "openai/gpt-oss-120b"
    assert build_judge().model == "qwen/qwen3.8-27b"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert isinstance(build_llm_client(), GeminiClient)
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        provider_settings()


def test_no_key_means_no_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("agent.llm_config._secret", lambda section, key: None)
    assert build_llm_client() is None and build_judge() is None


# ---- Judge ----

def test_openai_judge_strict_schema_and_parse():
    payload = {c: {"score": 4, "reason": "ok"} for c in CRITERIA}
    fake, chat = fake_openai([completion(content=json.dumps(payload))])
    judge = OpenAICompatibleJudge("unused", "qwen/qwen3.8-27b", client=fake,
                                  options={"reasoning_effort": "low", "reasoning_format": "hidden"})
    result, usage = judge.judge({"classifier": {}}, {"summary": "x"})
    assert result.plain_language.score == 4 and usage["input_tokens"] == 900
    fmt = chat.requests[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert chat.requests[0]["extra_body"] == {"reasoning_format": "hidden"}
    schema = JUDGE_JSON_SCHEMA
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(CRITERIA)
    assert schema["properties"]["faithfulness"]["properties"]["score"]["enum"] == [1, 2, 3, 4, 5]


def test_openai_judge_long_429_raises():
    fake, _ = fake_openai([rate_limited(40)])
    with pytest.raises(RateLimitError):
        OpenAICompatibleJudge("unused", "qwen/qwen3.8-27b", client=fake, sleep=lambda s: None).judge({}, {})


# ---- Grounded dataset caveat ----

def test_dataset_caveat_is_grounded_in_a_tool_not_the_prompt(tools):
    assert "two public datasets" not in SYSTEM_PROMPT and "different source" not in SYSTEM_PROMPT
    info = tools.get_condition_info()
    caveat = next(l for l in info["model_limitations"] if "two public datasets" in l)
    assert "different source" in caveat
    report = fallback_report(tools)
    assert caveat in report.limitations  # template limitations come from the tool output verbatim
    assert all(len(l) <= 300 for l in report.limitations)  # within the eval length limit


def test_describe_errors_for_openai():
    assert describe_llm_error(tool_use_failed()) == "The model produced an invalid tool call."
    assert MockLLMClient  # interface unchanged

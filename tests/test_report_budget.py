"""Report jobs vs the shared Groq budget: reservation size, reconciliation, the derived job timeout, the
deadline that ends a job as the template before RQ kills it, and replies cut off at the length limit.

Budget tests run on both backends (in memory, and fakeredis through make_budget) on a fake clock.
Numbers are the measured ones (2026-10-25 demo spans): a report's 4 calls reserve prompt estimates of
1,227 / 1,364 / 1,589 / 2,019 tokens plus max_completion_tokens each, and really use 999 / 1,102 /
1,276 / 1,745 tokens in total, of which 27 / 27 / 22 / 215 are completion tokens.
"""
import json
import time

import pytest
from rq import SimpleWorker

from agent.agent import (
    BudgetDeadlineError, LLMTurn, MockLLMClient, OpenAICompatibleClient, classify_llm_error, describe_llm_error,
    generate_report,
)
from agent.llm_config import load_llm_config, request_options
from agent.rate_limit import BudgetWaitTooLong, RedisModelBudget, estimate_tokens
from api.jobs import JOB_TIMEOUT_MAX_S, JOB_TIMEOUT_MIN_S, report_job_timeout, report_model_budget
from services.tracing import Tracer
from test_agent import CONFIDENT, full_script, make_tools, report_json  # noqa: F401  (make_tools is a fixture)
from test_api import ALICE, model_service, predict, report_script, store  # noqa: F401  (fixtures)
from test_providers import completion, fake_openai
from test_report_jobs import poll, post_report, queue_app, spans_in_db  # noqa: F401  (queue_app is a fixture)

MODEL = "openai/gpt-oss-120b"
PROMPT_ESTIMATES = [1227, 1364, 1589, 2019]
REAL_TOTAL = [999, 1102, 1276, 1745]
REAL_COMPLETION = [27, 27, 22, 215]
MEASURED_MAX_PER_CALL = 238     # largest completion of one call, demo spans
MEASURED_MAX_PER_REPORT = 400   # largest completion of a whole report, 216 eval reports


def cap():
    return request_options(MODEL)["max_completion_tokens"]


def fits_now(budget, tokens, output):
    """Reserve without waiting: the reservation, or None if the budget would make the caller wait."""
    event, _ = budget._try_reserve(tokens, output)
    return event


# ---- Reservation ----

def test_the_output_cap_is_well_above_measured_use():
    assert cap() >= 1.5 * MEASURED_MAX_PER_REPORT and cap() >= 1.5 * MEASURED_MAX_PER_CALL
    assert cap() < 2000  # the old cap made every call reserve 2.3-3.2x its real use


def test_a_call_reserves_its_prompt_estimate_plus_the_cap():
    llm = OpenAICompatibleClient("k", MODEL, "x", client=object(), options=request_options(MODEL))
    payload, specs = [{"role": "user", "content": "x" * 3500}], [{"type": "function"}]
    assert llm._reservation_size(payload, specs) == (estimate_tokens(payload, specs) + cap(), cap())


@pytest.mark.parametrize("output_cap, calls", [(2000, 2), (600, 4)])
def test_a_smaller_cap_fits_more_first_calls_in_a_minute(make_budget, output_cap, calls):
    budget = make_budget(MODEL, tpm=8000, rpm=30)
    reserved = [fits_now(budget, PROMPT_ESTIMATES[0] + output_cap, output_cap) for _ in range(calls + 1)]
    assert all(reserved[:calls]) and reserved[calls] is None


# ---- Reconciliation ----

def test_recording_real_usage_frees_the_unused_reservation(make_budget, budget_clock):
    budget = make_budget(MODEL, tpm=8000, rpm=30)
    size = PROMPT_ESTIMATES[0] + cap()
    events = [fits_now(budget, size, cap()) for _ in range(4)]
    assert all(events) and fits_now(budget, size, cap()) is None
    assert budget.used_last_minute() == (4, 4 * size)

    for event in events:
        budget.record(event, REAL_TOTAL[0], REAL_COMPLETION[0])
    assert budget.used_last_minute() == (4, 4 * REAL_TOTAL[0])
    assert fits_now(budget, size, cap()) and fits_now(budget, size, cap())  # room the estimates were holding
    assert budget.tokens_today == 4 * REAL_TOTAL[0] + 2 * size  # the daily total is corrected too


def test_three_concurrent_reports_never_exceed_the_budget(make_budget, budget_clock):
    """Three workers step through a report each; the window, reservations included, stays within tpm."""
    budget = make_budget(MODEL, tpm=8000, rpm=30)
    step, done_at, peak = [0, 0, 0], [None] * 3, 0
    while None in done_at:
        for worker in range(3):
            if done_at[worker] is not None:
                continue
            s = step[worker]
            event = fits_now(budget, PROMPT_ESTIMATES[s] + cap(), cap())
            if event is None:
                continue
            peak = max(peak, budget.used_last_minute()[1])
            assert REAL_TOTAL[s] <= PROMPT_ESTIMATES[s] + cap()  # what it records never exceeds what it reserved
            budget.record(event, REAL_TOTAL[s], REAL_COMPLETION[s])
            step[worker] += 1
            if step[worker] == 4:
                done_at[worker] = budget_clock.now
        budget_clock.now += 1.0
        assert budget_clock.now < 1e10
    assert peak <= 8000
    assert budget.used_last_minute()[1] <= 8000


# ---- Waiting with a deadline ----

def test_a_request_that_fits_is_reserved_even_with_no_time_left(make_budget, budget_clock):
    budget = make_budget(MODEL, tpm=8000)
    assert budget.acquire(1000, 100, max_wait=0) is not None and budget_clock.sleeps == []


def test_a_wait_past_max_wait_raises_without_sleeping(make_budget, budget_clock):
    budget = make_budget(MODEL, tpm=8000)
    budget.acquire(7500, 600)
    with pytest.raises(BudgetWaitTooLong) as raised:
        budget.acquire(1827, 600, max_wait=10)
    assert raised.value.wait == pytest.approx(60) and budget_clock.sleeps == []
    assert budget.used_last_minute() == (1, 7500)  # nothing reserved


def test_a_wait_within_max_wait_is_waited_out(make_budget, budget_clock):
    budget = make_budget(MODEL, tpm=8000)
    budget.acquire(7500, 600)
    assert budget.acquire(1827, 600, max_wait=90) is not None and budget_clock.sleeps == [60.0]


@pytest.mark.asyncio
async def test_async_wait_past_max_wait_raises_too(make_budget, budget_clock):
    budget = make_budget(MODEL, tpm=8000)
    await budget.aacquire(7500, 600)
    with pytest.raises(BudgetWaitTooLong):
        await budget.aacquire(1827, 600, max_wait=10)
    assert budget_clock.sleeps == []


def full_budget(make_redis, budget_clock):
    budget = RedisModelBudget(make_redis(), MODEL, tpm=8000, rpm=30, clock=budget_clock,
                              sleep=budget_clock.sleep, async_sleep=budget_clock.asleep)
    budget.acquire(7500, 600)  # another worker's calls: the next one waits about 60 s
    return budget


def test_a_client_past_its_deadline_gives_up_before_calling_groq(make_redis, budget_clock):
    client, chat = fake_openai([])
    llm = OpenAICompatibleClient("k", MODEL, "x", client=client, budget=full_budget(make_redis, budget_clock),
                                 options=request_options(MODEL))
    llm.deadline = time.monotonic() + 10
    with pytest.raises(BudgetDeadlineError) as raised:
        llm.generate("system", [{"role": "user", "text": "hi"}], [])
    assert chat.requests == [] and budget_clock.sleeps == []
    assert classify_llm_error(raised.value) == "rate_limit"
    assert "time limit" in describe_llm_error(raised.value)


# ---- Derived job timeout ----

@pytest.mark.parametrize("workers, timeout", [(0, 180), (1, 180), (2, 240), (3, 300), (5, 420), (50, 900)])
def test_job_timeout_grows_with_the_workers_sharing_the_budget(workers, timeout):
    assert report_job_timeout(workers, tpm=8000, rpm=30, max_completion_tokens=600) == timeout


def test_job_timeout_bounds_and_request_limit():
    assert report_job_timeout(1, tpm=None, rpm=None) == JOB_TIMEOUT_MIN_S
    assert report_job_timeout(10, tpm=None, rpm=5) == 60 + 60 * 8  # 40 calls at 5 per minute
    assert report_job_timeout(1000, tpm=8000, max_completion_tokens=600) == JOB_TIMEOUT_MAX_S


def test_job_timeout_uses_the_configured_report_model():
    config = load_llm_config()
    assert report_model_budget(config) == {"tpm": config["limits"][MODEL]["tpm"], "rpm": config["limits"][MODEL]["rpm"],
                                           "max_completion_tokens": cap()}


def test_submit_derives_the_timeout_from_the_listening_workers(queue_app):
    client, jobs = queue_app(is_async=True)
    first = post_report(client, predict(client)["trace_id"])["job_id"]
    for _ in range(3):
        SimpleWorker([jobs.queue], connection=jobs.client, serializer=jobs.queue.serializer).register_birth()
    second = post_report(client, predict(client)["trace_id"])["job_id"]
    timeouts = [jobs._fetch(job_id).timeout for job_id in (first, second)]
    assert timeouts == [180, 300]


# ---- The timeout path in the worker ----

def tool_turns():
    return [completion(tool_calls=[(name, "{}")], finish="tool_calls")
            for name in ("run_classifier", "get_gradcam", "get_condition_info")]


def groq_factory(make_redis, budget_clock, prediction):
    """llm_factory: a Groq client on a budget another worker has filled; if called, it writes a full report
    for prediction (a dict that the test fills after /predict)."""
    budget = full_budget(make_redis, budget_clock)
    chats = []

    def factory():
        client, chat = fake_openai(tool_turns() + [completion(content=report_script(prediction)[-1].text)])
        chats.append(chat)
        return OpenAICompatibleClient("k", MODEL, "x", client=client, budget=budget, options=request_options(MODEL))
    return factory, chats


def test_a_job_that_cannot_get_budget_in_time_ends_as_the_template(queue_app, store, make_redis, budget_clock):
    """Timeout 40 s, deadline 10 s, next budget slot in 60 s: the template now, not a job killed later."""
    prediction = {}
    factory, chats = groq_factory(make_redis, budget_clock, prediction)
    client, _ = queue_app(llm_factory=factory, job_timeout_s=40)
    prediction.update(predict(client))
    trace_id = prediction["trace_id"]
    body = poll(client, post_report(client, trace_id)["job_id"]).json()
    assert body["status"] == "done" and body["error"] is None
    assert body["result"]["source"] == "fallback" and "time limit" in body["result"]["reason"]
    assert chats[-1].requests == [] and budget_clock.sleeps == []  # Groq never called, nothing slept

    calls = [r for r in spans_in_db(store, trace_id) if r["name"] == "llm_call"]
    assert [(c["status"], c["error_type"]) for c in calls] == [("error", "BudgetDeadlineError")]
    used = [r for r in spans_in_db(store, trace_id) if r["name"] == "fallback_used"]
    assert used[0]["attributes"]["used"] is True


def test_a_job_with_time_to_wait_waits_and_uses_the_llm(queue_app, make_redis, budget_clock):
    """Timeout 300 s, deadline 270 s, next budget slot in 60 s: it waits, then the LLM writes the report."""
    prediction = {}
    factory, chats = groq_factory(make_redis, budget_clock, prediction)
    client, _ = queue_app(llm_factory=factory, job_timeout_s=300)
    prediction.update(predict(client))
    body = poll(client, post_report(client, prediction["trace_id"])["job_id"]).json()
    assert body["result"]["source"] == "llm", body["result"]["reason"]
    assert len(chats[-1].requests) == 4 and budget_clock.sleeps[0] == pytest.approx(60)


# ---- Replies cut off at the length limit ----

def cut_off(text='{"predicted_class": "Nor'):
    return LLMTurn(text=text, finish_reason="length", usage={"input_tokens": 1530, "output_tokens": 600})


def test_a_cut_off_reply_is_retried_and_traced(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    llm = MockLLMClient(full_script(truth)[:3] + [cut_off(), MockLLMClient.final(report_json(truth))])
    tracer = Tracer()
    result = generate_report(tools, llm, tracer=tracer)
    assert result.source == "llm"
    assert result.rejected == [{"step": 4, "text": '{"predicted_class": "Nor', "finish_reason": "length",
                                "problem": "the reply was cut off at the output-token limit (finish_reason 'length')"}]
    retry_messages = llm.calls[-1]
    assert not any(m["role"] == "assistant" and m["turn"].finish_reason == "length" for m in retry_messages)
    assert retry_messages[-1]["role"] == "user" and "cut off" in retry_messages[-1]["text"]

    events = [s for s in tracer.spans if s.name == "llm_truncated"]
    assert len(events) == 1 and events[0].status == "error"
    assert events[0].attributes == {"step": 4, "finish_reason": "length", "output_tokens": 600}


def test_cut_off_twice_falls_back_as_a_model_failure(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    result = generate_report(tools, MockLLMClient(full_script(truth)[:3] + [cut_off(), cut_off()]))
    assert result.source == "fallback" and result.error_kind == "model"
    assert result.reason == "Report cut off at the output-token limit after 1 retry."
    assert len(result.rejected) == 2


def test_groq_finish_reason_length_reaches_the_trace(make_tools):
    tools = make_tools(CONFIDENT)
    truth = tools.run_classifier()
    client, chat = fake_openai(tool_turns() + [
                                completion(content='{"predicted_class": "Nor', finish="length", completion_tokens=600),
                                completion(content=json.dumps(report_json(truth)))])
    llm = OpenAICompatibleClient("k", MODEL, "x", client=client, options=request_options(MODEL))
    tracer = Tracer()
    result = generate_report(tools, llm, tracer=tracer)
    assert result.source == "llm" and len(chat.requests) == 5
    assert all(r["max_completion_tokens"] == cap() for r in chat.requests)
    finishes = [s.attributes["finish_reason"] for s in tracer.spans if s.name == "llm_call"]
    assert finishes == ["tool_calls"] * 3 + ["length", "stop"]
    assert [s.name for s in tracer.spans].count("llm_truncated") == 1

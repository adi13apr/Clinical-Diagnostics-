"""Async eval runner: same results as the sync runner, real overlap, limits, timeouts, resume. No network."""
import asyncio
import copy
import json
import time
from types import SimpleNamespace

import httpx
import openai
import pytest
from PIL import Image

from agent.agent import (
    LLMTurn, MockLLMClient, OpenAICompatibleClient, ToolCall, agenerate_report, generate_report,
)
from agent.tools import ReportTools
from test_agent import CAVEAT
from evals import run as eval_run
from evals.dataset import load_cases
from evals.judge import MockJudge, OpenAICompatibleJudge
from evals.schema import CRITERIA
from services.tracing import Tracer

CASES = load_cases()
CONFIDENT = ([2.0, 0.0, 0.0], 0.1318)
GRADCAM = {"peak_region": "right upper zone", "focus": "diffuse", "border_share": 0.3}


# ---- fakes ----

def completion(content=None, tool_calls=None, prompt=900, completion_tokens=120):
    calls = [SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(name=n, arguments=a))
             for i, (n, a) in enumerate(tool_calls or [])]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or None),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion_tokens,
                              total_tokens=prompt + completion_tokens,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=None)))


class FakeChat:
    """Sync or async chat.completions.create with simulated latency."""

    def __init__(self, items, delay_s=0.0, is_async=False):
        self.items, self.delay_s, self.is_async, self.requests = list(items), delay_s, is_async, []
        self.completions = self
        self.chat = self

    def _next(self, kwargs):
        self.requests.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def create(self, **kwargs):
        if self.is_async:
            return self._acreate(kwargs)
        time.sleep(self.delay_s)
        return self._next(kwargs)

    async def _acreate(self, kwargs):
        await asyncio.sleep(self.delay_s)
        return self._next(kwargs)


def rate_limited(retry_after):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.RateLimitError("rate limited", body=None,
                                 response=httpx.Response(429, headers={"retry-after": str(retry_after)},
                                                         request=request))


@pytest.fixture
def make_tools(make_service, calibration_file, monkeypatch):
    monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: GRADCAM)

    def make(case=CONFIDENT):
        service = make_service(case[0], calibration_file(case[1]))
        return ReportTools(service, Image.new("L", (64, 64), 128))
    return make


def valid_report(truth, **overrides):
    data = {"predicted_class": truth["predicted_class"], "calibrated_probability": truth["calibrated_probability"],
            "status": truth["status"], "summary": "The model suggests Normal. This is a screening result.",
            "recommendation": "Share this result with a clinician.", "limitations": [CAVEAT],
            "disclaimer": "x"}
    data.update(overrides)
    return data


def scripts(truth):
    tools = [MockLLMClient.tool(n) for n in ("run_classifier", "get_gradcam", "get_condition_info")]
    good, bad = valid_report(truth), valid_report(truth, predicted_class="Tuberculosis")
    return {
        "happy path": tools + [MockLLMClient.final(good)],
        "rejected then fixed": tools + [MockLLMClient.final(bad), MockLLMClient.final(good)],
        "rejected twice -> template": tools + [MockLLMClient.final(bad), MockLLMClient.final(bad)],
        "llm error -> template": [lambda m: (_ for _ in ()).throw(ConnectionError("down"))],
    }


# ---- same results ----

@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["happy path", "rejected then fixed", "rejected twice -> template",
                                  "llm error -> template"])
async def test_async_report_loop_gives_identical_results(make_tools, name):
    truth = make_tools().run_classifier()
    sync_result = generate_report(make_tools(), MockLLMClient(scripts(truth)[name]))
    async_result = await agenerate_report(make_tools(), MockLLMClient(scripts(truth)[name]))
    for field in ("report", "source", "reason", "steps", "problems", "tool_calls", "rejected", "error_kind"):
        assert getattr(async_result, field) == getattr(sync_result, field), field


@pytest.mark.asyncio
async def test_async_groq_client_sends_the_same_request(make_tools):
    responses = [completion(tool_calls=[("run_classifier", "{}")]), completion(content="{}")]
    sync_fake, async_fake = FakeChat(copy.deepcopy(responses)), FakeChat(copy.deepcopy(responses), is_async=True)
    sync_llm = OpenAICompatibleClient("k", "m", "u", client=sync_fake, options={"max_completion_tokens": 50})
    async_llm = OpenAICompatibleClient("k", "m", "u", client=None, async_client=async_fake,
                                       options={"max_completion_tokens": 50})
    msgs = [{"role": "user", "text": "hi"}]
    for allowed in (True, False):
        assert sync_llm.generate("sys", msgs, [], tools_allowed=allowed) == \
            await async_llm.agenerate("sys", msgs, [], tools_allowed=allowed)
    assert sync_fake.requests == async_fake.requests


@pytest.mark.asyncio
async def test_case_record_identical_sync_and_async(make_service, calibration_file, monkeypatch, tmp_path):
    monkeypatch.setattr(ReportTools, "get_gradcam", lambda self: GRADCAM)
    Image.new("L", (64, 64), 128).save(tmp_path / "x.png")
    case = SimpleNamespace(case_id="c-test", relpath="pneumonia/x.png", snapshot=CASES[0].snapshot)
    service = make_service(CONFIDENT[0], calibration_file(CONFIDENT[1]))
    truth = ReportTools(service, Image.open(tmp_path / "x.png")).run_classifier()
    roots = {"pneumonia": tmp_path}
    a = eval_run.run_case(case, service, MockLLMClient(scripts(truth)["rejected then fixed"]), roots)
    b = await eval_run.arun_case(case, service, MockLLMClient(scripts(truth)["rejected then fixed"]), roots)
    for r in (a, b):
        r["agent_usage"].pop("report_latency_ms")  # timing is the only thing allowed to differ
    assert a == b


# ---- pipeline: overlap, identical records, resume, stops ----

def fake_record(case):
    return {"case_id": case.case_id, "report": {"summary": case.case_id}, "tool_outputs": {"case": case.case_id},
            "report_source": "llm", "agent_usage": {"llm_calls": 4}, "judge": None, "judge_error": None}


def normalise(records):
    out = {}
    for cid, r in records.items():
        r = copy.deepcopy(r)
        if r.get("judge"):
            r["judge"].pop("latency_ms")
        out[cid] = r
    return out


def run_sync(todo, records, report_s=0.0, judge=None, fail_at=None, events=None):
    lines = []

    def make(case):
        if events is not None:
            events.append(("report_start", case.case_id, time.perf_counter()))
        time.sleep(report_s)
        if fail_at and case.case_id == fail_at:
            raise eval_run.StopRun("report model rate limited")
        return fake_record(case)
    stop = eval_run.run_pipeline_sync(todo, records, make, judge or MockJudge(), lines.append, log=lambda s: None)
    return stop, lines


async def run_async(todo, records, report_s=0.0, judge=None, fail_at=None, events=None):
    lines = []

    async def make(case):
        if events is not None:
            events.append(("report_start", case.case_id, time.perf_counter()))
        await asyncio.sleep(report_s)
        if fail_at and case.case_id == fail_at:
            raise eval_run.StopRun("report model rate limited")
        if events is not None:
            events.append(("report_end", case.case_id, time.perf_counter()))
        return fake_record(case)
    stop = await eval_run.run_pipeline_async(todo, records, make, judge or MockJudge(), lines.append,
                                             log=lambda s: None)
    return stop, lines


class TimedJudge(MockJudge):
    def __init__(self, delay_s, events):
        super().__init__(delay_s=delay_s)
        self.events = events

    async def ajudge(self, tool_outputs, report):
        self.events.append(("judge_start", tool_outputs["case"], time.perf_counter()))
        return await super().ajudge(tool_outputs, report)


@pytest.mark.asyncio
async def test_judge_for_case_n_overlaps_report_for_case_n_plus_1():
    todo, events = CASES[:4], []
    started = time.perf_counter()
    stop, _ = await run_async(todo, {}, report_s=0.2, judge=TimedJudge(0.2, events), events=events)
    async_s = time.perf_counter() - started
    started = time.perf_counter()
    run_sync(todo, {}, report_s=0.2, judge=MockJudge(delay_s=0.2))
    sync_s = time.perf_counter() - started
    assert stop is None
    judge_1_start = next(t for kind, cid, t in events if kind == "judge_start" and cid == todo[0].case_id)
    report_2_end = next(t for kind, cid, t in events if kind == "report_end" and cid == todo[1].case_id)
    assert judge_1_start < report_2_end            # judging case 1 happened while case 2 was being written
    assert async_s < sync_s * 0.8                  # ~1.0 s vs ~1.6 s


@pytest.mark.asyncio
async def test_records_identical_to_sync_runner():
    sync_records, async_records = {}, {}
    run_sync(CASES[:5], sync_records)
    await run_async(CASES[:5], async_records)
    assert normalise(async_records) == normalise(sync_records)
    assert all(r["judge"]["scores"] for r in async_records.values())


@pytest.mark.asyncio
async def test_stop_and_resume_complete_the_run():
    todo, records = CASES[:5], {}
    stop, lines = await run_async(todo, records, fail_at=todo[2].case_id)
    assert "rate limited" in stop
    judged = {cid for cid, r in records.items() if r.get("judge")}
    assert judged == {todo[0].case_id, todo[1].case_id}            # cases before the stop are judged
    stop, _ = await run_async(todo, records)                        # resume with the same records
    full = {}
    run_sync(todo, full)
    assert stop is None and normalise(records) == normalise(full)


@pytest.mark.asyncio
async def test_judge_rate_limit_stops_like_sync_and_resume_rejudges():
    from agent.agent import RateLimitError

    class FlakyJudge(MockJudge):
        async def ajudge(self, tool_outputs, report):
            if self.calls == 1:
                self.calls += 1
                raise RateLimitError(3600, "tokens per day (TPD)", "200000", provider="Groq")
            return await super().ajudge(tool_outputs, report)

    todo, records = CASES[:4], {}
    stop, lines = await run_async(todo, records, judge=FlakyJudge())
    assert stop.startswith("judge:")
    assert any((line.get("judge_error") or "").startswith("RateLimitError") for line in lines)
    stop, _ = await run_async(todo, records)                        # resume: unjudged cases get judged
    assert stop is None and all(records[c.case_id]["judge"] for c in todo)


# ---- timeouts ----

@pytest.mark.asyncio
async def test_report_request_timeout_stops_the_run(make_tools):
    slow = FakeChat([completion(content="{}")], delay_s=1.0, is_async=True)
    llm = OpenAICompatibleClient("k", "m", "u", client=None, async_client=slow, timeout_s=0.05)
    tools, tracer = make_tools(), Tracer()
    result = await agenerate_report(tools, llm, tracer=tracer)
    assert result.source == "fallback" and result.error_kind == "infrastructure"
    assert result.reason == "LLM call failed (TimeoutError)."
    case = SimpleNamespace(case_id="c", snapshot=CASES[0].snapshot)
    with pytest.raises(eval_run.StopRun):                            # same as a network error: stop, resume later
        eval_run._finish_case(case, tools, tracer, result)


@pytest.mark.asyncio
async def test_judge_timeout_is_recorded_and_the_run_continues():
    payload = json.dumps({c: {"score": 4, "reason": "ok"} for c in CRITERIA})
    slow = FakeChat([completion(content=payload)] * 3, delay_s=0.3, is_async=True)
    judge = OpenAICompatibleJudge("k", "qwen", client=SimpleNamespace(), async_client=slow, timeout_s=0.05)
    records = {}
    stop, _ = await run_async(CASES[:2], records, judge=judge)
    assert stop is None                                              # like any non-rate-limit judge error
    assert all(r["judge"] is None and r["judge_error"].startswith("TimeoutError") for r in records.values())
    stop, _ = await run_async(CASES[:2], records)                    # resume re-judges them
    assert all(r["judge"] for r in records.values())


# ---- rate limits ----

# Every budget test runs on both backends (in memory and in fake Redis); see make_budget in conftest.py.

@pytest.mark.asyncio
async def test_async_budget_respects_rpm_tpm_and_otpm(make_budget, budget_clock):
    budget = make_budget("m", rpm=2, tpm=8000, otpm=1000)
    await budget.aacquire(100)
    await budget.aacquire(100)
    await budget.aacquire(100)                                       # 3rd request in a minute: waits
    assert budget_clock.sleeps == [60.0]
    budget = make_budget("m2", tpm=8000, otpm=1000)
    await budget.aacquire(2000, 600)
    budget_clock.now += 10
    await budget.aacquire(2000, 600)                                 # 600 + 600 output > 1000 OTPM: waits
    assert budget_clock.sleeps == [60.0, 50.0]


@pytest.mark.asyncio
async def test_sync_and_async_share_one_budget(make_budget, budget_clock):
    budget = make_budget(tpm=1000, sleep=lambda s: None)
    budget.acquire(800)                                              # the app (sync) used most of the minute
    await budget.aacquire(500)                                       # the async runner must wait for it
    assert budget_clock.sleeps == [60.0]


@pytest.mark.asyncio
async def test_async_record_corrects_the_reservation(make_budget):
    budget = make_budget(tpm=8000)
    reservation = await budget.aacquire(5000, 2000)
    await budget.arecord(reservation, 1200, 200)
    assert budget.used_last_minute() == (1, 1200) and budget.tokens_today == 1200


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2])
async def test_semaphore_caps_requests_in_flight(limit, make_budget):
    budget = make_budget(max_concurrent=limit)
    in_flight, peak = 0, 0

    async def request():
        nonlocal in_flight, peak
        async with budget.slot():
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1

    await asyncio.gather(*(request() for _ in range(6)))
    assert peak == limit


@pytest.mark.asyncio
async def test_async_client_short_429_retries_with_async_sleep():
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    fake = FakeChat([rate_limited(3), completion(content="{}")], is_async=True)
    llm = OpenAICompatibleClient("k", "m", "u", client=None, async_client=fake, async_sleep=fake_sleep)
    turn = await llm.agenerate("sys", [{"role": "user", "text": "hi"}], [])
    assert sleeps == [3.0] and turn.rate_limit_retries == 1


def test_budget_primitives_work_across_event_loops(make_budget):
    budget = make_budget(rpm=100)
    asyncio.run(budget.aacquire(1))
    asyncio.run(budget.aacquire(1))                                  # a new loop gets fresh asyncio primitives
    assert budget.used_last_minute()[0] == 2


def test_simulated_benchmark_overlaps_and_keeps_records_identical():
    from evals.benchmark import simulate
    sync_s, async_s, same = simulate(report_s=0.05, judge_s=0.05, n=4)
    assert same and async_s < sync_s * 0.8  # equal report/judge times: ~0.25 s vs ~0.4 s


def test_report_turn_types_unchanged():
    assert LLMTurn(tool_calls=[ToolCall("x", {})]).tool_calls[0].name == "x"

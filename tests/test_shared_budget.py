"""RedisModelBudget: one model's limits shared by several processes (API replicas) through Redis.

Each budget gets its own fakeredis client on one fake server, like two replicas sharing one Redis.
Time comes from budget_clock (conftest.py), which also drives fakeredis's key expiry: no real sleeping.
The tests that both backends must pass are in test_providers.py and test_async_eval.py (make_budget).
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from agent.agent import build_llm_client
from agent.rate_limit import BUDGETS, DailyCapReached, ModelBudget, RedisModelBudget, budget_for

ATTEMPTS = 40


class WouldWait(Exception):
    pass


def would_wait(seconds):
    raise WouldWait(seconds)


async def would_wait_async(seconds):
    raise WouldWait(seconds)


@pytest.fixture
def replicas(make_redis, budget_clock):
    """replicas(**limits) -> two RedisModelBudgets for one model, each with its own client."""
    def make(n=2, **limits):
        return [RedisModelBudget(make_redis(), "openai/gpt-oss-120b", clock=budget_clock, sleep=would_wait,
                                 async_sleep=would_wait_async, **limits) for _ in range(n)]
    return make


# Each case: limits, one request's (tokens, output), how many fit in a minute
CASES = {
    "rpm": ({"rpm": 10}, (1, 0), 10),
    "tpm": ({"tpm": 1000}, (300, 0), 3),
    "otpm": ({"tpm": 100_000, "otpm": 1000}, (10, 300), 3),
    "rpd": ({"rpd": 7}, (1, 0), 7),
}


@pytest.mark.parametrize("limits, size, allowed", CASES.values(), ids=CASES.keys())
def test_replicas_never_reserve_more_than_the_limit_together(replicas, limits, size, allowed):
    """40 threads on two replicas reserve at the same instant; exactly the limit gets through."""
    budgets = replicas(**limits)
    start = threading.Barrier(ATTEMPTS)

    def attempt(budget):
        start.wait()
        try:
            budget.acquire(*size)
            return True
        except (WouldWait, DailyCapReached):
            return False

    with ThreadPoolExecutor(ATTEMPTS) as pool:
        granted = list(pool.map(attempt, [budgets[i % 2] for i in range(ATTEMPTS)]))
    assert sum(granted) == allowed
    assert budgets[0].used_last_minute() == budgets[1].used_last_minute() == (allowed, allowed * size[0])
    assert budgets[0].requests_today == budgets[1].requests_today == allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("limits, size, allowed", CASES.values(), ids=CASES.keys())
async def test_replicas_never_reserve_more_than_the_limit_together_async(replicas, limits, size, allowed):
    budgets = replicas(**limits)

    async def attempt(budget):
        try:
            await budget.aacquire(*size)
            return True
        except (WouldWait, DailyCapReached):
            return False

    granted = await asyncio.gather(*(attempt(budgets[i % 2]) for i in range(ATTEMPTS)))
    assert sum(granted) == allowed
    assert budgets[1].used_last_minute() == (allowed, allowed * size[0])


def test_a_replica_waits_for_the_other_replicas_usage(replicas, budget_clock):
    a, b = replicas(tpm=8000)
    for budget in (a, b):
        budget.sleep = budget_clock.sleep
    a.acquire(5000)
    budget_clock.now += 10
    b.acquire(5000)                     # 5000 + 5000 > 8000: b waits for a's request to leave the window
    assert budget_clock.sleeps == [50.0]


def test_recorded_usage_corrects_the_shared_counters(replicas, budget_clock):
    a, b = replicas(tpm=8000, otpm=1000, tpd=100_000)
    first = a.acquire(5000, 600)
    a.record(first, 6000, 210)                                      # more input, far less output than reserved
    assert b.used_last_minute() == (1, 6000) and b.tokens_today == 6000
    b.acquire(1500, 600)                # 6000 + 1500 <= 8000 and 210 + 600 <= 1000: fits only after the record
    with pytest.raises(WouldWait):
        b.acquire(1000, 0)              # 7500 + 1000 > 8000
    a.record(first, 1000)               # the handle tracks the last value: records can be repeated
    assert a.used_last_minute() == (2, 2500) and a.tokens_today == 2500
    b.acquire(1000, 0)


def test_late_record_corrects_its_own_day_only(replicas, budget_clock):
    budget_clock.now = datetime(2026, 10, 18, 23, 59, 50, tzinfo=timezone.utc).timestamp()
    a, b = replicas(tpd=100_000)
    late = a.acquire(5000)
    budget_clock.now += 70                                          # left the window, and it's a new day
    b.acquire(100)
    a.record(late, 6000)                # yesterday's counter expired at midnight: nothing to correct
    assert b.tokens_today == 100 and b.used_last_minute() == (1, 100)
    budget_clock.now += 30
    today = a.acquire(2000)
    budget_clock.now += 61
    a.record(today, 2500)               # out of the window, but still today's request
    assert b.tokens_today == 2600


def test_daily_cap_resets_at_utc_midnight(replicas, make_redis, budget_clock):
    budget_clock.now = datetime(2026, 10, 18, 23, 59, 30, tzinfo=timezone.utc).timestamp()
    a, b = replicas(rpd=2, tpd=10_000)
    a.acquire(1000)
    b.acquire(1000)
    for budget in (a, b):
        with pytest.raises(DailyCapReached, match="request"):
            budget.acquire(1)

    redis = make_redis()
    day_keys = redis.keys("*:day:2026-10-18:*")
    midnight = int(datetime(2026, 10, 19, tzinfo=timezone.utc).timestamp())
    assert len(day_keys) == 2 and all(redis.expiretime(k) == midnight for k in day_keys)

    budget_clock.now += 31                                          # 00:00:01 UTC
    assert redis.keys("*:day:*") == []                              # expired, not just ignored
    assert a.requests_today == b.requests_today == 0
    a.acquire(1000)
    assert b.requests_today == 1 and b.tokens_today == 1000


def test_window_keys_expire_when_a_model_goes_idle(replicas, make_redis, budget_clock):
    a, = replicas(n=1, rpm=30)
    a.acquire(10)
    redis = make_redis()
    assert len(redis.keys("cdp:rate:{openai/gpt-oss-120b}:*")) == 5  # 3 window sets + 2 day counters
    budget_clock.now += 121
    assert sorted(k.decode().rsplit(":", 1)[1] for k in redis.keys("*")) == ["requests", "tokens"]  # day only


def test_streamlit_app_and_eval_runner_keep_in_memory_budgets(monkeypatch, make_redis):
    """Only the API (REDIS_URL set) passes a Redis registry; the default stays this process's memory."""
    from agent.rate_limit import BudgetRegistry
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("REDIS_URL", "redis://redis.internal:6379/0")  # not read implicitly
    local = build_llm_client()
    assert type(local.budget) is ModelBudget and local.budget is budget_for(local.model) is BUDGETS.get(local.model)
    client = make_redis()
    shared = build_llm_client(budgets=BudgetRegistry(client))
    assert isinstance(shared.budget, RedisModelBudget) and shared.budget.client is client

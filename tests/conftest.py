import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from services.model_handler import ModelService  # noqa: E402


class FixedLogitsModel(nn.Module):
    """Stands in for StudentEfficientNet: returns the same logits for every image."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1), None


@pytest.fixture
def calibration_file(tmp_path):
    """Writes a calibration.json with the given temperature and returns its path."""
    def write(temperature):
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps({"temperature": temperature}), encoding="utf-8")
        return path
    return write


@pytest.fixture
def make_service(monkeypatch):
    """ModelService whose network is a FixedLogitsModel: no checkpoint, Streamlit cache or GPU needed."""
    def make(logits, calibration_path):
        monkeypatch.setattr(ModelService, "_load_model", lambda self, path: FixedLogitsModel(logits))
        return ModelService("unused.pth", calibration_path=calibration_path)
    return make


# ---- Upload store / report cache backends (in memory, or Redis via fakeredis: no network) ----

def set_fakeredis_time(monkeypatch, time_fn):
    """fakeredis sets each database's time from time.time() in fakeredis._socket._base before every command."""
    import fakeredis._socket._base as fake_socket
    monkeypatch.setattr(fake_socket, "time", type("FakeTime", (), {"time": staticmethod(time_fn)}))


@pytest.fixture
def redis_clock(monkeypatch):
    """One fake clock for fakeredis's key expiry and the stores, so tests move time instead of sleeping."""
    now = [1_000_000.0]
    set_fakeredis_time(monkeypatch, lambda: now[0])
    return now


@pytest.fixture
def redis_server():
    import fakeredis
    return fakeredis.FakeServer()


@pytest.fixture
def make_redis(redis_server):
    """A new client on this test's fake server: two clients behave like two replicas sharing one Redis."""
    import fakeredis
    return lambda: fakeredis.FakeRedis(server=redis_server)


@pytest.fixture(params=["memory", "redis"])
def backend(request):
    return request.param


@pytest.fixture
def report_cache(backend, make_redis):
    from agent.agent import RedisReportCache, ReportCache
    return ReportCache() if backend == "memory" else RedisReportCache(make_redis())


@pytest.fixture
def make_upload_store(backend, make_redis, redis_clock):
    """make(ttl_s, max_items) -> UploadStore or RedisUploadStore, both on the redis_clock."""
    from api.state import RedisUploadStore, UploadStore

    def make(ttl_s=3600, max_items=100):
        clock = lambda: redis_clock[0]  # noqa: E731
        if backend == "memory":
            return UploadStore(ttl_s, max_items, clock=clock)
        return RedisUploadStore(make_redis(), ttl_s, max_items, clock=clock)
    return make


# ---- LLM rate-limit budgets (in memory, or Redis via fakeredis) ----

class FakeClock:
    """Wall clock for budgets: sleep()/asleep() record the wait and move time forward instead of waiting."""

    def __init__(self, now):
        self.now, self.sleeps = now, []

    def __call__(self):
        return self.now

    def today(self):
        return datetime.fromtimestamp(self.now, timezone.utc).date()

    def sleep(self, s):
        self.sleeps.append(round(s, 2))
        self.now += s

    async def asleep(self, s):
        self.sleep(s)


@pytest.fixture
def budget_clock(monkeypatch):
    """Starts at 2026-10-18 12:00 UTC; fakeredis's key expiry follows it too."""
    clock = FakeClock(datetime(2026, 10, 18, 12, tzinfo=timezone.utc).timestamp())
    set_fakeredis_time(monkeypatch, clock)
    return clock


@pytest.fixture
def make_budget(backend, make_redis, budget_clock):
    """make(model="m", **limits) -> ModelBudget or RedisModelBudget on budget_clock.

    Redis budgets for the same model share counters (each gets its own client, like a replica), so a
    test that builds several unrelated budgets gives them different model names.
    """
    from agent.rate_limit import ModelBudget, RedisModelBudget

    def make(model="m", **kwargs):
        kwargs = {"clock": budget_clock, "sleep": budget_clock.sleep, "async_sleep": budget_clock.asleep, **kwargs}
        if backend == "memory":
            return ModelBudget(model, today=budget_clock.today, **kwargs)
        return RedisModelBudget(make_redis(), model, **kwargs)
    return make

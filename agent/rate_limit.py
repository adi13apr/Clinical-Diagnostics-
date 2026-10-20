"""Client-side, token-aware rate limiting per model.

A ModelBudget enforces requests/min, tokens/min and (where a provider has one) output
tokens/min over a sliding 60 s window, plus daily request and token caps. Callers reserve an
estimate before a request and record the real usage afterwards. Budgets are shared per model
(budget_for), so the report agent and the eval judge never exceed a model's limits together.

The same budget serves sync callers (the app: acquire, time.sleep) and async callers (the eval
runner: aacquire, asyncio.sleep, an asyncio.Lock and a per-model asyncio.Semaphore). Both go
through _try_reserve, so they share one set of counters.
"""
import asyncio
import json
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

WINDOW_S = 60.0


class DailyCapReached(Exception):
    """A configured daily request/token cap would be exceeded."""

    def __init__(self, model, what, used, cap):
        super().__init__(f"{model}: daily {what} cap reached ({used}/{cap})")
        self.model, self.what, self.used, self.cap = model, what, used, cap


def estimate_tokens(*parts):
    """Rough prompt size (about 3.5 characters per token for English/JSON)."""
    return int(sum(len(p if isinstance(p, str) else json.dumps(p, default=str)) for p in parts) / 3.5) + 1


class ModelBudget:
    def __init__(self, model, rpm=None, tpm=None, rpd=None, tpd=None, otpm=None, max_concurrent=1,
                 clock=time.monotonic, sleep=time.sleep, async_sleep=asyncio.sleep,
                 today=lambda: datetime.now(timezone.utc).date()):
        self.model, self.rpm, self.tpm, self.rpd, self.tpd, self.otpm = model, rpm, tpm, rpd, tpd, otpm
        self.max_concurrent = max_concurrent
        self.clock, self.sleep, self.async_sleep, self.today = clock, sleep, async_sleep, today
        self._events = deque()  # [time, tokens, output_tokens] per request in the last minute
        self._day, self.requests_today, self.tokens_today = None, 0, 0
        self._lock = threading.Lock()
        self._async = {}  # event loop id -> (asyncio.Lock, asyncio.Semaphore)

    def _roll_day(self):
        if self._day != self.today():
            self._day, self.requests_today, self.tokens_today = self.today(), 0, 0

    def _prune(self, now):
        while self._events and now - self._events[0][0] >= WINDOW_S:
            self._events.popleft()

    def _try_reserve(self, estimated_tokens, estimated_output):
        """Reserve now if the request fits: (event, 0). Otherwise (None, seconds to wait). Never sleeps."""
        with self._lock:
            self._roll_day()
            if self.rpd is not None and self.requests_today + 1 > self.rpd:
                raise DailyCapReached(self.model, "request", self.requests_today, self.rpd)
            if self.tpd is not None and self.tokens_today + estimated_tokens > self.tpd:
                raise DailyCapReached(self.model, "token", self.tokens_today, self.tpd)
            now = self.clock()
            self._prune(now)
            over_rpm = self.rpm is not None and len(self._events) >= self.rpm
            used = sum(e[1] for e in self._events)
            over_tpm = self.tpm is not None and self._events and used + estimated_tokens > self.tpm
            used_out = sum(e[2] for e in self._events)
            over_otpm = self.otpm is not None and self._events and used_out + estimated_output > self.otpm
            if over_rpm or over_tpm or over_otpm:
                return None, max(WINDOW_S - (now - self._events[0][0]), 0.01)  # until the oldest request expires
            event = [now, estimated_tokens, estimated_output]
            self._events.append(event)
            self.requests_today += 1
            self.tokens_today += estimated_tokens
            return event, 0.0

    def acquire(self, estimated_tokens, estimated_output=0):
        """Block (time.sleep) until a request fits; returns a handle for record().

        estimated_output (usually max_completion_tokens) counts against otpm: Groq checks the
        request's expected output against its output-tokens-per-minute limit before running it.
        """
        while True:
            event, wait = self._try_reserve(estimated_tokens, estimated_output)
            if event is not None:
                return event
            self.sleep(wait)

    def _primitives(self):
        """asyncio.Lock + Semaphore for the running loop (asyncio objects can't be shared across loops)."""
        key = id(asyncio.get_running_loop())
        if key not in self._async:
            self._async = {key: (asyncio.Lock(), asyncio.Semaphore(self.max_concurrent))}  # drop old loops
        return self._async[key]

    async def aacquire(self, estimated_tokens, estimated_output=0):
        """Async acquire: waits with asyncio.sleep, so other coroutines run meanwhile.

        The asyncio.Lock makes waiters for this model queue in order instead of racing.
        """
        lock, _ = self._primitives()
        async with lock:
            while True:
                event, wait = self._try_reserve(estimated_tokens, estimated_output)
                if event is not None:
                    return event
                await self.async_sleep(wait)

    @asynccontextmanager
    async def slot(self):
        """At most max_concurrent requests to this model in flight at once (asyncio.Semaphore)."""
        _, semaphore = self._primitives()
        async with semaphore:
            yield

    def record(self, event, actual_tokens, actual_output=None):
        """Replace the reserved estimates with the real token counts."""
        with self._lock:
            if actual_tokens is not None:
                self.tokens_today += actual_tokens - event[1]
                event[1] = actual_tokens
            if actual_output is not None:
                event[2] = actual_output

    def used_last_minute(self):
        with self._lock:
            self._prune(self.clock())
            return len(self._events), sum(e[1] for e in self._events)


_budgets = {}
_budgets_lock = threading.Lock()


def budget_for(model, limits=None):
    """Process-wide shared budget for a model, built from config/llm.json limits on first use."""
    with _budgets_lock:
        if model not in _budgets:
            if limits is None:
                from agent.llm_config import load_llm_config
                limits = load_llm_config()["limits"].get(model, {})
            _budgets[model] = ModelBudget(model, rpm=limits.get("rpm"), tpm=limits.get("tpm"),
                                          rpd=limits.get("rpd"), tpd=limits.get("tpd"), otpm=limits.get("otpm"),
                                          max_concurrent=limits.get("max_concurrent") or 1)
        return _budgets[model]

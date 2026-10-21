"""Client-side, token-aware rate limiting per model.

A budget enforces requests/min, tokens/min and (where a provider has one) output tokens/min over a
sliding 60 s window, plus daily request and token caps. Callers reserve an estimate before a request
and record the real usage afterwards. Budgets are shared per model (budget_for / BudgetRegistry),
so the report agent and the eval judge never exceed a model's limits together.

Two backends keep the counters:
  ModelBudget       this process's memory (the Streamlit app, the eval runner, the API without REDIS_URL)
  RedisModelBudget  Redis, shared by every process using it (API replicas with REDIS_URL set)

Both serve sync callers (acquire, time.sleep) and async callers (aacquire, asyncio.sleep, an
asyncio.Lock and a per-model asyncio.Semaphore) through _try_reserve, so they share one set of counters.
"""
import asyncio
import json
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

WINDOW_S = 60.0


class DailyCapReached(Exception):
    """A configured daily request/token cap would be exceeded."""

    def __init__(self, model, what, used, cap):
        super().__init__(f"{model}: daily {what} cap reached ({used}/{cap})")
        self.model, self.what, self.used, self.cap = model, what, used, cap


def estimate_tokens(*parts):
    """Rough prompt size (about 3.5 characters per token for English/JSON)."""
    return int(sum(len(p if isinstance(p, str) else json.dumps(p, default=str)) for p in parts) / 3.5) + 1


class BaseBudget:
    """A model's limits and how to wait for them. Subclasses keep the counters.

    They implement _try_reserve (reserve now, or say how long to wait; never sleeps), record and
    used_last_minute.
    """

    def __init__(self, model, rpm=None, tpm=None, rpd=None, tpd=None, otpm=None, max_concurrent=1,
                 clock=time.monotonic, sleep=time.sleep, async_sleep=asyncio.sleep):
        self.model, self.rpm, self.tpm, self.rpd, self.tpd, self.otpm = model, rpm, tpm, rpd, tpd, otpm
        self.max_concurrent = max_concurrent
        self.clock, self.sleep, self.async_sleep = clock, sleep, async_sleep
        self._async = {}  # event loop id -> (asyncio.Lock, asyncio.Semaphore)

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

    async def _atry_reserve(self, estimated_tokens, estimated_output):
        return self._try_reserve(estimated_tokens, estimated_output)

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
                event, wait = await self._atry_reserve(estimated_tokens, estimated_output)
                if event is not None:
                    return event
                await self.async_sleep(wait)

    @asynccontextmanager
    async def slot(self):
        """At most max_concurrent requests to this model in flight at once, per process (asyncio.Semaphore)."""
        _, semaphore = self._primitives()
        async with semaphore:
            yield

    async def arecord(self, event, actual_tokens, actual_output=None):
        """Async record() for callers on an event loop."""
        self.record(event, actual_tokens, actual_output)


class ModelBudget(BaseBudget):
    """Counters in this process's memory."""

    def __init__(self, model, rpm=None, tpm=None, rpd=None, tpd=None, otpm=None, max_concurrent=1,
                 clock=time.monotonic, sleep=time.sleep, async_sleep=asyncio.sleep,
                 today=lambda: datetime.now(timezone.utc).date()):
        super().__init__(model, rpm, tpm, rpd, tpd, otpm, max_concurrent, clock, sleep, async_sleep)
        self.today = today
        self._events = deque()  # [time, tokens, output_tokens] per request in the last minute
        self._day, self.requests_today, self.tokens_today = None, 0, 0
        self._lock = threading.Lock()

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


# KEYS: requests, tokens, output (sorted sets), day requests, day tokens (counters)
# ARGV: now, window start (scores <= it are pruned), id, tokens, output, rpm, tpm, otpm, rpd, tpd,
#       day expiry (unix s), window key ttl (ms). An empty limit means "no limit".
# Returns {"ok"}, {"wait", oldest score} or {"day", "request"|"token", used, cap}.
RESERVE_SCRIPT = """
local tokens, output = tonumber(ARGV[4]), tonumber(ARGV[5])
local rpm, tpm, otpm = tonumber(ARGV[6]), tonumber(ARGV[7]), tonumber(ARGV[8])
local rpd, tpd = tonumber(ARGV[9]), tonumber(ARGV[10])

local requests_today = tonumber(redis.call('GET', KEYS[4]) or '0')
if rpd and requests_today + 1 > rpd then return {'day', 'request', requests_today, rpd} end
local tokens_today = tonumber(redis.call('GET', KEYS[5]) or '0')
if tpd and tokens_today + tokens > tpd then return {'day', 'token', tokens_today, tpd} end

local function total(key)
  local sum = 0
  for _, member in ipairs(redis.call('ZRANGE', key, 0, -1)) do
    sum = sum + tonumber(string.match(member, '|(%d+)$'))
  end
  return sum
end

for i = 1, 3 do redis.call('ZREMRANGEBYSCORE', KEYS[i], '-inf', ARGV[2]) end
local count = redis.call('ZCARD', KEYS[1])
if (rpm and count >= rpm)
    or (tpm and count > 0 and total(KEYS[2]) + tokens > tpm)
    or (otpm and count > 0 and total(KEYS[3]) + output > otpm) then
  return {'wait', redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')[2]}
end

redis.call('ZADD', KEYS[1], ARGV[1], ARGV[3])
redis.call('ZADD', KEYS[2], ARGV[1], ARGV[3] .. '|' .. ARGV[4])
redis.call('ZADD', KEYS[3], ARGV[1], ARGV[3] .. '|' .. ARGV[5])
for i = 1, 3 do redis.call('PEXPIRE', KEYS[i], ARGV[12]) end
redis.call('INCR', KEYS[4])
redis.call('INCRBY', KEYS[5], tokens)
redis.call('EXPIREAT', KEYS[4], ARGV[11])
redis.call('EXPIREAT', KEYS[5], ARGV[11])
return {'ok'}
"""

# KEYS: tokens, output (sorted sets), the reservation day's tokens counter
# ARGV: id, reserved tokens, actual tokens or "", reserved output, actual output or ""
RECORD_SCRIPT = """
local function replace(key, old, new)
  local score = redis.call('ZSCORE', key, ARGV[1] .. '|' .. old)
  if score then
    redis.call('ZREM', key, ARGV[1] .. '|' .. old)
    redis.call('ZADD', key, score, ARGV[1] .. '|' .. new)
  end
end
if ARGV[3] ~= '' then
  replace(KEYS[1], ARGV[2], ARGV[3])
  if redis.call('EXISTS', KEYS[3]) == 1 then
    redis.call('INCRBY', KEYS[3], tonumber(ARGV[3]) - tonumber(ARGV[2]))
  end
end
if ARGV[5] ~= '' then replace(KEYS[2], ARGV[4], ARGV[5]) end
return 1
"""


class Reservation:
    """Handle from RedisModelBudget.acquire: what record() needs to find the reservation again."""

    def __init__(self, id, day, tokens, output):
        self.id, self.day, self.tokens, self.output = id, day, tokens, output


def _amount(member):
    """Token count from a sorted-set member "id|tokens"."""
    return int(member.rsplit(b"|", 1)[1])


def _limit(value):
    return "" if value is None else str(value)


class RedisModelBudget(BaseBudget):
    """Counters in Redis, so every process using the same Redis shares one budget per model.

    Keys (prefix "cdp:" by default; {model} is a hash tag, so a model's keys share one cluster slot):
      rate:{model}:requests                       sorted set: reservation id, scored by time
      rate:{model}:tokens / :output               sorted sets: "id|tokens", scored by time
      rate:{model}:day:YYYY-MM-DD:requests        counter; expires at the next UTC midnight
      rate:{model}:day:YYYY-MM-DD:tokens          counter; expires at the next UTC midnight

    A reservation is one Lua script (prune, check every limit, add), so two replicas can't both take
    the last tokens. Scores come from clock, a wall clock: time.monotonic differs between processes,
    so replicas' clocks only need to agree to well under a second. max_concurrent (slot) and the
    asyncio.Lock stay per process. Redis errors propagate (the report falls back to the template).
    """

    def __init__(self, client, model, rpm=None, tpm=None, rpd=None, tpd=None, otpm=None, max_concurrent=1,
                 clock=time.time, sleep=time.sleep, async_sleep=asyncio.sleep, prefix="cdp:"):
        super().__init__(model, rpm, tpm, rpd, tpd, otpm, max_concurrent, clock, sleep, async_sleep)
        self.client = client
        self.key = f"{prefix}rate:{{{model}}}"
        self._reserve = client.register_script(RESERVE_SCRIPT)
        self._record = client.register_script(RECORD_SCRIPT)

    def _day(self, now):
        return datetime.fromtimestamp(now, timezone.utc).date()

    def _window_keys(self):
        return [f"{self.key}:requests", f"{self.key}:tokens", f"{self.key}:output"]

    def _day_keys(self, day):
        return [f"{self.key}:day:{day.isoformat()}:requests", f"{self.key}:day:{day.isoformat()}:tokens"]

    def _try_reserve(self, estimated_tokens, estimated_output):
        """Reserve now if the request fits: (Reservation, 0). Otherwise (None, seconds to wait). Never sleeps."""
        now = self.clock()
        day = self._day(now)
        midnight = datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc)
        rid, tokens, output = uuid.uuid4().hex, int(estimated_tokens), int(estimated_output)
        result = self._reserve(
            keys=self._window_keys() + self._day_keys(day),
            args=[repr(now), repr(now - WINDOW_S), rid, tokens, output, _limit(self.rpm), _limit(self.tpm),
                  _limit(self.otpm), _limit(self.rpd), _limit(self.tpd), int(midnight.timestamp()),
                  int(2 * WINDOW_S * 1000)])
        status = result[0].decode()
        if status == "day":
            raise DailyCapReached(self.model, result[1].decode(), int(result[2]), int(result[3]))
        if status == "wait":
            return None, max(WINDOW_S - (now - float(result[1])), 0.01)  # until the oldest request expires
        return Reservation(rid, day, tokens, output), 0.0

    async def _atry_reserve(self, estimated_tokens, estimated_output):
        return await asyncio.to_thread(self._try_reserve, estimated_tokens, estimated_output)

    def record(self, event, actual_tokens, actual_output=None):
        """Replace the reserved estimates with the real token counts.

        The daily total of the reservation's own day is corrected, even after the request left the
        window; a day whose counter already expired is left alone.
        """
        if actual_tokens is None and actual_output is None:
            return
        tokens = "" if actual_tokens is None else int(actual_tokens)
        output = "" if actual_output is None else int(actual_output)
        self._record(keys=self._window_keys()[1:] + self._day_keys(event.day)[1:],
                     args=[event.id, event.tokens, tokens, event.output, output])
        if tokens != "":
            event.tokens = tokens
        if output != "":
            event.output = output

    async def arecord(self, event, actual_tokens, actual_output=None):
        await asyncio.to_thread(self.record, event, actual_tokens, actual_output)

    def used_last_minute(self):
        requests, tokens, _ = self._window_keys()
        start = f"({self.clock() - WINDOW_S!r}"  # exclusive: a request exactly 60 s old has left the window
        return (self.client.zcount(requests, start, "+inf"),
                sum(_amount(m) for m in self.client.zrangebyscore(tokens, start, "+inf")))

    def _today(self, index):
        return int(self.client.get(self._day_keys(self._day(self.clock()))[index]) or 0)

    @property
    def requests_today(self):
        return self._today(0)

    @property
    def tokens_today(self):
        return self._today(1)


class BudgetRegistry:
    """One budget per model, built from config/llm.json limits on first use.

    Without a Redis client the budgets are ModelBudgets (this process only). With one they are
    RedisModelBudgets, shared by every process using that Redis (the API with REDIS_URL set).
    """

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self._budgets = {}
        self._lock = threading.Lock()

    def get(self, model, limits=None):
        with self._lock:
            if model not in self._budgets:
                if limits is None:
                    from agent.llm_config import load_llm_config
                    limits = load_llm_config()["limits"].get(model, {})
                kwargs = dict(rpm=limits.get("rpm"), tpm=limits.get("tpm"), rpd=limits.get("rpd"),
                              tpd=limits.get("tpd"), otpm=limits.get("otpm"),
                              max_concurrent=limits.get("max_concurrent") or 1)
                self._budgets[model] = (RedisModelBudget(self.redis_client, model, **kwargs)
                                        if self.redis_client is not None else ModelBudget(model, **kwargs))
            return self._budgets[model]


BUDGETS = BudgetRegistry()  # in memory: the Streamlit app and the eval runner


def budget_for(model, limits=None):
    """Process-wide in-memory budget for a model."""
    return BUDGETS.get(model, limits)

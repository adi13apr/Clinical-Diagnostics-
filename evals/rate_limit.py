"""Client-side rate limiting so a live eval stays under the Gemini free-tier quota."""
import threading
import time


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart (shared by everything that holds it)."""

    def __init__(self, requests_per_minute, clock=time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / requests_per_minute
        self.clock, self.sleep = clock, sleep
        self._next = None
        self._lock = threading.Lock()
        self.calls = 0

    def wait(self):
        with self._lock:
            now = self.clock()
            if self._next is not None and now < self._next:
                self.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval
            self.calls += 1


class RateLimitedModels:
    """Wraps client.models so every generate_content call goes through the limiter."""

    def __init__(self, models, limiter):
        self._models, self._limiter = models, limiter

    def generate_content(self, **kwargs):
        self._limiter.wait()
        return self._models.generate_content(**kwargs)


class RateLimitedClient:
    """Object with a .models attribute, usable as GeminiClient(client=...) / GeminiJudge(client=...).

    Holds a reference to the wrapped client: a google-genai Client closes its HTTP connection
    when garbage-collected, so it must outlive the wrapper.
    """

    def __init__(self, client, limiter):
        self.client = client
        self.models = RateLimitedModels(client.models, limiter)


def rate_limited_client(client, limiter):
    return RateLimitedClient(client, limiter)

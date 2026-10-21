"""API state: recent uploads (in this process's memory, or in Redis to share them between replicas)
and the shared model behind one lock."""
import io
import json
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

from PIL import Image

REDIS_CONNECT_TIMEOUT_S = 2.0
REDIS_TIMEOUT_S = 5.0


class Upload:
    """One uploaded image and its prediction, owned by the user who uploaded it."""

    def __init__(self, user_id, data, prediction, created):
        self.user_id = user_id
        self.data = data               # the uploaded bytes (already validated as JPEG/PNG)
        self.prediction = prediction   # ModelService.predict() result
        self.created = created
        self.heatmap_png = None        # filled by the first GET /gradcam
        self.cam_map = None            # filled by the first GET /gradcam/{id}/map: (cam, regions)

    def image(self):
        """Decode again on use: the same bytes give the same pixels, so the report cache still matches."""
        image = Image.open(io.BytesIO(self.data))
        image.load()
        return image


class UploadStore:
    """Recent uploads by trace_id so /gradcam and /report can find the image later.

    Kept in this process's memory, never written to disk, dropped after ttl_s or when the store is
    full (oldest first). A restart or a second replica loses them; the client then predicts again.
    """

    def __init__(self, ttl_s=3600, max_items=100, clock=time.monotonic):
        self.ttl_s, self.max_items, self.clock = ttl_s, max_items, clock
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def put(self, trace_id, user_id, data, prediction):
        with self._lock:
            self._prune()
            self._items[trace_id] = Upload(user_id, data, prediction, self.clock())
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def get(self, trace_id, user_id):
        """The upload, or None if unknown, expired or another user's (callers answer 404 either way)."""
        with self._lock:
            self._prune()
            upload = self._items.get(trace_id)
        return upload if upload is not None and upload.user_id == user_id else None

    def save_heatmap(self, trace_id, png):
        """Keep the Grad-CAM PNG with the upload, so the next /gradcam doesn't recompute it."""
        with self._lock:
            upload = self._items.get(trace_id)
            if upload is not None:
                upload.heatmap_png = png

    def save_cam_map(self, trace_id, cam_map):
        """Keep the raw map (cam, regions) with the upload."""
        with self._lock:
            upload = self._items.get(trace_id)
            if upload is not None:
                upload.cam_map = cam_map

    def _prune(self):
        now = self.clock()
        while self._items:
            trace_id, upload = next(iter(self._items.items()))
            if now - upload.created <= self.ttl_s:
                break
            del self._items[trace_id]

    def __len__(self):
        return len(self._items)


class StoreUnavailable(Exception):
    """The shared store (Redis) couldn't be reached; the API answers 503."""


class RedisUploadStore:
    """UploadStore shared by every API replica through Redis, with the same interface and rules.

    Keys (prefix "cdp:" by default):
      upload:{trace_id}          hash: user_id, data (raw bytes), prediction (JSON), created; expires after ttl_s
      upload:{trace_id}:heatmap  Grad-CAM PNG bytes; expires with the upload
      upload:{trace_id}:cam      JSON {"cam", "regions"}; expires with the upload
      uploads                    sorted set trace_id -> created, for the max_items cap and len()

    Values are bytes or JSON, never pickle. Redis holds them in memory only if it runs without
    persistence (no RDB/AOF); otherwise the X-rays reach Redis's disk.
    """

    def __init__(self, client, ttl_s=3600, max_items=100, prefix="cdp:", clock=time.time):
        self.client, self.ttl_s, self.max_items, self.clock = client, ttl_s, max_items, clock
        self.prefix = prefix
        self.index = f"{prefix}uploads"

    def key(self, trace_id, part=None):
        key = f"{self.prefix}upload:{trace_id}"
        return f"{key}:{part}" if part else key

    def put(self, trace_id, user_id, data, prediction):
        now = self.clock()
        ttl_ms = int(self.ttl_s * 1000)
        with self._errors():
            pipe = self.client.pipeline()  # MULTI/EXEC: the upload never exists without its expiry
            pipe.delete(self.key(trace_id, "heatmap"), self.key(trace_id, "cam"))
            pipe.hset(self.key(trace_id), mapping={
                "user_id": user_id, "data": data, "prediction": json.dumps(prediction), "created": repr(now)})
            pipe.pexpire(self.key(trace_id), ttl_ms)
            pipe.zadd(self.index, {trace_id: now})
            pipe.pexpire(self.index, ttl_ms)
            pipe.execute()
            self._prune(now)
            extra = self.client.zcard(self.index) - self.max_items
            if extra > 0:  # oldest first, across every replica (two puts at once may over/undershoot by one)
                oldest = [member.decode() for member, _ in self.client.zpopmin(self.index, extra)]
                self.client.delete(*[self.key(t, part) for t in oldest for part in (None, "heatmap", "cam")])

    def get(self, trace_id, user_id):
        """The upload, or None if unknown, expired or another user's (callers answer 404 either way)."""
        with self._errors():
            pipe = self.client.pipeline()
            pipe.hgetall(self.key(trace_id))
            pipe.mget(self.key(trace_id, "heatmap"), self.key(trace_id, "cam"))
            fields, (heatmap, cam) = pipe.execute()
        if not fields or fields[b"user_id"].decode() != user_id:
            return None
        upload = Upload(user_id, fields[b"data"], json.loads(fields[b"prediction"]), float(fields[b"created"]))
        upload.heatmap_png = heatmap
        if cam is not None:
            cam = json.loads(cam)
            upload.cam_map = (cam["cam"], cam["regions"])
        return upload

    def save_heatmap(self, trace_id, png):
        self._save_part(trace_id, "heatmap", png)

    def save_cam_map(self, trace_id, cam_map):
        cam, regions = cam_map
        self._save_part(trace_id, "cam", json.dumps({"cam": cam, "regions": regions}))

    def _save_part(self, trace_id, part, value):
        """Expires together with the upload, and never brings back an upload that has already expired."""
        with self._errors():
            remaining_ms = self.client.pttl(self.key(trace_id))
            if remaining_ms > 0:
                self.client.set(self.key(trace_id, part), value, px=remaining_ms)

    def _prune(self, now):
        # Redis expires the uploads themselves; this drops their entries from the index
        self.client.zremrangebyscore(self.index, "-inf", f"({now - self.ttl_s!r}")

    def __len__(self):
        with self._errors():
            self._prune(self.clock())
            return self.client.zcard(self.index)

    @contextmanager
    def _errors(self):
        import redis
        try:
            yield
        except redis.RedisError as e:
            raise StoreUnavailable(f"Shared upload store unavailable ({type(e).__name__})") from e


def connect_redis(url):
    """Redis client for REDIS_URL. Connects on first use; short timeouts so a dead Redis is a quick 503."""
    import redis
    return redis.Redis.from_url(url, socket_connect_timeout=REDIS_CONNECT_TIMEOUT_S, socket_timeout=REDIS_TIMEOUT_S)


def redis_location(client):
    """host:port/db of a Redis client, for logs (never the URL: it may hold a password)."""
    kwargs = client.connection_pool.connection_kwargs
    return f"{kwargs.get('host', kwargs.get('path', '?'))}:{kwargs.get('port', '')}/{kwargs.get('db', 0)}"


class SerializedModel:
    """ModelService wrapper that lets one request use the network at a time.

    Grad-CAM attaches hooks to the shared network, so another request's forward pass in between
    would overwrite its activations. API requests run model work in worker threads, so the calls
    are serialized here. Everything else (temperature, classes, ...) passes straight through.
    """

    def __init__(self, model_service):
        self.service = model_service
        self._lock = threading.RLock()

    def __getattr__(self, name):
        return getattr(self.service, name)

    def predict(self, *args, **kwargs):
        with self._lock:
            return self.service.predict(*args, **kwargs)

    def generate_heatmap(self, *args, **kwargs):
        with self._lock:
            return self.service.generate_heatmap(*args, **kwargs)

    def compute_gradcam(self, *args, **kwargs):
        with self._lock:
            return self.service.compute_gradcam(*args, **kwargs)

"""Process-wide API state: recent uploads (in memory only) and the shared model behind one lock."""
import io
import threading
import time
from collections import OrderedDict

from PIL import Image


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

    def _prune(self):
        now = self.clock()
        while self._items:
            trace_id, upload = next(iter(self._items.items()))
            if now - upload.created <= self.ttl_s:
                break
            del self._items[trace_id]

    def __len__(self):
        return len(self._items)


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

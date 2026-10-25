"""Upload checks shared by the API and the MCP server: a real JPEG/PNG, within the size and pixel limits.

No web framework here. Callers turn ImageRejected into their own error (the API: 400/413/415).
"""
import io

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG"}   # what Pillow finds in the bytes, whatever the client claimed
MAX_IMAGE_PIXELS = 40_000_000       # decompression-bomb guard (a 6000 x 6000 scan is 36M)
DEFAULT_MAX_UPLOAD_MB = 10

# ImageRejected.kind values
NOT_AN_IMAGE = "not_an_image"
TOO_LARGE = "too_large"
EMPTY = "empty"


class ImageRejected(ValueError):
    """The bytes aren't an acceptable image; kind is NOT_AN_IMAGE, TOO_LARGE or EMPTY."""

    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind


def too_large(max_bytes):
    return f"File too large (limit {max_bytes / 1024 / 1024:g} MB)"


def check_size(data, max_bytes):
    """Raise ImageRejected if data is over max_bytes or empty."""
    if len(data) > max_bytes:
        raise ImageRejected(too_large(max_bytes), TOO_LARGE)
    if not data:
        raise ImageRejected("Empty file", EMPTY)


def decode_image(data):
    """Pillow image from the bytes, if they really are a JPEG/PNG of a sane pixel count."""
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise ImageRejected("Only JPEG and PNG images are accepted", NOT_AN_IMAGE)
        if image.size[0] * image.size[1] > MAX_IMAGE_PIXELS:
            raise ImageRejected("Image has too many pixels", TOO_LARGE)
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise ImageRejected("The file is not a readable JPEG or PNG image", NOT_AN_IMAGE)
    return image

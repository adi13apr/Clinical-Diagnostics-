"""Which image files the MCP tools may open.

Only regular .jpg/.jpeg/.png files inside an allowed directory (MCP_ALLOWED_DIRS, default <repo>/data).
Network and device paths are refused before anything touches the filesystem: opening \\\\host\\share on
Windows sends the user's NTLM credentials to that host.
"""
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ALLOWED_DIRS = (REPO / "data",)
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_PATH_CHARS = 1024
DRIVE = re.compile(r"^[A-Za-z]:")


class PathRejected(ValueError):
    """The path can't be used; the message is safe to show to the MCP client."""


def allowed_dirs_from_env(environ=None):
    """Existing directories from MCP_ALLOWED_DIRS (os.pathsep-separated), else the default."""
    env = os.environ if environ is None else environ
    raw = env.get("MCP_ALLOWED_DIRS", "")
    dirs = [Path(p.strip()) for p in raw.split(os.pathsep) if p.strip()] or list(DEFAULT_ALLOWED_DIRS)
    return [d.resolve() for d in dirs if d.is_dir()]


def resolve_image_path(raw, allowed_dirs):
    """The real path of an allowed image file, or PathRejected.

    Relative paths are taken relative to the first allowed directory (never the server's working
    directory, which the MCP client picks). Symlinks and '..' are resolved before the containment check.
    """
    if not allowed_dirs:
        raise PathRejected("No allowed image directory exists. Set MCP_ALLOWED_DIRS to the folder(s) "
                           "holding the X-rays (separated by ';' on Windows, ':' elsewhere).")
    if not isinstance(raw, str) or not raw.strip():
        raise PathRejected("image_path is empty")
    raw = raw.strip()
    if len(raw) > MAX_PATH_CHARS or "\0" in raw:
        raise PathRejected("image_path is not a valid file path")
    if raw.startswith(("\\\\", "//")):
        raise PathRejected("Network and device paths (\\\\server\\share, \\\\?\\...) are not allowed")
    if ":" in DRIVE.sub("", raw):
        raise PathRejected("image_path must not contain ':' except after a drive letter")

    path = Path(raw)
    if not path.is_absolute():
        path = allowed_dirs[0] / path
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise PathRejected(f"No such file: {raw}")
    if not any(resolved.is_relative_to(d) for d in allowed_dirs):
        raise PathRejected(f"{raw} is outside the allowed image directories: "
                           + ", ".join(str(d) for d in allowed_dirs))
    if not resolved.is_file():
        raise PathRejected(f"Not a file: {raw}")
    if resolved.suffix.lower() not in ALLOWED_SUFFIXES:
        raise PathRejected("Only .jpg, .jpeg and .png files are accepted")
    return resolved


def read_limited(path, max_bytes):
    """At most max_bytes + 1 bytes of the file, so an oversized file is detected without reading it all."""
    with open(path, "rb") as f:
        return f.read(max_bytes + 1)

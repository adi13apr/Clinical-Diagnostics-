"""API settings: environment variables first (Hugging Face Spaces), then .streamlit/secrets.toml (local dev)."""
import logging
import os
import socket
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SECRETS_PATH = REPO / ".streamlit" / "secrets.toml"
DEFAULT_MODEL_PATH = REPO / "best_student_model_aligned_final.pth"

# Any port on localhost / 127.0.0.1 (Vite, CRA, ...). Starlette matches the whole origin.
LOCALHOST_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
DEFAULT_MAX_UPLOAD_MB = 10

log = logging.getLogger(__name__)


class Settings:
    def __init__(self, supabase_url=None, supabase_key=None, jwt_secret=None, frontend_urls=(),
                 max_upload_bytes=DEFAULT_MAX_UPLOAD_MB * 1024 * 1024, model_path=DEFAULT_MODEL_PATH,
                 upload_ttl_s=3600, upload_max_items=100, supabase_url_source=None, check_keys_on_startup=False,
                 redis_url=None, served_by=None):
        self.supabase_url = supabase_url.rstrip("/") if supabase_url else None
        self.supabase_key = supabase_key            # anon key; RLS does the access control
        self.jwt_secret = jwt_secret                # only for legacy HS256 projects
        self.frontend_urls = [u.rstrip("/") for u in frontend_urls if u]
        self.max_upload_bytes = int(max_upload_bytes)
        self.model_path = str(model_path)
        self.upload_ttl_s = upload_ttl_s            # how long /gradcam and /report can find an upload
        self.upload_max_items = upload_max_items
        self.supabase_url_source = supabase_url_source  # where the URL came from, for the startup log
        self.check_keys_on_startup = check_keys_on_startup  # fetch the JWKS once at startup (off in tests)
        self.redis_url = redis_url or None          # set: uploads and the report cache live in Redis (shared)
        self.served_by = served_by or None          # set: every response names this replica (X-Served-By)

    @classmethod
    def from_env(cls, environ=None, secrets_path=SECRETS_PATH):
        env = os.environ if environ is None else environ
        secrets = {}
        if Path(secrets_path).exists():
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
        supabase = secrets.get("supabase", {})

        def get(env_name, secret_key):
            return env.get(env_name) or supabase.get(secret_key)

        frontend = get("FRONTEND_URL", "frontend_url") or ""
        env_url, file_url = env.get("SUPABASE_URL"), supabase.get("url")
        if env_url and file_url and env_url.rstrip("/") != file_url.rstrip("/"):
            # A stale SUPABASE_URL in the shell silently wins over secrets.toml; tokens from the
            # project in secrets.toml (and the web app) then can't be verified
            log.warning("SUPABASE_URL in the environment (%s) differs from [supabase] url in %s (%s). "
                        "The API uses the environment's value; unset SUPABASE_URL and SUPABASE_KEY to use "
                        "secrets.toml.", env_url, secrets_path, file_url)
        source = "environment variable SUPABASE_URL" if env_url else (f"{secrets_path}" if file_url else None)
        return cls(
            supabase_url=env_url or file_url,
            supabase_url_source=source,
            check_keys_on_startup=True,
            supabase_key=env.get("SUPABASE_KEY") or env.get("SUPABASE_ANON_KEY") or supabase.get("key"),
            jwt_secret=get("SUPABASE_JWT_SECRET", "jwt_secret"),
            frontend_urls=[u.strip() for u in frontend.split(",")],
            max_upload_bytes=float(env.get("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)) * 1024 * 1024,
            model_path=env.get("MODEL_PATH") or DEFAULT_MODEL_PATH,
            redis_url=env.get("REDIS_URL") or secrets.get("redis", {}).get("url"),
            # Opt-in (docker-compose.yml): the container's hostname, so a client behind the load
            # balancer can see which replica answered. Off by default, so public hosts don't show it.
            served_by=socket.gethostname() if env.get("SERVED_BY_HEADER", "").lower() in ("1", "true") else None,
        )

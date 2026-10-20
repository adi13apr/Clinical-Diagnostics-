"""API settings: environment variables first (Hugging Face Spaces), then .streamlit/secrets.toml (local dev)."""
import os
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SECRETS_PATH = REPO / ".streamlit" / "secrets.toml"
DEFAULT_MODEL_PATH = REPO / "best_student_model_aligned_final.pth"

# Any port on localhost / 127.0.0.1 (Vite, CRA, ...). Starlette matches the whole origin.
LOCALHOST_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
DEFAULT_MAX_UPLOAD_MB = 10


class Settings:
    def __init__(self, supabase_url=None, supabase_key=None, jwt_secret=None, frontend_urls=(),
                 max_upload_bytes=DEFAULT_MAX_UPLOAD_MB * 1024 * 1024, model_path=DEFAULT_MODEL_PATH,
                 upload_ttl_s=3600, upload_max_items=100):
        self.supabase_url = supabase_url.rstrip("/") if supabase_url else None
        self.supabase_key = supabase_key            # anon key; RLS does the access control
        self.jwt_secret = jwt_secret                # only for legacy HS256 projects
        self.frontend_urls = [u.rstrip("/") for u in frontend_urls if u]
        self.max_upload_bytes = int(max_upload_bytes)
        self.model_path = str(model_path)
        self.upload_ttl_s = upload_ttl_s            # how long /gradcam and /report can find an upload
        self.upload_max_items = upload_max_items

    @classmethod
    def from_env(cls, environ=None, secrets_path=SECRETS_PATH):
        env = os.environ if environ is None else environ
        supabase = {}
        if Path(secrets_path).exists():
            with open(secrets_path, "rb") as f:
                supabase = tomllib.load(f).get("supabase", {})

        def get(env_name, secret_key):
            return env.get(env_name) or supabase.get(secret_key)

        frontend = get("FRONTEND_URL", "frontend_url") or ""
        return cls(
            supabase_url=get("SUPABASE_URL", "url"),
            supabase_key=env.get("SUPABASE_KEY") or env.get("SUPABASE_ANON_KEY") or supabase.get("key"),
            jwt_secret=get("SUPABASE_JWT_SECRET", "jwt_secret"),
            frontend_urls=[u.strip() for u in frontend.split(",")],
            max_upload_bytes=float(env.get("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)) * 1024 * 1024,
            model_path=env.get("MODEL_PATH") or DEFAULT_MODEL_PATH,
        )

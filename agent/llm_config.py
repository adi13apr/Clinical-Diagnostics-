"""Provider selection and keys (config/llm.json + st.secrets / env)."""
import json
import os
from pathlib import Path

LLM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llm.json"


def load_llm_config(path=LLM_CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _secret(section, key):
    try:
        import streamlit as st
        return dict(st.secrets.get(section, {})).get(key)
    except Exception:
        return None  # no secrets file outside Streamlit


def provider_settings(config=None):
    """(provider name, settings dict with resolved 'api_key') for the configured provider."""
    config = config or load_llm_config()
    name = os.environ.get("LLM_PROVIDER") or config["provider"]
    if name not in config["providers"]:
        raise ValueError(f"Unknown LLM provider {name!r}; expected one of {sorted(config['providers'])}")
    settings = dict(config["providers"][name])
    settings["api_key"] = _secret(settings["secret_section"], "api_key") or os.environ.get(settings["env_key"])
    # Optional per-deployment model overrides, e.g. [groq] report_model = "..."
    for role in ("report_model", "judge_model"):
        settings[role] = _secret(settings["secret_section"], role) or settings[role]
    return name, settings


def request_options(model, config=None):
    """API request options for a model; keys starting with '_' are comments and never sent."""
    options = (config or load_llm_config()).get("request_options", {}).get(model, {})
    return {k: v for k, v in options.items() if not k.startswith("_")}

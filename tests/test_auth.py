"""Google sign-in (Supabase OAuth with PKCE): URL building, code exchange, and the redirect callback."""
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from streamlit.testing.v1 import AppTest
from supabase_auth.helpers import generate_pkce_challenge

from services.auth_handler import AuthService, OAuthStateError, PendingOAuthStore, strip_query

SUPABASE_URL = "https://proj.supabase.co"


class FakeAuth:
    def __init__(self):
        self.exchanges = []

    def exchange_code_for_session(self, params):
        self.exchanges.append(params)
        return SimpleNamespace(user=SimpleNamespace(email="dr@example.com"), session=object())


def make_service(pending=None):
    return AuthService(client=SimpleNamespace(auth=FakeAuth()), url=SUPABASE_URL, pending=pending or PendingOAuthStore())


def state_of(url):
    redirect_to = parse_qs(urlsplit(url).query)["redirect_to"][0]
    return parse_qs(urlsplit(redirect_to).query)["oauth_state"][0]


def test_sign_in_url_uses_pkce_and_returns_to_the_app():
    pending = PendingOAuthStore()
    url = make_service(pending).google_sign_in_url("https://clinical-diagnosis.streamlit.app/?old=1")
    parts = urlsplit(url)
    q = parse_qs(parts.query)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == f"{SUPABASE_URL}/auth/v1/authorize"
    assert q["provider"] == ["google"] and q["code_challenge_method"] == ["s256"]
    redirect_to = q["redirect_to"][0]
    assert redirect_to.startswith("https://clinical-diagnosis.streamlit.app/?oauth_state=") and "old=1" not in redirect_to
    verifier = pending._items[state_of(url)][0]
    assert q["code_challenge"] == [generate_pkce_challenge(verifier)]  # the challenge matches the stored verifier


def test_states_are_random_and_single_use():
    service = make_service()
    first, second = (service.google_sign_in_url("http://localhost:8501/") for _ in range(2))
    assert state_of(first) != state_of(second)
    user = service.complete_google_sign_in("auth-code-1", state_of(first))
    assert user.email == "dr@example.com"
    verifier = service.client.auth.exchanges[0]["code_verifier"]
    assert service.client.auth.exchanges[0] == {"auth_code": "auth-code-1", "code_verifier": verifier}
    with pytest.raises(OAuthStateError):  # replaying the same state fails
        service.complete_google_sign_in("auth-code-1", state_of(first))


def test_unknown_or_expired_state_is_rejected():
    now = [0.0]
    service = make_service(PendingOAuthStore(ttl_s=600, clock=lambda: now[0]))
    with pytest.raises(OAuthStateError):
        service.complete_google_sign_in("code", "not-a-real-state")
    url = service.google_sign_in_url("http://localhost:8501/")
    now[0] = 601
    with pytest.raises(OAuthStateError, match="expired"):
        service.complete_google_sign_in("code", state_of(url))
    assert service.client.auth.exchanges == []  # nothing was sent to Supabase


def test_pending_store_is_bounded():
    store = PendingOAuthStore(max_items=2)
    for s in ("a", "b", "c"):
        store.put(s, f"v-{s}")
    assert store.pop("a") is None and store.pop("c") == "v-c"


def test_strip_query():
    assert strip_query("https://x.streamlit.app/?a=1#f") == "https://x.streamlit.app/"
    assert strip_query("http://localhost:8501") == "http://localhost:8501/"


# ---- The redirect callback, through Streamlit (AppTest runs the script in this process) ----

def callback_app():
    from types import SimpleNamespace
    import streamlit as st
    from services.auth_handler import AuthService
    from ui.login_pages import handle_oauth_callback

    class Auth:
        def exchange_code_for_session(self, params):
            if params["auth_code"] == "bad":
                raise RuntimeError("invalid grant")
            return SimpleNamespace(user=SimpleNamespace(email="dr@example.com"))

    if "user" not in st.session_state:
        st.session_state.user = None
    service = AuthService(client=SimpleNamespace(auth=Auth()), url="https://proj.supabase.co")
    if not st.session_state.user:
        handle_oauth_callback(service)
    st.write(f"user: {st.session_state.user.email if st.session_state.user else None}")


def run_callback(**params):
    at = AppTest.from_function(callback_app)
    for k, v in params.items():
        at.query_params[k] = v
    return at.run(timeout=60)


def test_callback_signs_the_user_in_and_clears_the_code():
    url = AuthService(client=SimpleNamespace(auth=FakeAuth()), url=SUPABASE_URL).google_sign_in_url(
        "http://localhost:8501/")  # uses the shared server-side store, like the real app
    at = run_callback(code="good", oauth_state=state_of(url))
    assert not at.exception and not at.error
    assert at.session_state.user.email == "dr@example.com"
    assert dict(at.query_params) == {}


@pytest.mark.parametrize("params,message", [
    ({"error": "access_denied", "error_description": "User cancelled"}, "User cancelled"),
    ({"code": "good"}, "without its sign-in state"),
    ({"code": "good", "oauth_state": "forged"}, "expired or was already used"),
])
def test_callback_errors_are_shown(params, message):
    at = run_callback(**params)
    assert not at.exception
    assert message in at.error[0].value and at.session_state.user is None
    assert dict(at.query_params) == {}


def test_failed_exchange_is_shown():
    url = AuthService(client=SimpleNamespace(auth=FakeAuth()), url=SUPABASE_URL).google_sign_in_url(
        "http://localhost:8501/")
    at = run_callback(code="bad", oauth_state=state_of(url))
    assert "invalid grant" in at.error[0].value and at.session_state.user is None


def login_app():
    from types import SimpleNamespace
    from services.auth_handler import AuthService
    from ui.login_pages import show_login_ui
    show_login_ui(AuthService(client=SimpleNamespace(auth=None), url="https://proj.supabase.co"))


def test_login_page_without_known_app_url_still_renders(monkeypatch):
    # No redirect_url and no st.context.url: email login must keep working. Patched so the test doesn't
    # depend on a developer's local .streamlit/secrets.toml (AppTest reads it).
    monkeypatch.setattr("ui.login_pages.app_base_url", lambda: None)
    at = AppTest.from_function(login_app).run(timeout=60)
    assert not at.exception
    google = [b for b in at.button if b.label == "Continue with Google"]
    assert google and google[0].disabled
    assert any(b.label == "Log In" for b in at.button)


def test_login_page_links_to_supabase_google_authorize(monkeypatch):
    monkeypatch.setattr("ui.login_pages.app_base_url", lambda: "https://clinical-diagnosis.streamlit.app/")
    at = AppTest.from_function(login_app).run(timeout=60)
    assert not at.exception
    link = at.get("link_button")[0].proto
    assert link.label == "Continue with Google"
    assert link.url.startswith("https://proj.supabase.co/auth/v1/authorize?") and "provider=google" in link.url
    at.run(timeout=60)
    assert at.get("link_button")[0].proto.url == link.url  # reruns reuse the same sign-in state


def test_privacy_page_is_public():
    # Served before any service setup, so it works without secrets or a login (Google links to it)
    at = AppTest.from_file("../app.py")  # relative to this test file
    at.query_params["page"] = "privacy"
    at.run(timeout=120)
    assert not at.exception
    assert at.title[0].value == "Privacy policy"
    text = " ".join(m.value for m in at.markdown)
    assert "not stored" in text and "Groq" in text and "github.com/adi13apr/Clinical-Diagnostics-/issues" in text


def test_login_page_links_to_privacy_policy():
    at = AppTest.from_function(login_app).run(timeout=60)
    assert any("?page=privacy" in m.value for m in at.markdown)


def test_no_code_means_nothing_happens():
    at = run_callback()
    assert not at.error and at.session_state.user is None

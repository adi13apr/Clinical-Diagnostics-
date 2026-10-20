import secrets
import threading
import time
from urllib.parse import urlencode, urlsplit, urlunsplit

import streamlit as st
from supabase import create_client
from supabase_auth.helpers import generate_pkce_challenge, generate_pkce_verifier

# How long a "Continue with Google" link stays valid (Google sign-in + consent must finish within it)
OAUTH_STATE_TTL_S = 600


class OAuthStateError(Exception):
    """The Google sign-in came back with an unknown, expired or already-used state."""


class PendingOAuthStore:
    """PKCE code verifiers for sign-ins in progress, keyed by a random single-use state.

    Returning from Google starts a fresh Streamlit session, so the verifier can't live in
    session_state or in the Supabase client's memory. It is kept here, server-side, and only
    the random state travels in the redirect URL.
    """

    def __init__(self, ttl_s=OAUTH_STATE_TTL_S, max_items=1000, clock=time.monotonic):
        self.ttl_s, self.max_items, self.clock = ttl_s, max_items, clock
        self._items = {}
        self._lock = threading.Lock()

    def put(self, state, verifier):
        with self._lock:
            self._prune()
            if len(self._items) >= self.max_items:
                self._items.pop(next(iter(self._items)))  # drop the oldest
            self._items[state] = (verifier, self.clock())

    def pop(self, state):
        """The verifier for `state` (removing it), or None if unknown or expired."""
        with self._lock:
            self._prune()
            item = self._items.pop(state, None)
            return item[0] if item else None

    def _prune(self):
        now = self.clock()
        for state in [s for s, (_, t) in self._items.items() if now - t > self.ttl_s]:
            del self._items[state]


_PENDING_OAUTH = PendingOAuthStore()


def strip_query(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", "", ""))


class AuthService:
    def __init__(self, client=None, url=None, pending=None):
        self.url = url or st.secrets["supabase"]["url"]
        self.client = client or create_client(self.url, st.secrets["supabase"]["key"])
        self.pending = pending or _PENDING_OAUTH

    def login_email(self, email, password):
        return self.client.auth.sign_in_with_password({"email": email, "password": password})

    def sign_up(self, email, password):
        return self.client.auth.sign_up({"email": email, "password": password})

    def google_sign_in_url(self, redirect_base):
        """Supabase authorize URL for Google (PKCE). Google sends the browser back to
        redirect_base?oauth_state=...&code=..., which complete_google_sign_in() finishes."""
        state = secrets.token_urlsafe(24)
        verifier = generate_pkce_verifier()
        self.pending.put(state, verifier)
        redirect_to = f"{strip_query(redirect_base)}?{urlencode({'oauth_state': state})}"
        query = urlencode({
            "provider": "google",
            "redirect_to": redirect_to,
            "code_challenge": generate_pkce_challenge(verifier),
            "code_challenge_method": "s256",
        })
        return f"{self.url.rstrip('/')}/auth/v1/authorize?{query}"

    def complete_google_sign_in(self, code, state):
        """Exchange the code from the redirect for a session; returns the Supabase user."""
        verifier = self.pending.pop(state)
        if verifier is None:
            raise OAuthStateError("This Google sign-in link has expired or was already used. Please try again.")
        response = self.client.auth.exchange_code_for_session({"auth_code": code, "code_verifier": verifier})
        return response.user

    def logout(self):
        self.client.auth.sign_out()
        st.session_state.user = None

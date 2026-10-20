"""Supabase login checks for the API, and a database client that acts as the logged-in user."""
import logging
import time

import jwt
from postgrest import SyncPostgrestClient
from postgrest.constants import DEFAULT_POSTGREST_CLIENT_HEADERS

AUDIENCE = "authenticated"   # Supabase access tokens of signed-in users
ASYMMETRIC_ALGORITHMS = ["ES256", "RS256"]

log = logging.getLogger(__name__)


class AuthError(Exception):
    """The token is missing, malformed, expired or not signed by this Supabase project (-> 401)."""


class AuthUnavailable(Exception):
    """The token couldn't be checked because Supabase's key or user endpoint is unreachable (-> 503)."""


class CurrentUser:
    def __init__(self, id, email, token):
        self.id, self.email, self.token = id, email, token


def describe(error):
    """'Type: message' for the error and each underlying cause, for the log (never contains the token)."""
    parts = []
    while error is not None and len(parts) < 5:
        parts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__
    return " <- ".join(parts)


class SupabaseJWTVerifier:
    """Checks a Supabase access token: signature, expiry, audience and issuer.

    Projects with asymmetric signing keys (ES256/RS256) are checked locally against the
    project's JWKS, which PyJWKClient fetches once and caches. Legacy HS256 tokens are checked
    locally with the project's JWT secret if it's configured; otherwise Supabase is asked
    (GET /auth/v1/user), which costs one network call per request.

    Claims that need no key (issuer, expiry, audience, subject) are checked first, so a token
    that's expired or from another Supabase project gets a 401 even while keys can't be fetched.

    verify() may block on the network, so the API calls it in a worker thread.
    """

    def __init__(self, supabase_url, supabase_key=None, jwt_secret=None, jwk_client=None, remote=None,
                 leeway_s=30):
        self.issuer = f"{supabase_url.rstrip('/')}/auth/v1"
        self.jwks_url = f"{self.issuer}/.well-known/jwks.json"
        self.jwt_secret = jwt_secret
        self.leeway_s = leeway_s
        self.jwk_client = jwk_client or jwt.PyJWKClient(self.jwks_url, cache_keys=True, lifespan=3600, timeout=10)
        self._remote = remote
        self._supabase_key = supabase_key

    def verify(self, token):
        try:
            alg = jwt.get_unverified_header(token).get("alg")
            unverified = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError as e:
            raise self._reject("Malformed token", e)
        self._check_claims(unverified)
        if alg == "HS256":
            if self.jwt_secret:
                return self._decode(token, self.jwt_secret, "HS256")
            return self._ask_supabase(token)
        if alg not in ASYMMETRIC_ALGORITHMS:
            raise self._reject(f"Unsupported token algorithm {alg!r}")
        try:
            key = self.jwk_client.get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientConnectionError as e:
            log.error("Could not fetch Supabase signing keys from %s: %s", self.jwks_url, describe(e))
            raise AuthUnavailable("Could not fetch Supabase signing keys") from e
        except jwt.PyJWTError as e:
            raise self._reject("Token signed with an unknown key", e)
        return self._decode(token, key, alg)

    def check_keys(self):
        """Fetch the signing keys once (at startup) and log the outcome; never raises."""
        try:
            keys = self.jwk_client.get_signing_keys()
        except Exception as e:
            log.error("Could not fetch Supabase signing keys from %s: %s. Every ES256/RS256 login will "
                      "get 503 until this is fixed; check SUPABASE_URL.", self.jwks_url, describe(e))
            return False
        log.info("Supabase signing keys loaded from %s: %s", self.jwks_url,
                 ", ".join(f"{k.key_id} ({k.algorithm_name})" for k in keys))
        return True

    def _check_claims(self, claims):
        """Checks that need no key. A token failing them can't be valid, whatever the signature."""
        iss = claims.get("iss")
        if iss != self.issuer:
            # The common setup mistake: the web app and the API point at different Supabase projects
            raise self._reject(f"Token was issued by {iss!r}, but this API accepts tokens from "
                               f"{self.issuer!r}. Check SUPABASE_URL on the API.")
        exp = claims.get("exp")
        if not isinstance(exp, (int, float)):
            raise self._reject("Invalid token")
        if exp < time.time() - self.leeway_s:
            raise self._reject("Token expired")
        if claims.get("aud") != AUDIENCE and AUDIENCE not in (claims.get("aud") or []):
            raise self._reject("Invalid token")
        if not claims.get("sub"):
            raise self._reject("Invalid token")

    def _reject(self, message, error=None):
        log.warning("Rejected token: %s%s", message, f" ({describe(error)})" if error else "")
        return AuthError(message)

    def _decode(self, token, key, alg):
        try:
            claims = jwt.decode(token, key, algorithms=[alg], audience=AUDIENCE, issuer=self.issuer,
                                leeway=self.leeway_s, options={"require": ["exp", "sub", "aud", "iss"]})
        except jwt.ExpiredSignatureError as e:
            raise self._reject("Token expired", e)
        except jwt.PyJWTError as e:
            raise self._reject("Invalid token", e)
        return CurrentUser(claims["sub"], claims.get("email"), token)

    def _ask_supabase(self, token):
        from supabase_auth.errors import AuthApiError, AuthRetryableError
        if self._remote is None:
            from supabase_auth import SyncGoTrueClient
            self._remote = SyncGoTrueClient(url=self.issuer, headers={"apikey": self._supabase_key or ""},
                                            auto_refresh_token=False, persist_session=False)
        try:
            response = self._remote.get_user(token)
        except AuthRetryableError as e:
            log.error("Supabase auth is unreachable at %s/user: %s", self.issuer, describe(e))
            raise AuthUnavailable("Supabase auth is unreachable") from e
        except AuthApiError as e:
            raise self._reject("Invalid token", e)
        if response is None or response.user is None:
            raise self._reject("Invalid token")
        return CurrentUser(response.user.id, response.user.email, token)


def user_db_factory(supabase_url, supabase_key):
    """make(token) -> PostgREST client that sends the user's token, so Supabase RLS sees that user.

    It has the same .table(...) API as the supabase client the services already take.
    """
    def make(token):
        return SyncPostgrestClient(f"{supabase_url}/rest/v1", headers={
            **DEFAULT_POSTGREST_CLIENT_HEADERS, "apikey": supabase_key, "Authorization": f"Bearer {token}"})
    return make

"""Supabase login checks for the API, and a database client that acts as the logged-in user."""
import jwt
from postgrest import SyncPostgrestClient
from postgrest.constants import DEFAULT_POSTGREST_CLIENT_HEADERS

AUDIENCE = "authenticated"   # Supabase access tokens of signed-in users
ASYMMETRIC_ALGORITHMS = ["ES256", "RS256"]


class AuthError(Exception):
    """The token is missing, malformed, expired or not signed by this Supabase project (-> 401)."""


class AuthUnavailable(Exception):
    """The token couldn't be checked because Supabase's key or user endpoint is unreachable (-> 503)."""


class CurrentUser:
    def __init__(self, id, email, token):
        self.id, self.email, self.token = id, email, token


class SupabaseJWTVerifier:
    """Checks a Supabase access token: signature, expiry, audience and issuer.

    Projects with asymmetric signing keys (ES256/RS256) are checked locally against the
    project's JWKS, which PyJWKClient fetches once and caches. Legacy HS256 tokens are checked
    locally with the project's JWT secret if it's configured; otherwise Supabase is asked
    (GET /auth/v1/user), which costs one network call per request.

    verify() may block on the network, so the API calls it in a worker thread.
    """

    def __init__(self, supabase_url, supabase_key=None, jwt_secret=None, jwk_client=None, remote=None,
                 leeway_s=30):
        self.issuer = f"{supabase_url.rstrip('/')}/auth/v1"
        self.jwt_secret = jwt_secret
        self.leeway_s = leeway_s
        self.jwk_client = jwk_client or jwt.PyJWKClient(
            f"{self.issuer}/.well-known/jwks.json", cache_keys=True, lifespan=3600, timeout=10)
        self._remote = remote
        self._supabase_key = supabase_key

    def verify(self, token):
        try:
            alg = jwt.get_unverified_header(token).get("alg")
        except jwt.PyJWTError as e:
            raise AuthError("Malformed token") from e
        if alg == "HS256":
            if self.jwt_secret:
                return self._decode(token, self.jwt_secret, "HS256")
            return self._ask_supabase(token)
        if alg not in ASYMMETRIC_ALGORITHMS:
            raise AuthError(f"Unsupported token algorithm {alg!r}")
        try:
            key = self.jwk_client.get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientConnectionError as e:
            raise AuthUnavailable("Could not fetch Supabase signing keys") from e
        except jwt.PyJWTError as e:
            raise AuthError("Token signed with an unknown key") from e
        return self._decode(token, key, alg)

    def _decode(self, token, key, alg):
        try:
            claims = jwt.decode(token, key, algorithms=[alg], audience=AUDIENCE, issuer=self.issuer,
                                leeway=self.leeway_s, options={"require": ["exp", "sub", "aud", "iss"]})
        except jwt.ExpiredSignatureError as e:
            raise AuthError("Token expired") from e
        except jwt.PyJWTError as e:
            raise AuthError("Invalid token") from e
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
            raise AuthUnavailable("Supabase auth is unreachable") from e
        except AuthApiError as e:
            raise AuthError("Invalid token") from e
        if response is None or response.user is None:
            raise AuthError("Invalid token")
        return CurrentUser(response.user.id, response.user.email, token)


def user_db_factory(supabase_url, supabase_key):
    """make(token) -> PostgREST client that sends the user's token, so Supabase RLS sees that user.

    It has the same .table(...) API as the supabase client the services already take.
    """
    def make(token):
        return SyncPostgrestClient(f"{supabase_url}/rest/v1", headers={
            **DEFAULT_POSTGREST_CLIENT_HEADERS, "apikey": supabase_key, "Authorization": f"Bearer {token}"})
    return make

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Final, cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

CLIENT_ID: Final[str] = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER: Final[str] = "https://auth.openai.com"
AUTHORIZE_URL: Final[str] = f"{ISSUER}/oauth/authorize"
TOKEN_URL: Final[str] = f"{ISSUER}/oauth/token"
REDIRECT_PATH: Final[str] = "/auth/callback"
REDIRECT_PORT: Final[int] = 1455
SCOPE: Final[str] = "openid profile email offline_access"
ORIGINATOR: Final[str] = "opencode"

DEFAULT_AUTH_FILE: Final[Path] = Path("data/.auth/openai-oauth.json")

logger = logging.getLogger("llm.oauth_openai")

ALLOWED_MODELS: Final[set[str]] = {
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
}

MODEL_MAP: Final[dict[str, str]] = {
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-none": "gpt-5.2",
    "gpt-5.2-low": "gpt-5.2",
    "gpt-5.2-medium": "gpt-5.2",
    "gpt-5.2-high": "gpt-5.2",
    "gpt-5.2-xhigh": "gpt-5.2",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-5.2-codex-low": "gpt-5.2-codex",
    "gpt-5.2-codex-medium": "gpt-5.2-codex",
    "gpt-5.2-codex-high": "gpt-5.2-codex",
    "gpt-5.2-codex-xhigh": "gpt-5.2-codex",
    "gpt-5.3-codex": "gpt-5.3-codex",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
}


@dataclass(frozen=True, slots=True)
class AuthTokens:
    type: str
    access: str
    refresh: str
    expires: int
    account_id: str | None = None
    id_token: str | None = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _base64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def generate_pkce() -> tuple[str, str]:
    verifier = _base64url(secrets.token_bytes(32))
    challenge = _base64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def generate_state() -> str:
    return _base64url(secrets.token_bytes(16))


def build_authorize_url(redirect_uri: str, challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "state": state,
        "codex_cli_simplified_flow": "true",
        "originator": ORIGINATOR,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _json_loads_dict(raw: bytes) -> dict[str, object]:
    try:
        parsed_obj = cast(object, json.loads(raw.decode("utf-8")))
    except Exception as e:
        raise RuntimeError("Invalid JSON from token endpoint") from e
    if not isinstance(parsed_obj, dict):
        raise RuntimeError("Unexpected token endpoint response")
    out: dict[str, object] = {}
    for k, v in cast(dict[object, object], parsed_obj).items():
        if isinstance(k, str):
            out[k] = v
    return out


def _coerce_str_key_dict(obj: object) -> dict[str, object] | None:
    if not isinstance(obj, dict):
        return None
    out: dict[str, object] = {}
    for k, v in cast(dict[object, object], obj).items():
        if isinstance(k, str):
            out[k] = v
    return out


def _token_request(form: dict[str, str]) -> dict[str, object]:
    payload = urlencode(form).encode("utf-8")
    req = Request(TOKEN_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with cast(HTTPResponse, urlopen(req, timeout=20)) as resp:
            status = resp.getcode()
            body = resp.read()
    except HTTPError as e:
        body = e.read()
        msg = body.decode("utf-8", errors="replace")
        raise RuntimeError(f"Token request failed: HTTP {e.code} {msg}") from e
    except URLError as e:
        raise RuntimeError(f"Token request failed: {e.reason}") from e

    if status != 200:
        text = body.decode("utf-8", errors="replace")
        raise RuntimeError(f"Token request failed: HTTP {status} {text}")

    return _json_loads_dict(body)


def _get_str(data: dict[str, object], key: str) -> str | None:
    val = data.get(key)
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return str(val)


def _get_int(data: dict[str, object], key: str, default: int) -> int:
    val = data.get(key)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(val)
        except Exception:
            return default
    return default


def exchange_authorization_code(
    code: str, verifier: str, redirect_uri: str
) -> AuthTokens:
    data = _token_request(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        }
    )

    access = _get_str(data, "access_token")
    refresh = _get_str(data, "refresh_token")
    if not access or not refresh:
        raise RuntimeError("Token exchange failed: missing access_token/refresh_token")

    expires_in = _get_int(data, "expires_in", default=3600)
    id_token = _get_str(data, "id_token")
    return AuthTokens(
        type="oauth",
        access=access,
        refresh=refresh,
        expires=_now_ms() + max(0, expires_in) * 1000,
        id_token=id_token,
    )


def refresh_access_token(refresh_token: str) -> AuthTokens:
    data = _token_request(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        }
    )
    access = _get_str(data, "access_token")
    if not access:
        raise RuntimeError("Token refresh failed: missing access_token")

    new_refresh = _get_str(data, "refresh_token") or refresh_token
    expires_in = _get_int(data, "expires_in", default=3600)
    id_token = _get_str(data, "id_token")
    return AuthTokens(
        type="oauth",
        access=access,
        refresh=new_refresh,
        expires=_now_ms() + max(0, expires_in) * 1000,
        id_token=id_token,
    )


def decode_jwt_claims(token: str | None) -> dict[str, object] | None:
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None

    pad = "=" * ((4 - len(parts[1]) % 4) % 4)
    try:
        payload = base64.urlsafe_b64decode(parts[1] + pad)
        parsed_obj = cast(object, json.loads(payload.decode("utf-8")))
    except Exception:
        return None
    if not isinstance(parsed_obj, dict):
        return None
    out: dict[str, object] = {}
    for k, v in cast(dict[object, object], parsed_obj).items():
        if isinstance(k, str):
            out[k] = v
    return out


def extract_account_id(tokens: AuthTokens) -> str | None:
    claims_id = decode_jwt_claims(tokens.id_token)
    if claims_id:
        account_id = _extract_account_id_from_claims(claims_id)
        if account_id:
            return account_id

    claims_access = decode_jwt_claims(tokens.access)
    if claims_access:
        account_id = _extract_account_id_from_claims(claims_access)
        if account_id:
            return account_id

    return None


def _extract_account_id_from_claims(claims: dict[str, object]) -> str | None:
    direct = claims.get("chatgpt_account_id")
    if direct is not None:
        return str(direct)

    auth_claim = _coerce_str_key_dict(claims.get("https://api.openai.com/auth"))
    if auth_claim:
        nested = auth_claim.get("chatgpt_account_id")
        if nested is not None:
            return str(nested)

    orgs = claims.get("organizations")
    if isinstance(orgs, list):
        org_list = cast(list[object], orgs)
        if not org_list:
            return None
        org0 = _coerce_str_key_dict(org_list[0])
        if org0:
            org_id = org0.get("id")
            if org_id is not None:
                return str(org_id)

    return None


def save_auth(auth: AuthTokens, auth_file: Path = DEFAULT_AUTH_FILE) -> None:
    account_id = auth.account_id or extract_account_id(auth)
    auth_to_write = dataclasses.replace(auth, account_id=account_id)

    auth_file.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "openai": {
            "type": auth_to_write.type,
            "access": auth_to_write.access,
            "refresh": auth_to_write.refresh,
            "expires": auth_to_write.expires,
            "account_id": auth_to_write.account_id,
            "id_token": auth_to_write.id_token,
        }
    }
    _ = auth_file.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(auth_file, 0o600)


def load_saved_auth(auth_file: Path = DEFAULT_AUTH_FILE) -> AuthTokens | None:
    if not auth_file.exists():
        return None
    try:
        parsed_obj = cast(object, json.loads(auth_file.read_text(encoding="utf-8")))
    except Exception:
        return None

    root = _coerce_str_key_dict(parsed_obj)
    if root is None:
        return None
    openai_obj = _coerce_str_key_dict(root.get("openai"))
    if openai_obj is None:
        return None

    t = openai_obj.get("type")
    if t != "oauth":
        return None

    refresh_val = openai_obj.get("refresh")
    if not refresh_val:
        return None

    access = openai_obj.get("access")
    refresh = str(refresh_val)
    expires_raw = openai_obj.get("expires")
    expires: int
    if isinstance(expires_raw, int):
        expires = expires_raw
    elif isinstance(expires_raw, float):
        expires = int(expires_raw)
    elif isinstance(expires_raw, str):
        try:
            expires = int(expires_raw)
        except Exception:
            expires = 0
    else:
        expires = 0

    account_id_raw = openai_obj.get("account_id")
    account_id = str(account_id_raw) if account_id_raw is not None else None
    id_token_raw = openai_obj.get("id_token")
    id_token = str(id_token_raw) if id_token_raw is not None else None

    return AuthTokens(
        type="oauth",
        access=str(access) if access is not None else "",
        refresh=refresh,
        expires=expires,
        account_id=account_id,
        id_token=id_token,
    )


class _CallbackHandler(BaseHTTPRequestHandler):
    expected_state: str = ""
    result_code: str | None = None
    result_error: str | None = None
    done_event: threading.Event = threading.Event()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != REDIRECT_PATH:
            self.send_response(404)
            self.end_headers()
            _ = self.wfile.write(b"Not found")
            return

        params = parse_qs(parsed.query)
        state = params.get("state", [""])[0]
        code = params.get("code", [""])[0]
        error = params.get("error", [""])[0]

        if error:
            self.__class__.result_error = f"OAuth error: {error}"
            self.send_response(400)
            self.end_headers()
            _ = self.wfile.write(b"Authorization failed")
            self.__class__.done_event.set()
            return

        if state != self.__class__.expected_state:
            self.__class__.result_error = "State mismatch"
            self.send_response(400)
            self.end_headers()
            _ = self.wfile.write(b"State mismatch")
            self.__class__.done_event.set()
            return

        if not code:
            self.__class__.result_error = "Missing authorization code"
            self.send_response(400)
            self.end_headers()
            _ = self.wfile.write(b"Missing authorization code")
            self.__class__.done_event.set()
            return

        self.__class__.result_code = code
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        _ = self.wfile.write(
            b"<html><body><h1>Authorization successful</h1><p>You can close this window.</p></body></html>"
        )
        self.__class__.done_event.set()


def login_with_browser_oauth(
    *, open_browser: bool = True, auth_file: Path = DEFAULT_AUTH_FILE
) -> AuthTokens:
    verifier, challenge = generate_pkce()
    state = generate_state()
    redirect_uri = f"http://localhost:{REDIRECT_PORT}{REDIRECT_PATH}"

    _CallbackHandler.expected_state = state
    _CallbackHandler.result_code = None
    _CallbackHandler.result_error = None
    _CallbackHandler.done_event = threading.Event()

    server = HTTPServer(("localhost", REDIRECT_PORT), _CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        authorize_url = build_authorize_url(redirect_uri, challenge, state)
        print(f"Open this URL to login:\n{authorize_url}")
        if open_browser:
            _ = webbrowser.open(authorize_url)

        finished = _CallbackHandler.done_event.wait(timeout=300)
        if not finished:
            raise TimeoutError("OAuth callback timeout")
        if _CallbackHandler.result_error:
            raise RuntimeError(_CallbackHandler.result_error)
        if not _CallbackHandler.result_code:
            raise RuntimeError("OAuth code missing")

        tokens = exchange_authorization_code(
            _CallbackHandler.result_code, verifier, redirect_uri
        )
        tokens = dataclasses.replace(tokens, account_id=extract_account_id(tokens))
        save_auth(tokens, auth_file=auth_file)
        return tokens
    finally:
        server.shutdown()
        server.server_close()


def get_reusable_auth(
    auth_file: Path = DEFAULT_AUTH_FILE,
    force_login: bool = False,
    open_browser: bool = True,
) -> AuthTokens:
    if not force_login:
        saved = load_saved_auth(auth_file=auth_file)
        if saved:
            if saved.access and saved.expires > _now_ms():
                logger.info("oauth_reuse_hit auth_file=%s", auth_file)
                return saved
            logger.info("oauth_reuse_expired_refresh_start auth_file=%s", auth_file)
            refreshed = refresh_access_token(saved.refresh)
            account_id = saved.account_id or extract_account_id(refreshed)
            refreshed = dataclasses.replace(refreshed, account_id=account_id)
            save_auth(refreshed, auth_file=auth_file)
            logger.info("oauth_reuse_refresh_success auth_file=%s", auth_file)
            return refreshed
        logger.warning("oauth_reuse_missing_saved_auth auth_file=%s", auth_file)

    if not open_browser:
        logger.error("oauth_retry_unavailable_no_browser auth_file=%s", auth_file)
        raise RuntimeError("No saved OAuth session. Login is required.")
    logger.info(
        "oauth_retry_login_start auth_file=%s force_login=%s", auth_file, force_login
    )
    return login_with_browser_oauth(open_browser=True, auth_file=auth_file)


def normalize_model(model: str) -> str:
    mapped = MODEL_MAP.get(model, model)
    if mapped not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported OpenAI OAuth model: {model}")
    return mapped

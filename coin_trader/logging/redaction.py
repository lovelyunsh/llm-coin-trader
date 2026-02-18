import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from structlog.types import EventDict, WrappedLogger
else:
    EventDict = dict[str, Any]
    WrappedLogger = Any

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "secret",
    "api_secret",
    "apisecret",
    "password",
    "passwd",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "auth",
    "signature",
    "private_key",
    "privatekey",
    "master_key",
    "masterkey",
    "jwt",
    "bearer",
}


SENSITIVE_PATTERNS = [
    re.compile(
        r"(api[_-]?key|api[_-]?secret|secret|password|token|authorization|signature|private[_-]?key|master[_-]?key|jwt|bearer)[\s:=\"']+([^\s,}\"']+)",
        re.IGNORECASE,
    ),
]


REDACTED = "***REDACTED***"


def _normalize_key(key: str) -> str:
    return re.sub(r"[\s_-]+", "", key).lower()


_SENSITIVE_KEYS_NORMALIZED = {_normalize_key(k) for k in SENSITIVE_KEYS}


def _is_sensitive_dict_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if normalized in _SENSITIVE_KEYS_NORMALIZED:
        return True
    # Also catch common prefixes/suffixes (e.g. x-api-key, client_secret)
    return any(normalized.endswith(sk) for sk in _SENSITIVE_KEYS_NORMALIZED)


def redact_string(text: str) -> str:
    if not text:
        return text

    # Handle multi-part authorization values (e.g. "Authorization: Bearer <token>")
    redacted = re.sub(
        r"(\bauthorization\b[\s:=\"']+)([^,}\"']+(?:\s+[^,}\"']+)*)",
        lambda m: f"{m.group(1)}{REDACTED}",
        text,
        flags=re.IGNORECASE,
    )

    def _replace_match(match: re.Match[str]) -> str:
        full = match.group(0)
        secret = match.group(match.re.groups)
        if not secret:
            return full
        return full[: -len(secret)] + REDACTED

    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(_replace_match, redacted)
    return redacted


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for k, v in data.items():
        key_str = k if isinstance(k, str) else str(k)
        if _is_sensitive_dict_key(key_str):
            redacted[k] = REDACTED
        else:
            redacted[k] = redact_value(v)
    return redacted


def redact_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        return redact_string(value)

    if isinstance(value, bytes):
        try:
            return redact_string(value.decode("utf-8", errors="replace"))
        except Exception:
            return REDACTED

    if isinstance(value, dict):
        # Prefer a typed signature when possible, but keep runtime flexibility.
        return redact_dict({str(k): v for k, v in value.items()})

    if isinstance(value, list):
        return [redact_value(v) for v in value]

    if isinstance(value, tuple):
        return tuple(redact_value(v) for v in value)

    if isinstance(value, set):
        return [redact_value(v) for v in value]

    # Pydantic models or similar objects
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return redact_dict({str(k): v for k, v in dumped.items()})
        except Exception:
            return redact_string(str(value))

    return value


def redact_sensitive_data(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """structlog processor that redacts sensitive data from all event dict values"""

    return {k: redact_value(v) for k, v in event_dict.items()}

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from coin_trader.llm.codex_client import parse_sse_text


class _AuthTokensProto(Protocol):
    type: str
    access: str
    refresh: str
    expires: int
    account_id: str | None
    id_token: str | None


class _AuthTokensCtor(Protocol):
    def __call__(
        self,
        *,
        type: str,
        access: str,
        refresh: str,
        expires: int,
        account_id: str | None = ...,
        id_token: str | None = ...,
    ) -> _AuthTokensProto: ...


class _LoadSavedAuth(Protocol):
    def __call__(self, auth_file: Path) -> _AuthTokensProto | None: ...


class _SaveAuth(Protocol):
    def __call__(self, auth: _AuthTokensProto, auth_file: Path) -> None: ...


class _OAuthOpenAIModule(Protocol):
    AuthTokens: _AuthTokensCtor

    generate_pkce: Callable[[], tuple[str, str]]
    generate_state: Callable[[], str]
    normalize_model: Callable[[str], str]
    load_saved_auth: _LoadSavedAuth
    save_auth: _SaveAuth


oauth_openai = cast(
    _OAuthOpenAIModule,
    cast(object, importlib.import_module("coin_trader.llm.oauth_openai")),
)


def test_parse_sse_text_delta_events() -> None:
    raw = "\n".join(
        [
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}",
        ]
    )
    assert parse_sse_text(raw) == "Hello"


def test_parse_sse_text_final_output_event() -> None:
    raw = (
        "data: "
        + "{\"type\":\"response.completed\",\"output\":[{\"content\":[{\"type\":\"output_text\",\"text\":\"Final output\"}]}]}"
    )
    assert parse_sse_text(raw) == "Final output"


def test_parse_sse_text_empty_or_invalid_data() -> None:
    assert parse_sse_text("") == ""
    assert parse_sse_text("data: not-json") == ""
    assert parse_sse_text("event: message\nfoo") == ""


def test_parse_sse_text_done_sentinel_skipped() -> None:
    assert parse_sse_text("data: [DONE]") == ""


def test_generate_pkce_returns_two_different_strings() -> None:
    verifier, challenge = oauth_openai.generate_pkce()
    assert isinstance(verifier, str) and verifier
    assert isinstance(challenge, str) and challenge
    assert verifier != challenge


def test_generate_state_returns_non_empty_string() -> None:
    state = oauth_openai.generate_state()
    assert isinstance(state, str)
    assert state


def test_normalize_model_maps_known_names() -> None:
    assert oauth_openai.normalize_model("gpt-5.2-high") == "gpt-5.2"
    assert oauth_openai.normalize_model("gpt-5.2-codex-xhigh") == "gpt-5.2-codex"


def test_normalize_model_raises_for_unknown() -> None:
    try:
        _ = oauth_openai.normalize_model("not-a-real-model")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_load_saved_auth_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert oauth_openai.load_saved_auth(tmp_path / "does_not_exist.json") is None


def test_save_auth_load_saved_auth_roundtrip(tmp_path: Path) -> None:
    auth_file = tmp_path / "openai-oauth.json"
    tokens = oauth_openai.AuthTokens(
        type="oauth",
        access="access_token",
        refresh="refresh_token",
        expires=123456,
        account_id="acct_123",
        id_token=None,
    )
    oauth_openai.save_auth(tokens, auth_file=auth_file)

    loaded = oauth_openai.load_saved_auth(auth_file=auth_file)
    assert loaded is not None
    assert loaded.type == "oauth"
    assert loaded.access == "access_token"
    assert loaded.refresh == "refresh_token"
    assert loaded.expires == 123456
    assert loaded.account_id == "acct_123"

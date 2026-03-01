"""Tests for module-level utility functions in coin_trader/exchange/base.py"""


from coin_trader.exchange.base import _API_METRICS, _as_str_object_dict, _to_int, get_api_metrics

# ---------------------------------------------------------------------------
# _as_str_object_dict
# ---------------------------------------------------------------------------


def test_as_str_object_dict_valid_dict_returned() -> None:
    d = {"key": "value", "num": 42}
    result = _as_str_object_dict(d)
    assert result is d


def test_as_str_object_dict_empty_dict_returned() -> None:
    result = _as_str_object_dict({})
    assert result == {}


def test_as_str_object_dict_list_returns_none() -> None:
    assert _as_str_object_dict([1, 2, 3]) is None


def test_as_str_object_dict_str_returns_none() -> None:
    assert _as_str_object_dict("hello") is None


def test_as_str_object_dict_int_returns_none() -> None:
    assert _as_str_object_dict(99) is None


def test_as_str_object_dict_int_key_returns_none() -> None:
    # dict with a non-str key must return None
    assert _as_str_object_dict({1: "value"}) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _to_int
# ---------------------------------------------------------------------------


def test_to_int_int_passthrough() -> None:
    assert _to_int(7) == 7


def test_to_int_float_truncated() -> None:
    assert _to_int(3.9) == 3


def test_to_int_str_numeric() -> None:
    assert _to_int("42") == 42


def test_to_int_bool_true_is_one() -> None:
    # bool is handled before int branch
    assert _to_int(True) == 1


def test_to_int_bool_false_is_zero() -> None:
    assert _to_int(False) == 0


def test_to_int_invalid_str_uses_default() -> None:
    assert _to_int("abc") == 0


def test_to_int_invalid_str_custom_default() -> None:
    assert _to_int("xyz", default=99) == 99


def test_to_int_none_uses_default() -> None:
    # None falls through to the final 'return default'
    assert _to_int(None) == 0  # type: ignore[arg-type]


def test_to_int_none_custom_default() -> None:
    assert _to_int(None, default=-1) == -1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_api_metrics
# ---------------------------------------------------------------------------


def test_get_api_metrics_returns_correct_structure() -> None:
    metrics = get_api_metrics()
    assert "total_requests" in metrics
    assert "total_429" in metrics
    assert "last_429_ts" in metrics
    assert "per_endpoint" in metrics
    assert isinstance(metrics["total_requests"], int)
    assert isinstance(metrics["total_429"], int)
    assert isinstance(metrics["per_endpoint"], dict)


def test_get_api_metrics_reflects_modified_state() -> None:
    original_requests = _to_int(_API_METRICS.get("total_requests", 0), 0)
    # Directly mutate the module-level dict to simulate a request being counted
    _API_METRICS["total_requests"] = original_requests + 5
    try:
        metrics = get_api_metrics()
        assert metrics["total_requests"] == original_requests + 5
    finally:
        # Restore original value to avoid polluting other tests
        _API_METRICS["total_requests"] = original_requests

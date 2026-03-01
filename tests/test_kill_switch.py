"""Tests for KillSwitch in coin_trader/safety/kill_switch.py"""

import pytest

from coin_trader.safety.kill_switch import KillSwitch

# ---------------------------------------------------------------------------
# is_active
# ---------------------------------------------------------------------------


def test_is_active_default_false(tmp_path: pytest.TempPathFactory) -> None:
    ks = KillSwitch(kill_switch_file=tmp_path / "RUN" / "kill_switch")
    assert ks.is_active() is False


def test_is_active_true_after_activate(tmp_path: pytest.TempPathFactory) -> None:
    ks = KillSwitch(kill_switch_file=tmp_path / "RUN" / "kill_switch")
    ks.activate("testing")
    assert ks.is_active() is True


def test_is_active_true_when_file_exists(tmp_path: pytest.TempPathFactory) -> None:
    kill_file = tmp_path / "RUN" / "kill_switch"
    kill_file.parent.mkdir(parents=True, exist_ok=True)
    kill_file.write_text("external trigger", encoding="utf-8")

    # New instance — _active flag is False, but file exists
    ks = KillSwitch(kill_switch_file=kill_file)
    assert ks.is_active() is True


def test_is_active_false_after_deactivate(tmp_path: pytest.TempPathFactory) -> None:
    ks = KillSwitch(kill_switch_file=tmp_path / "RUN" / "kill_switch")
    ks.activate("testing")
    ks.deactivate()
    assert ks.is_active() is False


# ---------------------------------------------------------------------------
# activate
# ---------------------------------------------------------------------------


def test_activate_creates_file_with_reason(tmp_path: pytest.TempPathFactory) -> None:
    kill_file = tmp_path / "RUN" / "kill_switch"
    ks = KillSwitch(kill_switch_file=kill_file)
    ks.activate("emergency stop")
    assert kill_file.exists()
    assert kill_file.read_text(encoding="utf-8") == "emergency stop"


def test_activate_sets_internal_flag(tmp_path: pytest.TempPathFactory) -> None:
    ks = KillSwitch(kill_switch_file=tmp_path / "RUN" / "kill_switch")
    ks.activate("flag check")
    assert ks._active is True


def test_activate_creates_parent_dirs(tmp_path: pytest.TempPathFactory) -> None:
    kill_file = tmp_path / "deep" / "nested" / "dir" / "kill_switch"
    ks = KillSwitch(kill_switch_file=kill_file)
    ks.activate("nested dirs")
    assert kill_file.exists()


# ---------------------------------------------------------------------------
# deactivate
# ---------------------------------------------------------------------------


def test_deactivate_removes_file(tmp_path: pytest.TempPathFactory) -> None:
    kill_file = tmp_path / "RUN" / "kill_switch"
    ks = KillSwitch(kill_switch_file=kill_file)
    ks.activate("to be removed")
    assert kill_file.exists()
    ks.deactivate()
    assert not kill_file.exists()


def test_deactivate_clears_internal_flag(tmp_path: pytest.TempPathFactory) -> None:
    ks = KillSwitch(kill_switch_file=tmp_path / "RUN" / "kill_switch")
    ks.activate("flag check")
    ks.deactivate()
    assert ks._active is False


def test_deactivate_no_error_if_file_missing(tmp_path: pytest.TempPathFactory) -> None:
    ks = KillSwitch(kill_switch_file=tmp_path / "RUN" / "kill_switch")
    # File was never created — deactivate must not raise
    ks.deactivate()
    assert ks._active is False


# ---------------------------------------------------------------------------
# get_reason
# ---------------------------------------------------------------------------


def test_get_reason_returns_none_when_no_file(tmp_path: pytest.TempPathFactory) -> None:
    ks = KillSwitch(kill_switch_file=tmp_path / "RUN" / "kill_switch")
    assert ks.get_reason() is None


def test_get_reason_returns_file_content(tmp_path: pytest.TempPathFactory) -> None:
    kill_file = tmp_path / "RUN" / "kill_switch"
    kill_file.parent.mkdir(parents=True, exist_ok=True)
    kill_file.write_text("  specific reason  ", encoding="utf-8")

    ks = KillSwitch(kill_switch_file=kill_file)
    assert ks.get_reason() == "specific reason"

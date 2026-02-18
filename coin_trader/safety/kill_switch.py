"""Kill switch - immediate halt of all trading activity"""

from __future__ import annotations

import signal
import threading
from pathlib import Path
from types import FrameType


class KillSwitch:
    _file: Path
    _active: bool
    _lock: threading.Lock

    def __init__(self, kill_switch_file: Path, *, install_signal_handlers: bool = False) -> None:
        self._file = kill_switch_file
        self._active = False
        self._lock = threading.Lock()
        if install_signal_handlers:
            self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        prev_sigint = signal.getsignal(signal.SIGINT)

        def _handler(signum: int, frame: FrameType | None) -> None:
            self.activate(f"Signal received: {signum}")
            prev = prev_sigterm if signum == signal.SIGTERM else prev_sigint
            if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
                raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def is_active(self) -> bool:
        with self._lock:
            return self._active or self._file.exists()

    def activate(self, reason: str = "Manual activation") -> None:
        with self._lock:
            self._active = True
            self._file.parent.mkdir(parents=True, exist_ok=True)
            _ = self._file.write_text(reason, encoding="utf-8")

    def deactivate(self) -> None:
        with self._lock:
            self._active = False
            if self._file.exists():
                self._file.unlink()

    def get_reason(self) -> str | None:
        if self._file.exists():
            return self._file.read_text(encoding="utf-8").strip()
        return None

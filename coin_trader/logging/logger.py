from __future__ import annotations

import atexit
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import structlog
except ModuleNotFoundError:  # pragma: no cover
    structlog = None

if TYPE_CHECKING:
    from structlog.types import EventDict, WrappedLogger
else:
    EventDict = dict[str, Any]
    WrappedLogger = Any

from coin_trader.logging.redaction import redact_sensitive_data

EVENT_FILE_MAP = {
    "decision": "decisions.jsonl",
    "symbol_decision": "symbol_decisions.jsonl",
    "order": "orders.jsonl",
    "fill": "fills.jsonl",
    "balance": "balances.jsonl",
    "safety": "safety.jsonl",
}

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"
_BG_RED = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"
_BG_BLUE = "\033[44m"

_LEVEL_STYLES: dict[str, str] = {
    "debug": _DIM,
    "info": _CYAN,
    "warning": f"{_BOLD}{_YELLOW}",
    "error": f"{_BOLD}{_RED}",
    "critical": f"{_BOLD}{_BG_RED}{_WHITE}",
}

_EVENT_ICONS: dict[str, str] = {
    "trader_started": f"{_BG_BLUE}{_WHITE}{_BOLD} START {_RESET}",
    "trader_stopped": f"{_BG_RED}{_WHITE}{_BOLD} STOP  {_RESET}",
    "paper_mode_active": f"{_BG_BLUE}{_WHITE} PAPER {_RESET}",
    "live_mode_active": f"{_BG_RED}{_WHITE}{_BOLD} LIVE  {_RESET}",
    "order_placed": f"{_BG_GREEN}{_WHITE}{_BOLD} ORDER {_RESET}",
    "order_cancelled": f"{_BG_YELLOW}{_WHITE}{_BOLD} CANCEL{_RESET}",
    "order_failed": f"{_BG_RED}{_WHITE}{_BOLD} FAIL  {_RESET}",
    "execution_blocked": f"{_BG_RED}{_WHITE}{_BOLD} BLOCK {_RESET}",
    "execution_rejected": f"{_YELLOW} REJECT{_RESET}",
    "circuit_breaker": f"{_BG_RED}{_WHITE}{_BOLD} BREAK {_RESET}",
    "price_anomaly": f"{_BG_YELLOW}{_WHITE}{_BOLD} ALERT {_RESET}",
    "api_failure_threshold": f"{_BG_RED}{_WHITE}{_BOLD} API!! {_RESET}",
}

_SKIP_CONSOLE_KEYS = frozenset(
    {"event", "level", "ts", "timestamp", "run_id", "mode", "exchange", "logger", "exc_info"}
)


def _format_kv(key: str, value: object) -> str:
    return f"{_DIM}{key}={_RESET}{value}"


def _pretty_console(event: str, level: str, kv: dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    level_style = _LEVEL_STYLES.get(level, "")

    icon = _EVENT_ICONS.get(event)
    if icon:
        header = f"{_DIM}{ts}{_RESET} {icon}"
    else:
        lvl_tag = level.upper()[:4].ljust(4)
        header = f"{_DIM}{ts}{_RESET} {level_style}{lvl_tag}{_RESET}"

    event_str = f"{_BOLD}{event}{_RESET}"

    detail_parts = [
        _format_kv(k, v) for k, v in kv.items() if k not in _SKIP_CONSOLE_KEYS and v is not None
    ]
    details = f"  {' '.join(detail_parts)}" if detail_parts else ""

    return f"{header} {event_str}{details}"


_current_mode: str = os.getenv("TRADING_MODE", "paper")


def set_trading_mode(mode: str) -> None:
    global _current_mode
    _current_mode = mode


def add_common_fields(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("run_id", os.getenv("RUN_ID", "unknown"))
    event_dict.setdefault("mode", _current_mode)
    event_dict.setdefault("exchange", os.getenv("EXCHANGE", "upbit"))
    return event_dict


class EventFileWriter:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}

    def write_event(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "decision":
            symbol = str(data.get("symbol", "unknown"))
            safe_name = symbol.replace("/", "_").lower()
            filename = f"decisions_{safe_name}.jsonl"
        else:
            filename = EVENT_FILE_MAP.get(event_type, "general.jsonl")
        filepath = self.log_dir / filename
        if filename not in self._handles:
            self._handles[filename] = open(filepath, "a", encoding="utf-8")
        self._handles[filename].write(json.dumps(data, default=str) + "\n")
        self._handles[filename].flush()

        if event_type == "decision":
            self._compact_holds(filename)

    def _compact_holds(self, filename: str, hold_ttl_sec: float = 3600.0) -> None:
        filepath = self.log_dir / filename

        if filename in self._handles:
            self._handles[filename].close()
            del self._handles[filename]

        try:
            lines = filepath.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            return
        if len(lines) < 3:
            return

        entries: list[dict[str, Any]] = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        from datetime import datetime as _dt

        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = []

        for entry in entries:
            action = str(entry.get("final_action", "")).upper()
            if action == "HOLD":
                current_group.append(entry)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([entry])
        if current_group:
            groups.append(current_group)

        kept: list[dict[str, Any]] = []
        for group in groups:
            if len(group) <= 1 or str(group[0].get("final_action", "")).upper() != "HOLD":
                kept.extend(group)
                continue
            try:
                t_first = _dt.fromisoformat(str(group[0].get("ts", "")).replace("Z", "+00:00"))
                t_last = _dt.fromisoformat(str(group[-1].get("ts", "")).replace("Z", "+00:00"))
                span = (t_last - t_first).total_seconds()
            except (ValueError, TypeError):
                kept.extend(group)
                continue

            if span <= hold_ttl_sec:
                kept.extend(group)
            else:
                kept.append(group[0])
                kept.append(group[-1])

        if len(kept) < len(entries):
            filepath.write_text(
                "\n".join(json.dumps(e, default=str) for e in kept) + "\n",
                encoding="utf-8",
            )

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles.clear()


_file_writer: EventFileWriter | None = None

_LEVEL_MAP = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_fallback_level = _LEVEL_MAP["INFO"]


class _FallbackLogger:
    def __init__(self, name: str | None = None, context: dict[str, Any] | None = None) -> None:
        self._name = name or "coin_trader"
        self._context = dict(context or {})

    def bind(self, **kwargs: Any) -> "_FallbackLogger":
        merged = dict(self._context)
        merged.update(kwargs)
        return _FallbackLogger(self._name, merged)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log("DEBUG", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log("ERROR", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._log("ERROR", event, **kwargs)

    def _log(self, level_name: str, event: str, **kwargs: Any) -> None:
        level = _LEVEL_MAP.get(level_name, 20)
        if level < _fallback_level:
            return

        payload: dict[str, Any] = dict(self._context)
        payload.update(kwargs)

        from coin_trader.logging.redaction import redact_dict

        payload = redact_dict(payload)

        line = _pretty_console(event, level_name.lower(), payload)
        print(line, file=sys.stderr)


def _pretty_structlog_renderer(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> str:
    event = str(event_dict.pop("event", ""))
    level = str(event_dict.pop("level", method_name))
    event_dict.pop("timestamp", None)
    return _pretty_console(event, level, event_dict)


def setup_logging(log_dir: Path | None = None, log_level: str = "INFO") -> None:
    global _file_writer
    global _fallback_level

    log_dir = log_dir or Path("logs")
    _file_writer = EventFileWriter(log_dir)
    atexit.register(close_logging)

    _fallback_level = _LEVEL_MAP.get(log_level.upper(), 20)

    if structlog is None:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            add_common_fields,
            redact_sensitive_data,
            _pretty_structlog_renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_fallback_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def close_logging() -> None:
    global _file_writer
    if _file_writer is None:
        return
    try:
        _file_writer.close()
    finally:
        _file_writer = None


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    if structlog is None:
        return _FallbackLogger(name)
    return structlog.get_logger(name)


def log_event(event_type: str, data: dict[str, Any]) -> None:
    if _file_writer is None:
        return

    from coin_trader.logging.redaction import redact_dict

    payload: dict[str, Any] = dict(data)
    payload.setdefault("event_type", event_type)
    payload.setdefault("run_id", os.getenv("RUN_ID", "unknown"))
    payload.setdefault("mode", _current_mode)
    payload.setdefault("exchange", os.getenv("EXCHANGE", "upbit"))
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    _file_writer.write_event(event_type, redact_dict(payload))

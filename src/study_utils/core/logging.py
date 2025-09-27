"""Logging helpers shared across study_utils subcommands."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

__all__ = [
    "JsonLogFormatter",
    "configure_logger",
]


class JsonLogFormatter(logging.Formatter):
    """Emit log records as structured JSON lines."""

    _RESERVED = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: _coerce_value(value)
            for key, value in record.__dict__.items()
            if key not in self._RESERVED
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        return json.dumps(payload, ensure_ascii=True)


def configure_logger(
    name: str,
    *,
    log_dir: Path,
    level: str = "INFO",
    verbose: bool = False,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    filename: str | None = None,
) -> tuple[logging.Logger, Path]:
    """Configure and return a namespaced logger with JSON file output."""

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    file_level = _coerce_level(level)
    if verbose:
        file_level = logging.DEBUG

    target_dir = _prepare_log_dir(log_dir)
    log_name = filename or f"{name.rsplit('.', 1)[-1]}.log"
    file_path = _prepare_log_file(target_dir, log_name)

    file_handler, file_path = _ensure_file_handler(
        logger=logger,
        path=file_path,
        filename=log_name,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    file_handler.setLevel(file_level)

    if verbose:
        _enable_console_handler(logger)
    else:
        _disable_console_handler(logger)

    return logger, file_path


def _coerce_level(level: str) -> int:
    name = level.upper()
    numeric = logging.getLevelName(name)
    if isinstance(numeric, int):
        return numeric
    return logging.INFO


def _ensure_file_handler(
    *,
    logger: logging.Logger,
    path: Path,
    filename: str,
    max_bytes: int,
    backup_count: int,
) -> tuple[RotatingFileHandler, Path]:
    managed: RotatingFileHandler | None = None
    for handler in logger.handlers:
        if getattr(handler, "_study_utils_file", False):
            managed = handler  # type: ignore[assignment]
            break
    if managed is None:
        try:
            managed = RotatingFileHandler(
                path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            active_path = path
        except PermissionError:
            fallback_dir = _prepare_log_dir(_fallback_log_dir())
            active_path = _prepare_log_file(fallback_dir, filename)
            managed = RotatingFileHandler(
                active_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        managed.setFormatter(JsonLogFormatter())
        managed._study_utils_file = True  # type: ignore[attr-defined]
        logger.addHandler(managed)
    else:
        managed.baseFilename = str(path)
        active_path = path
    return managed, Path(active_path)


def _enable_console_handler(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if getattr(handler, "_study_utils_console", False):
            handler.setLevel(logging.DEBUG)
            return
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    console._study_utils_console = True  # type: ignore[attr-defined]
    logger.addHandler(console)


def _disable_console_handler(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_study_utils_console", False):
            logger.removeHandler(handler)
            handler.close()


def _coerce_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _coerce_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_value(item) for item in value]
    return repr(value)


def _prepare_log_dir(log_dir: Path) -> Path:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            log_dir.chmod(0o700)
        except PermissionError:  # pragma: no cover - depends on filesystem
            pass
        return log_dir
    except PermissionError:
        fallback = Path(tempfile.gettempdir()) / "study-utils-logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _prepare_log_file(log_dir: Path, filename: str) -> Path:
    path = log_dir / filename
    try:
        path.touch(exist_ok=True)
    except PermissionError:  # pragma: no cover - depends on filesystem
        fallback = Path(tempfile.gettempdir()) / "study-utils-logs"
        fallback.mkdir(parents=True, exist_ok=True)
        path = fallback / filename
        path.touch(exist_ok=True)
    try:
        path.chmod(0o600)
    except PermissionError:  # pragma: no cover - depends on filesystem
        pass
    return path


def _fallback_log_dir() -> Path:
    return Path(tempfile.gettempdir()) / "study-utils-logs"

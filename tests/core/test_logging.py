from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from study_utils.core import logging as core_logging


def test_configure_logger_writes_json(tmp_path):
    log_dir = tmp_path / "logs"
    logger, log_path = core_logging.configure_logger(
        "study_utils.test",
        log_dir=log_dir,
        level="INFO",
        verbose=False,
        filename="test.log",
    )

    logger.info("hello world", extra={"event": "unit", "value": 3})
    logger.info("stack info", stack_info=True)

    class _Helper:
        def __repr__(self):  # noqa: D401
            return "helper"

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception(
            "with error",
            extra={
                "event": "unit",
                "value": {
                    "items": [Path(log_dir), 1],
                    "mapping": {"k": "v"},
                },
                "obj": _Helper(),
            },
        )
    for handler in logger.handlers:
        handler.flush()

    contents = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents
    first = json.loads(contents[0])
    assert first["message"] == "hello world"

    payload = json.loads(contents[-1])
    assert payload["exception"]
    assert payload["extra"]["obj"] == "helper"
    assert payload["extra"]["value"]["items"][0]

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_configure_logger_adds_console_handler(tmp_path):
    logger, _ = core_logging.configure_logger(
        "study_utils.test_verbose",
        log_dir=tmp_path / "logs",
        level="INFO",
        verbose=True,
        filename="verbose.log",
    )

    console_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_study_utils_console", False)
    ]
    assert console_handlers

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_configure_logger_fallback_directory(tmp_path, monkeypatch):
    target = tmp_path / "blocked"

    original_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):  # noqa: D401, ANN001
        if self == target:
            raise PermissionError("denied")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    logger, log_path = core_logging.configure_logger(
        "study_utils.test_blocked",
        log_dir=target,
        filename="blocked.log",
    )

    # Ensure the fallback location was used.
    assert log_path.parent != target
    assert log_path.exists()

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_configure_logger_fallback_file(tmp_path, monkeypatch):
    target = tmp_path / "no-touch"
    target.mkdir(parents=True)

    original_touch = Path.touch

    def fake_touch(self, *args, **kwargs):  # noqa: D401, ANN001
        if self.parent == target:
            raise PermissionError("no touch")
        return original_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fake_touch)

    logger, log_path = core_logging.configure_logger(
        "study_utils.test_touch",
        log_dir=target,
        filename="touch.log",
    )

    assert log_path.parent != target
    assert log_path.exists()

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_configure_logger_rotating_handler_fallback(tmp_path, monkeypatch):
    calls = {"count": 0}
    fallback_dir = tmp_path / "rotate-fallback"
    original_handler = core_logging.RotatingFileHandler

    def fake_handler(path, *args, **kwargs):  # noqa: D401, ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("denied")
        return original_handler(path, *args, **kwargs)

    monkeypatch.setattr(core_logging, "RotatingFileHandler", fake_handler)
    monkeypatch.setattr(core_logging, "_fallback_log_dir", lambda: fallback_dir)

    logger, log_path = core_logging.configure_logger(
        "study_utils.test_rotating_fallback",
        log_dir=tmp_path / "primary",
        filename="rotate.log",
    )

    assert log_path.parent == fallback_dir
    assert calls["count"] == 2

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_fallback_log_dir_uses_tempdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    path = core_logging._fallback_log_dir()

    assert path == tmp_path / "study-utils-logs"


def test_console_handler_toggle(tmp_path):
    log_dir = tmp_path / "logs"
    logger_name = "study_utils.test_toggle"

    logger, _ = core_logging.configure_logger(
        logger_name,
        log_dir=log_dir,
        verbose=True,
        filename="toggle.log",
    )

    console_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_study_utils_console", False)
    ]
    assert len(console_handlers) == 1

    # Calling configure_logger again with verbose=True should reuse the handler.
    core_logging.configure_logger(
        logger_name,
        log_dir=log_dir,
        verbose=True,
        filename="toggle.log",
    )
    console_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_study_utils_console", False)
    ]
    assert len(console_handlers) == 1

    # Disabling verbose should remove the console handler.
    core_logging.configure_logger(
        logger_name,
        log_dir=log_dir,
        verbose=False,
        filename="toggle.log",
    )
    console_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_study_utils_console", False)
    ]
    assert not console_handlers

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_coerce_level_defaults():
    assert core_logging._coerce_level("bogus") == logging.INFO

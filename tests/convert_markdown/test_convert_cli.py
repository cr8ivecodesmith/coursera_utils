from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from study_utils.convert_markdown import cli
from study_utils.convert_markdown import config as cfg
from study_utils.convert_markdown import converter
from study_utils.core import config_templates
from study_utils.core.workspace import WorkspaceError, WorkspaceLayout


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[
            tuple[str, tuple[object, ...], dict[str, object]]
        ] = []

    def debug(self, *args, **kwargs) -> None:
        self.messages.append(("debug", args, kwargs))

    def info(self, *args, **kwargs) -> None:  # pragma: no cover - defensive
        self.messages.append(("info", args, kwargs))

    def error(self, *args, **kwargs) -> None:  # pragma: no cover - defensive
        self.messages.append(("error", args, kwargs))


def _load_result() -> cfg.LoadResult:
    layout = WorkspaceLayout(
        home=Path("/tmp/ws"),
        directories={
            "converted": Path("/tmp/ws/converted"),
            "logs": Path("/tmp/ws/logs"),
        },
        created={
            "home": True,
            "converted": True,
            "logs": True,
        },
    )
    return cfg.LoadResult(
        config=cfg.ConvertMarkdownConfig(
            extensions=("pdf",),
            output_dir=Path("/tmp/ws/converted"),
            collision=cfg.CollisionPolicy.SKIP,
            log_level="DEBUG",
        ),
        layout=layout,
        config_path=None,
    )


def test_cli_invokes_executor_and_prints_summary(monkeypatch, capsys):
    captured: dict[str, object] = {}
    dummy_logger = DummyLogger()

    load_result = _load_result()

    def fake_load_config(**kwargs):
        captured.update(kwargs)
        return load_result

    monkeypatch.setattr(cli, "load_config", fake_load_config)

    fake_dependencies = object()
    monkeypatch.setattr(cli, "_build_dependencies", lambda: fake_dependencies)

    def fake_configure_logger(name, *, log_dir, level):
        captured["logger_call"] = (name, log_dir, level)
        return dummy_logger, Path("/tmp/ws/logs/convert.log")

    monkeypatch.setattr(cli, "configure_logger", fake_configure_logger)

    summary = cli.ExecutionSummary(
        requested=(Path("input.pdf"),),
        processed=(Path("/tmp/ws/input.pdf"),),
        outcomes=(
            converter.ConversionOutcome(
                source=Path("/tmp/ws/input.pdf"),
                status=converter.ConversionStatus.SUCCESS,
                output_path=Path("/tmp/ws/converted/input.md"),
            ),
            converter.ConversionOutcome(
                source=Path("/tmp/ws/bad.pdf"),
                status=converter.ConversionStatus.FAILED,
                reason="boom",
            ),
        ),
    )

    def fake_run_conversion(paths, *, config, dependencies, logger):
        captured["run_args"] = {
            "paths": paths,
            "config": config,
            "dependencies": dependencies,
            "logger": logger,
        }
        return summary

    monkeypatch.setattr(cli, "run_conversion", fake_run_conversion)

    code = cli.main(
        [
            "--config",
            "config.toml",
            "--workspace",
            "workspace",
            "--output-dir",
            "cli-out",
            "--extensions",
            "pdf",
            "--overwrite",
            "--log-level",
            "debug",
            "input.pdf",
        ]
    )

    assert code == 1

    overrides = captured["overrides"]
    assert isinstance(overrides, cfg.ConfigOverrides)
    assert overrides.output_dir == Path("cli-out")
    assert overrides.extensions == ["pdf"]
    assert overrides.collision is cfg.CollisionPolicy.OVERWRITE
    assert overrides.log_level == "debug"

    assert captured["config_path"] == Path("config.toml")
    assert captured["workspace_path"] == Path("workspace")

    assert captured["logger_call"] == (
        "study_utils.convert_markdown",
        Path("/tmp/ws/logs"),
        "DEBUG",
    )
    run_args = captured["run_args"]
    assert run_args["paths"] == [Path("input.pdf")]
    assert run_args["config"] is load_result.config
    assert run_args["dependencies"] is fake_dependencies
    assert run_args["logger"] is dummy_logger

    captured_io = capsys.readouterr()
    assert "converted: 1" in captured_io.out
    assert "failed:    1" in captured_io.out
    assert "log file:   /tmp/ws/logs/convert.log" in captured_io.out
    assert captured_io.err == ""


def test_cli_conflicting_collision_flags_error(monkeypatch):
    parser_error = {"message": None}

    class FakeParser:
        def parse_args(self, _):
            return SimpleNamespace(
                paths=[Path("file.pdf")],
                config=None,
                workspace=None,
                output_dir=None,
                extensions=None,
                overwrite=True,
                version_output=True,
                log_level=None,
            )

        def error(self, message: str):
            parser_error["message"] = message
            raise SystemExit(2)

    monkeypatch.setattr(cli, "_build_parser", lambda: FakeParser())

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert exc_info.value.code == 2
    assert parser_error["message"] == (
        "--overwrite and --version-output are mutually exclusive."
    )


def test_cli_surfaces_config_errors(monkeypatch, capsys):
    def fake_load_config(**_):
        raise cfg.ConvertMarkdownConfigError("boom")

    monkeypatch.setattr(cli, "load_config", fake_load_config)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["input.pdf"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "boom" in captured.err


def test_cli_version_flag_sets_collision(monkeypatch):
    captured: dict[str, object] = {}
    load_result = _load_result()

    def fake_load_config(**kwargs):
        captured["call"] = kwargs
        return load_result

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "_build_dependencies", lambda: object())

    def fake_configure_logger(*args, **kwargs):
        return DummyLogger(), Path("/tmp/ws/logs/run.log")

    monkeypatch.setattr(cli, "configure_logger", fake_configure_logger)

    summary = cli.ExecutionSummary(
        requested=(Path("input.pdf"),),
        processed=(Path("input.pdf"),),
        outcomes=(
            converter.ConversionOutcome(
                source=Path("input.pdf"),
                status=converter.ConversionStatus.SUCCESS,
                output_path=Path("/tmp/out/input.md"),
            ),
        ),
    )

    def fake_run_conversion(*args, **kwargs):
        return summary

    monkeypatch.setattr(cli, "run_conversion", fake_run_conversion)

    code = cli.main(["--version-output", "input.pdf"])

    assert code == 0
    overrides = captured["call"]["overrides"]
    assert overrides.collision is cfg.CollisionPolicy.VERSION


def test_cli_reports_dependency_errors(monkeypatch, capsys):
    load_result = _load_result()
    monkeypatch.setattr(cli, "load_config", lambda **_: load_result)

    def fake_build_dependencies():
        raise converter.DependencyError("install markitdown")

    monkeypatch.setattr(cli, "_build_dependencies", fake_build_dependencies)

    # Ensure configure_logger would raise if called to prove short-circuit.
    def fail_configure_logger(*_, **__):
        raise RuntimeError("should not be called")

    monkeypatch.setattr(cli, "configure_logger", fail_configure_logger)

    code = cli.main(["input.pdf"])

    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "install markitdown"


def test_build_dependencies_placeholder_raises():
    with pytest.raises(converter.DependencyError):
        cli._build_dependencies()


def test_cli_config_init_writes_template(tmp_path: Path, capsys):
    template = config_templates.get_template("convert_markdown")
    code = cli.main(
        [
            "config",
            "init",
            "--workspace",
            str(tmp_path),
        ]
    )

    assert code == 0
    target = tmp_path / "config" / cfg.CONFIG_FILENAME
    assert target.exists()
    assert target.read_text(encoding="utf-8") == template.read_text()

    captured = capsys.readouterr()
    assert str(target) in captured.out
    assert captured.err == ""


def test_cli_config_init_path_override_requires_force(
    tmp_path: Path, capsys
) -> None:
    template = config_templates.get_template("convert_markdown")
    custom = tmp_path / "custom.toml"
    custom.write_text("existing", encoding="utf-8")

    code = cli.main(
        [
            "config",
            "init",
            "--path",
            str(custom),
        ]
    )

    assert code == 1
    captured = capsys.readouterr()
    assert "Config already exists" in captured.err
    assert custom.read_text(encoding="utf-8") == "existing"

    code = cli.main(
        [
            "config",
            "init",
            "--path",
            str(custom),
            "--force",
        ]
    )

    assert code == 0
    assert custom.read_text(encoding="utf-8") == template.read_text()

    captured = capsys.readouterr()
    assert str(custom) in captured.out


def test_cli_config_init_relative_path(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.chdir(tmp_path)
    code = cli.main(
        [
            "config",
            "init",
            "--path",
            "local.toml",
        ]
    )

    assert code == 0
    target = tmp_path / "local.toml"
    assert target.exists()

    captured = capsys.readouterr()
    assert str(target.resolve()) in captured.out


def test_cli_config_init_workspace_error(monkeypatch, capsys):
    def fail_workspace(*_args, **_kwargs):
        raise WorkspaceError("workspace boom")

    monkeypatch.setattr(
        cli.workspace_mod,
        "ensure_workspace",
        fail_workspace,
    )

    code = cli.main(["config", "init"])

    assert code == 1
    captured = capsys.readouterr()
    assert "workspace boom" in captured.err

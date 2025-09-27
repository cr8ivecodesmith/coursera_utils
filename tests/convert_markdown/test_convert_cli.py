from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from study_utils.convert_markdown import cli
from study_utils.convert_markdown import config as cfg
from study_utils.core.workspace import WorkspaceLayout


def test_cli_invokes_loader_and_reports_scaffolding(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def fake_load_config(**kwargs):
        captured.update(kwargs)
        layout = WorkspaceLayout(
            home=Path("/tmp/ws"),
            directories={"converted": Path("/tmp/ws/converted")},
            created={"home": True, "converted": True},
        )
        result = cfg.LoadResult(
            config=cfg.ConvertMarkdownConfig(
                extensions=("pdf",),
                output_dir=Path("/tmp/ws/converted"),
                collision=cfg.CollisionPolicy.SKIP,
                log_level="INFO",
            ),
            layout=layout,
            config_path=None,
        )
        return result

    monkeypatch.setattr(cli, "load_config", fake_load_config)

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

    captured_io = capsys.readouterr()
    assert code == 0
    assert "conversion pipeline is pending" in captured_io.out
    assert captured["config_path"] == Path("config.toml")
    assert captured["workspace_path"] == Path("workspace")
    overrides = captured["overrides"]
    assert isinstance(overrides, cfg.ConfigOverrides)
    assert overrides.output_dir == Path("cli-out")
    assert overrides.extensions == ["pdf"]
    assert overrides.collision is cfg.CollisionPolicy.OVERWRITE
    assert overrides.log_level == "debug"


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

    def fake_load_config(**kwargs):
        captured.update(kwargs)
        layout = WorkspaceLayout(
            home=Path("/tmp/ws"),
            directories={"converted": Path("/tmp/ws/converted")},
            created={"home": True, "converted": True},
        )
        return cfg.LoadResult(
            config=cfg.ConvertMarkdownConfig(
                extensions=("pdf",),
                output_dir=Path("/tmp/ws/converted"),
                collision=cfg.CollisionPolicy.SKIP,
                log_level="INFO",
            ),
            layout=layout,
            config_path=None,
        )

    monkeypatch.setattr(cli, "load_config", fake_load_config)

    code = cli.main(["--version-output", "input.pdf"])

    assert code == 0
    overrides = captured["overrides"]
    assert overrides.collision is cfg.CollisionPolicy.VERSION

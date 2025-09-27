from __future__ import annotations

from pathlib import Path

import pytest

from study_utils.convert_markdown import config as cfg


def test_load_config_defaults_use_workspace(tmp_path):
    workspace_root = tmp_path / "workspace"

    result = cfg.load_config(env={}, workspace_path=workspace_root)

    assert result.layout.home == workspace_root.resolve()
    assert result.config_path is None
    assert result.config.output_dir == result.layout.path_for("converted")
    assert result.config.extensions == (
        "pdf",
        "docx",
        "html",
        "txt",
        "epub",
    )
    assert result.config.collision is cfg.CollisionPolicy.SKIP
    assert result.config.log_level == "INFO"


def test_load_config_reads_config_file(tmp_path):
    workspace_root = tmp_path / "ws"
    config_dir = workspace_root / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / cfg.CONFIG_FILENAME
    config_file.write_text(
        """
        [paths]
        output_dir = "custom"

        [execution]
        extensions = ["pdf", "DOCX", ".html"]
        collision = "version"

        [logging]
        level = "warning"
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    result = cfg.load_config(
        config_path=config_file,
        env={},
        workspace_path=workspace_root,
    )

    expected_output = (workspace_root / "custom").resolve()
    assert result.config_path == config_file
    assert result.config.output_dir == expected_output
    assert result.config.extensions == ("pdf", "docx", "html")
    assert result.config.collision is cfg.CollisionPolicy.VERSION
    assert result.config.log_level == "WARNING"


def test_load_config_env_overrides_file(tmp_path):
    workspace_root = tmp_path / "env-ws"
    config_dir = workspace_root / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / cfg.CONFIG_FILENAME
    config_file.write_text(
        """
        [paths]
        output_dir = "file-out"

        [execution]
        extensions = ["pdf"]
        collision = "skip"

        [logging]
        level = "info"
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    env_map = {
        cfg.CONFIG_ENV: str(config_file),
        f"{cfg.ENV_PREFIX}OUTPUT_DIR": str(tmp_path / "env-out"),
        f"{cfg.ENV_PREFIX}EXTENSIONS": "html epub",
        f"{cfg.ENV_PREFIX}COLLISION": "overwrite",
        f"{cfg.ENV_PREFIX}LOG_LEVEL": "error",
    }

    result = cfg.load_config(env=env_map, workspace_path=workspace_root)

    assert result.config.output_dir == (tmp_path / "env-out").resolve()
    assert result.config.extensions == ("html", "epub")
    assert result.config.collision is cfg.CollisionPolicy.OVERWRITE
    assert result.config.log_level == "ERROR"


def test_load_config_cli_overrides_env(tmp_path):
    workspace_root = tmp_path / "cli-ws"
    env_map = {
        f"{cfg.ENV_PREFIX}OUTPUT_DIR": str(tmp_path / "env-out"),
        f"{cfg.ENV_PREFIX}EXTENSIONS": "html",
        f"{cfg.ENV_PREFIX}COLLISION": "skip",
        f"{cfg.ENV_PREFIX}LOG_LEVEL": "warning",
    }
    overrides = cfg.ConfigOverrides(
        output_dir=Path("cli-out"),
        extensions=["txt"],
        collision=cfg.CollisionPolicy.OVERWRITE,
        log_level="debug",
    )

    result = cfg.load_config(
        env=env_map,
        overrides=overrides,
        workspace_path=workspace_root,
    )

    assert result.config.output_dir == (workspace_root / "cli-out").resolve()
    assert result.config.extensions == ("txt",)
    assert result.config.collision is cfg.CollisionPolicy.OVERWRITE
    assert result.config.log_level == "DEBUG"


def test_load_config_missing_explicit_file_raises(tmp_path):
    workspace_root = tmp_path / "missing"
    missing = tmp_path / "does-not-exist.toml"

    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg.load_config(config_path=missing, workspace_path=workspace_root)


def test_load_config_invalid_env_collision_raises(tmp_path):
    workspace_root = tmp_path / "invalid"
    env_map = {f"{cfg.ENV_PREFIX}COLLISION": "bogus"}

    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg.load_config(env=env_map, workspace_path=workspace_root)


def test_load_config_bad_toml_surfaces_error(tmp_path, monkeypatch):
    workspace_root = tmp_path / "bad-toml"
    config_dir = workspace_root / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / cfg.CONFIG_FILENAME
    config_file.write_text("", encoding="utf-8")

    def fake_load_toml(_):
        raise cfg.core_config.TomlConfigError("broken")

    monkeypatch.setattr(cfg.core_config, "load_toml", fake_load_toml)

    with pytest.raises(cfg.ConvertMarkdownConfigError) as exc_info:
        cfg.load_config(config_path=config_file, workspace_path=workspace_root)

    assert "broken" in str(exc_info.value)


def test_load_config_merge_error_surfaces(tmp_path):
    workspace_root = tmp_path / "bad-merge"
    config_dir = workspace_root / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / cfg.CONFIG_FILENAME
    config_file.write_text("[unexpected]\nvalue = 1\n", encoding="utf-8")

    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg.load_config(config_path=config_file, workspace_path=workspace_root)


def test_coerce_optional_path_handles_values():
    path = Path("foo")
    assert cfg._coerce_optional_path(path) is path
    assert cfg._coerce_optional_path("  ") is None
    assert cfg._coerce_optional_path("bar") == Path("bar")
    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg._coerce_optional_path(123)


def test_normalize_extensions_validation():
    assert cfg._normalize_extensions(["pdf", ".PDF"]) == ("pdf",)
    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg._normalize_extensions(None)
    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg._normalize_extensions(["   "])
    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg._normalize_extensions([])


def test_resolve_collision_validation():
    assert (
        cfg._resolve_collision(None, None, cfg.CollisionPolicy.VERSION)
        is cfg.CollisionPolicy.VERSION
    )
    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg._resolve_collision(None, None, 42)


def test_resolve_log_level_validation():
    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg._resolve_log_level(None, None, None)
    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg._resolve_log_level(None, None, 123)
    with pytest.raises(cfg.ConvertMarkdownConfigError):
        cfg._resolve_log_level(None, None, "   ")

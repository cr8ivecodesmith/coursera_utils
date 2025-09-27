from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from study_utils.core import workspace
from study_utils.rag import data_dir


def test_get_data_home_uses_env_and_creates_subdirs(tmp_path, monkeypatch):
    root = tmp_path / "rag-data"
    monkeypatch.setenv(data_dir.DATA_HOME_ENV, str(root))

    home = data_dir.get_data_home()

    assert home == root.resolve()
    for expected in ("config", "rag_dbs", "rag_sessions", "logs"):
        assert (home / expected).is_dir()


def test_get_data_home_rejects_file_path(tmp_path, monkeypatch):
    root = tmp_path / "as-file"
    root.write_text("not a dir", encoding="utf-8")
    monkeypatch.setenv(data_dir.DATA_HOME_ENV, str(root))

    with pytest.raises(data_dir.DataDirError):
        data_dir.get_data_home()


def test_require_subdir_rejects_file_when_not_creating(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv(data_dir.DATA_HOME_ENV, str(root))
    sub = data_dir.require_subdir("config")
    file_path = sub
    file_path.rmdir()
    file_path.write_text("oops", encoding="utf-8")

    with pytest.raises(data_dir.DataDirError):
        data_dir.require_subdir("config", create=False)


def test_config_path_detects_directory_conflict(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv(data_dir.DATA_HOME_ENV, str(root))
    cfg_dir = data_dir.config_dir()
    bad = cfg_dir / "rag.toml"
    bad.mkdir()

    with pytest.raises(data_dir.DataDirError):
        data_dir.config_path()


def test_describe_layout_returns_all_paths(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv(data_dir.DATA_HOME_ENV, str(root))
    layout = data_dir.describe_layout()

    assert layout["home"] == root.resolve()
    for key in ("config", "vector_dbs", "sessions", "logs"):
        assert isinstance(layout[key], Path)


def test_require_subdir_unknown_key():
    with pytest.raises(KeyError):
        data_dir.require_subdir("unknown")


def test_get_data_home_with_custom_env_mapping(tmp_path):
    env = {data_dir.DATA_HOME_ENV: str(tmp_path / "mapped")}
    home = data_dir.get_data_home(env=env)
    assert home == (tmp_path / "mapped").resolve()


def test_get_data_home_handles_resolve_failure(tmp_path, monkeypatch):
    target = tmp_path / "broken"

    original_resolve = Path.resolve

    def fake_resolve(self):
        if self == target.expanduser():
            raise FileNotFoundError
        return original_resolve(self)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    env = {data_dir.DATA_HOME_ENV: str(target)}

    home = data_dir.get_data_home(env=env)
    assert home == target.expanduser().absolute()


def test_get_data_home_ignores_chmod_errors(tmp_path, monkeypatch):
    monkeypatch.setenv(data_dir.DATA_HOME_ENV, str(tmp_path / "chmod"))

    def raise_permission(self, mode):  # noqa: ARG001
        raise PermissionError

    monkeypatch.setattr(Path, "chmod", raise_permission)

    home = data_dir.get_data_home()
    assert home.exists()


def test_vector_session_log_dir_helpers(tmp_path, monkeypatch):
    monkeypatch.setenv(data_dir.DATA_HOME_ENV, str(tmp_path / "dirs"))

    cfg = data_dir.config_dir()
    vec = data_dir.vector_db_dir()
    ses = data_dir.sessions_dir()
    logs = data_dir.logs_dir()

    assert cfg.is_dir() and vec.is_dir() and ses.is_dir() and logs.is_dir()


def test_get_data_home_uses_default_when_env_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_dir, "DEFAULT_DATA_HOME", tmp_path / "default")
    home = data_dir.get_data_home(env={})
    assert home == (tmp_path / "default").resolve()


def test_require_subdir_create_false_guard(tmp_path, monkeypatch):
    file_path = tmp_path / "config-file"
    file_path.write_text("not a directory", encoding="utf-8")

    layout = workspace.WorkspaceLayout(
        home=tmp_path,
        directories=MappingProxyType({"config": file_path}),
        created=MappingProxyType({"home": False, "config": False}),
    )

    monkeypatch.setattr(
        data_dir,
        "_resolve_layout",
        lambda env=None, create=False: layout,
    )

    with pytest.raises(data_dir.DataDirError):
        data_dir.config_dir(create=False)

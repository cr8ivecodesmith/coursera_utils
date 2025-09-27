from __future__ import annotations

from pathlib import Path

import pytest

from study_utils.core import workspace


def test_ensure_workspace_creates_directories(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(root))

    layout = workspace.ensure_workspace()

    assert layout.home == root.resolve()
    for name, path in layout.items():
        assert path.is_dir()
        assert layout.created[name] is True
    assert layout.created["home"] is True


def test_ensure_workspace_is_idempotent(tmp_path, monkeypatch):
    root = tmp_path / "existing"
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(root))

    first = workspace.ensure_workspace()
    second = workspace.ensure_workspace()

    assert first.home == second.home
    assert all(not created for created in second.created.values())


def test_ensure_workspace_respects_custom_path(tmp_path):
    custom = tmp_path / "custom-root"

    layout = workspace.ensure_workspace(path=custom)

    assert layout.home == custom.resolve()


def test_ensure_workspace_without_create(tmp_path, monkeypatch):
    root = tmp_path / "deferred"
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(root))

    layout = workspace.ensure_workspace(create=False)

    assert layout.home == root.resolve()
    assert not root.exists()
    assert all(not created for created in layout.created.values())


def test_describe_layout_returns_mapping(tmp_path, monkeypatch):
    root = tmp_path / "layout"
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(root))

    mapping = workspace.describe_layout()

    assert mapping["home"] == root.resolve()
    assert "config" in mapping


def test_ensure_workspace_errors_when_path_is_file(tmp_path, monkeypatch):
    root = tmp_path / "file"
    root.write_text("not a dir", encoding="utf-8")
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(root))

    with pytest.raises(workspace.WorkspaceError):
        workspace.ensure_workspace()


def test_path_for_unknown_key_errors(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(root))

    layout = workspace.ensure_workspace(create=False)

    with pytest.raises(KeyError):
        layout.path_for("unknown")


def test_ensure_workspace_uses_env_mapping(tmp_path):
    env_home = tmp_path / "env-home"
    env = {workspace.WORKSPACE_ENV: str(env_home)}

    layout = workspace.ensure_workspace(env=env)

    assert layout.home == env_home.resolve()


def test_workspace_fallback_on_permission(tmp_path, monkeypatch):
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(workspace, "_fallback_base", lambda: fallback)
    real_ensure_dir = workspace._ensure_dir

    def fake_ensure_dir(path: Path) -> bool:
        if path == workspace.DEFAULT_WORKSPACE:
            raise PermissionError("denied")
        return real_ensure_dir(path)

    monkeypatch.setattr(workspace, "_ensure_dir", fake_ensure_dir)
    monkeypatch.delenv(workspace.WORKSPACE_ENV, raising=False)

    layout = workspace.ensure_workspace()

    assert layout.home == fallback.resolve()
    assert layout.created["home"] is True


def test_ensure_dir_detects_non_directory(tmp_path, monkeypatch):
    target = tmp_path / "shall-be-dir"
    real_is_dir = Path.is_dir

    def fake_is_dir(self: Path) -> bool:
        if self == target:
            return False
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", fake_is_dir)

    with pytest.raises(workspace.WorkspaceError):
        workspace._ensure_dir(target)


def test_workspace_error_when_all_candidates_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_fallback_base", lambda: tmp_path / "fb")
    monkeypatch.delenv(workspace.WORKSPACE_ENV, raising=False)

    def deny(path: Path) -> bool:  # noqa: D401
        raise PermissionError("nope")

    monkeypatch.setattr(workspace, "_ensure_dir", deny)

    with pytest.raises(workspace.WorkspaceError):
        workspace.ensure_workspace()

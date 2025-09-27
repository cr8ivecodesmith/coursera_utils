"""Shared workspace bootstrap helpers for study-utils commands."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, MutableMapping


WORKSPACE_ENV = "STUDY_UTILS_DATA_HOME"
DEFAULT_WORKSPACE = Path.home() / ".study-utils-data"

# Logical directory names maintained for backward compatibility with the
# existing RAG tooling while giving new commands (like convert-markdown) a
# stable place to write outputs.
_SUBDIRS = {
    "config": "config",
    "logs": "logs",
    "rag_dbs": "rag_dbs",
    "rag_sessions": "rag_sessions",
    "converted": "converted",
}


class WorkspaceError(RuntimeError):
    """Raised when the workspace layout cannot be prepared."""


@dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved workspace paths and creation metadata."""

    home: Path
    directories: Mapping[str, Path]
    created: Mapping[str, bool]

    def path_for(self, key: str) -> Path:
        try:
            return self.directories[key]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown workspace directory '{key}'.") from exc

    def items(self) -> tuple[tuple[str, Path], ...]:
        """Return a tuple of directory name/path pairs."""

        return tuple(self.directories.items())


def ensure_workspace(
    *,
    env: Mapping[str, str] | None = None,
    path: Path | None = None,
    create: bool = True,
    subdirs: Mapping[str, str] | None = None,
) -> WorkspaceLayout:
    """Ensure the shared workspace exists and return its layout."""

    env_map = _coerce_env(env)
    resolved_subdirs = dict(subdirs or _SUBDIRS)
    base, has_override = _resolve_base(env_map, override=path)

    candidates: list[Path] = [base]
    if create and not has_override:
        fallback = _fallback_base()
        if fallback != base:
            candidates.append(fallback)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return _materialize_layout(
                base=candidate,
                create=create,
                subdirs=resolved_subdirs,
            )
        except PermissionError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise WorkspaceError(
            "Unable to prepare workspace at {0}".format(base)
        ) from last_error


def describe_layout(
    *, env: Mapping[str, str] | None = None, path: Path | None = None
) -> Mapping[str, Path]:
    """Return the workspace layout without creating directories."""

    layout = ensure_workspace(env=env, path=path, create=False)
    mapping: MutableMapping[str, Path] = {
        "home": layout.home,
    }
    mapping.update(layout.directories)
    return MappingProxyType(dict(mapping))


def _coerce_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is None:
        return os.environ
    return env


def _resolve_base(
    env: Mapping[str, str], *, override: Path | None
) -> tuple[Path, bool]:
    if override is not None:
        target = override
        provided = True
    else:
        custom = env.get(WORKSPACE_ENV)
        if custom is not None:
            custom = custom.strip()
        if custom:
            target = Path(custom).expanduser()
            provided = True
        else:
            target = DEFAULT_WORKSPACE
            provided = False
    try:
        return target.expanduser().resolve(), provided
    except FileNotFoundError:
        return target.expanduser().absolute(), provided


def _fallback_base() -> Path:
    return Path(tempfile.gettempdir()) / "study-utils-data"


def _materialize_layout(
    *, base: Path, create: bool, subdirs: Mapping[str, str]
) -> WorkspaceLayout:
    _validate_candidate(base)

    created: MutableMapping[str, bool] = {}

    if create:
        created["home"] = _ensure_dir(base)
    else:
        created["home"] = False

    directories: MutableMapping[str, Path] = {}
    for key, relative in subdirs.items():
        candidate = base / relative
        if create:
            created[key] = _ensure_dir(candidate)
        else:
            created[key] = False
            if candidate.exists() and not candidate.is_dir():
                message = (
                    "Expected workspace directory for '{0}' but found a file: "
                    "{1}"
                ).format(key, candidate)
                raise WorkspaceError(message)
        directories[key] = candidate

    return WorkspaceLayout(
        home=base,
        directories=MappingProxyType(dict(directories)),
        created=MappingProxyType(dict(created)),
    )


def _validate_candidate(path: Path) -> None:
    if path.exists() and not path.is_dir():
        message = (
            "Configured workspace exists and is not a directory: {0}"
        ).format(path)
        raise WorkspaceError(message)


def _ensure_dir(path: Path) -> bool:
    existed = path.exists()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:  # pragma: no cover - depends on platform
        message = (
            "Expected directory but found a non-directory entry: {0}"
        ).format(path)
        raise WorkspaceError(message) from exc
    _chmod_safe(path, 0o700)
    if path.exists() and not path.is_dir():
        message = (
            "Expected directory but found a non-directory entry: {0}"
        ).format(path)
        raise WorkspaceError(message)
    return not existed


def _chmod_safe(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except (PermissionError, NotImplementedError):
        return

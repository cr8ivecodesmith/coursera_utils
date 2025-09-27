"""Configuration loader for the convert-markdown workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence

from study_utils.core import config as core_config
from study_utils.core import workspace as workspace_mod

CONFIG_FILENAME = "convert_markdown.toml"
CONFIG_ENV = "STUDY_CONVERT_MARKDOWN_CONFIG"
ENV_PREFIX = "STUDY_CONVERT_MARKDOWN_"

_DEFAULT_EXTENSIONS: tuple[str, ...] = (
    "pdf",
    "docx",
    "html",
    "txt",
    "epub",
)
_DEFAULT_COLLISION = "skip"
_DEFAULT_LOG_LEVEL = "INFO"


class ConvertMarkdownConfigError(RuntimeError):
    """Raised when configuration parsing or validation fails."""


class CollisionPolicy(Enum):
    """Supported strategies for handling name collisions."""

    SKIP = "skip"
    OVERWRITE = "overwrite"
    VERSION = "version"

    @classmethod
    def from_value(cls, value: str) -> "CollisionPolicy":
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        expected = ", ".join(member.value for member in cls)
        raise ConvertMarkdownConfigError(
            f"Unknown collision policy '{value}'. Expected one of: {expected}."
        )


@dataclass(frozen=True)
class ConvertMarkdownConfig:
    """Fully resolved configuration for a conversion run."""

    extensions: tuple[str, ...]
    output_dir: Path
    collision: CollisionPolicy
    log_level: str


@dataclass(frozen=True)
class ConfigOverrides:
    """CLI-sourced overrides applied on top of file/env options."""

    extensions: Optional[Sequence[str]] = None
    output_dir: Optional[Path] = None
    collision: Optional[CollisionPolicy] = None
    log_level: Optional[str] = None


@dataclass(frozen=True)
class LoadResult:
    """Result of loading configuration, including workspace context."""

    config: ConvertMarkdownConfig
    layout: workspace_mod.WorkspaceLayout
    config_path: Optional[Path]


def load_config(
    *,
    config_path: Optional[Path] = None,
    overrides: Optional[ConfigOverrides] = None,
    env: Optional[Mapping[str, str]] = None,
    workspace_path: Optional[Path] = None,
) -> LoadResult:
    """Load configuration applying precedence CLI > env > TOML defaults."""

    overrides = overrides or ConfigOverrides()
    env_map = env or os.environ

    layout = workspace_mod.ensure_workspace(env=env_map, path=workspace_path)
    default_path = layout.path_for("config") / CONFIG_FILENAME

    requested_path = _resolve_config_path(
        config_path=config_path,
        env_map=env_map,
        default_path=default_path,
    )

    defaults = _default_table()
    config_exists = requested_path.exists()
    loaded_path: Optional[Path]

    if config_exists:
        loaded_path = requested_path
        try:
            parsed = core_config.load_toml(requested_path)
        except core_config.TomlConfigError as exc:
            raise ConvertMarkdownConfigError(str(exc)) from exc
        try:
            core_config.merge_defaults(defaults, parsed)
        except core_config.TomlConfigError as exc:
            raise ConvertMarkdownConfigError(str(exc)) from exc
    else:
        loaded_path = None
        if config_path is not None or _has_env_config(env_map):
            raise ConvertMarkdownConfigError(
                f"Config file not found: {requested_path}"
            )

    file_options = defaults

    env_output_dir = _parse_env_path(env_map, "OUTPUT_DIR")
    env_extensions = _parse_env_extensions(env_map)
    env_collision = _parse_env_collision(env_map)
    env_log_level = _parse_env_string(env_map, "LOG_LEVEL")

    output_dir = _resolve_output_dir(
        candidate=_pick_first(
            overrides.output_dir,
            env_output_dir,
            _coerce_optional_path(file_options["paths"]["output_dir"]),
        ),
        layout=layout,
    )

    extensions = _normalize_extensions(
        _pick_first(
            overrides.extensions,
            env_extensions,
            file_options["execution"]["extensions"],
        )
    )

    collision = _resolve_collision(
        overrides.collision,
        env_collision,
        file_options["execution"]["collision"],
    )

    log_level = _resolve_log_level(
        overrides.log_level,
        env_log_level,
        file_options["logging"]["level"],
    )

    config = ConvertMarkdownConfig(
        extensions=extensions,
        output_dir=output_dir,
        collision=collision,
        log_level=log_level,
    )
    return LoadResult(config=config, layout=layout, config_path=loaded_path)


def _default_table() -> MutableMapping[str, MutableMapping[str, object]]:
    return {
        "paths": {"output_dir": None},
        "execution": {
            "extensions": list(_DEFAULT_EXTENSIONS),
            "collision": _DEFAULT_COLLISION,
        },
        "logging": {"level": _DEFAULT_LOG_LEVEL},
    }


def _resolve_config_path(
    *,
    config_path: Optional[Path],
    env_map: Mapping[str, str],
    default_path: Path,
) -> Path:
    if config_path is not None:
        return config_path.expanduser()
    env_candidate = env_map.get(CONFIG_ENV)
    if env_candidate:
        return Path(env_candidate).expanduser()
    return default_path


def _has_env_config(env_map: Mapping[str, str]) -> bool:
    env_candidate = env_map.get(CONFIG_ENV)
    return bool(env_candidate and env_candidate.strip())


def _coerce_optional_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        return Path(raw)
    raise ConvertMarkdownConfigError(
        "paths.output_dir must be a string when provided."
    )


def _resolve_output_dir(
    *, candidate: Optional[Path], layout: workspace_mod.WorkspaceLayout
) -> Path:
    if candidate is None:
        return layout.path_for("converted")
    if not candidate.is_absolute():
        return (layout.home / candidate).resolve()
    return candidate.expanduser().resolve()


def _normalize_extensions(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        raise ConvertMarkdownConfigError(
            "At least one extension must be configured."
        )
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        stripped = item.strip().lower()
        if not stripped:
            raise ConvertMarkdownConfigError(
                "Extensions must be non-empty strings."
            )
        normalized = stripped.lstrip(".")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    if not result:
        raise ConvertMarkdownConfigError(
            "At least one extension must be configured."
        )
    return tuple(result)


def _resolve_collision(
    override: Optional[CollisionPolicy],
    env_value: Optional[CollisionPolicy],
    file_value: object,
) -> CollisionPolicy:
    if override is not None:
        return override
    if env_value is not None:
        return env_value
    if isinstance(file_value, CollisionPolicy):
        return file_value
    if isinstance(file_value, str):
        return CollisionPolicy.from_value(file_value)
    raise ConvertMarkdownConfigError(
        "execution.collision must be one of: skip, overwrite, version."
    )


def _resolve_log_level(
    override: Optional[str],
    env_value: Optional[str],
    file_value: object,
) -> str:
    candidate = _pick_first(override, env_value, file_value)
    if candidate is None:
        raise ConvertMarkdownConfigError("logging.level must be provided.")
    if not isinstance(candidate, str):
        raise ConvertMarkdownConfigError("logging.level must be a string.")
    level = candidate.strip()
    if not level:
        raise ConvertMarkdownConfigError(
            "logging.level must be a non-empty string."
        )
    return level.upper()


def _parse_env_extensions(
    env_map: Mapping[str, str],
) -> Optional[Sequence[str]]:
    raw = _parse_env_string(env_map, "EXTENSIONS")
    if raw is None:
        return None
    parts = [part for part in raw.replace(",", " ").split() if part]
    return parts or None


def _parse_env_collision(
    env_map: Mapping[str, str],
) -> Optional[CollisionPolicy]:
    raw = _parse_env_string(env_map, "COLLISION")
    if raw is None:
        return None
    return CollisionPolicy.from_value(raw)


def _parse_env_path(env_map: Mapping[str, str], key: str) -> Optional[Path]:
    raw = _parse_env_string(env_map, key)
    if raw is None:
        return None
    return Path(raw).expanduser()


def _parse_env_string(env_map: Mapping[str, str], key: str) -> Optional[str]:
    raw = env_map.get(f"{ENV_PREFIX}{key}")
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _pick_first(*candidates: object) -> object:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None

"""Configuration helpers for the generate-document command."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from study_utils.core import workspace as workspace_mod
from study_utils.core.workspace import WorkspaceError


CONFIG_FILENAME = "documents.toml"


@dataclass(frozen=True)
class GenerateOptions:
    """CLI-facing options derived from argument parsing."""

    extensions: set[str]
    level_limit: int
    config_path: Path
    doc_type: str


def find_config_path(
    arg: Optional[str],
    *,
    workspace_path: Optional[Path] = None,
) -> Path:
    """Locate the documents config respecting CLI overrides and workspace."""

    if arg:
        path = Path(arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return path

    try:
        layout = workspace_mod.ensure_workspace(
            path=workspace_path,
            create=False,
        )
    except WorkspaceError as exc:
        raise FileNotFoundError(str(exc)) from exc

    workspace_candidate = layout.path_for("config") / CONFIG_FILENAME
    if workspace_candidate.exists():
        return workspace_candidate.resolve()

    cwd_cfg = Path.cwd() / CONFIG_FILENAME
    if cwd_cfg.exists():
        return cwd_cfg.resolve()

    raise FileNotFoundError(
        "documents.toml not found. Run `study generate-document config init`."
    )


def load_documents_config(path: Path) -> Dict[str, Dict[str, str]]:
    """Load a TOML file and return a dict of document types."""
    try:
        try:
            import tomllib  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback for older Pythons
            import tomli as tomllib  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "Python 3.11+ required (tomllib) to read TOML config"
        ) from exc

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    if not isinstance(raw, dict) or not raw:
        raise ValueError("documents.toml is empty or invalid")

    data: Dict[str, Dict[str, str]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        entry = {
            "prompt": prompt.strip(),
            "description": str(value.get("description", "")).strip(),
            "model": str(
                value.get(
                    "model",
                    os.getenv("OPENAI_TITLE_MODEL", "gpt-4o-mini"),
                )
            ).strip(),
        }
        data[str(key).strip()] = entry

    if not data:
        raise ValueError(
            "documents.toml does not contain any valid document types"
        )

    return data

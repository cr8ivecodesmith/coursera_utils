import re
import json

from pathlib import Path
from typing import Optional, List, Sequence, Tuple

try:  # shared helpers for file discovery and AI access
    from ..core import (
        iter_text_files,
        load_client,
        parse_extensions,
        read_text_file,
    )  # type: ignore
except Exception:  # pragma: no cover - fallback for alternate execution
    try:
        from study_utils.core import (
            iter_text_files,
            load_client,
            parse_extensions,
            read_text_file,
        )  # type: ignore
    except Exception:  # pragma: no cover - final fallback when unavailable
        load_client = None  # type: ignore


_slug_re = re.compile(r"[^a-z0-9]+")


def _find_config(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    p = Path("quizzer.toml").resolve()
    if p.exists():
        return p
    # Fallback to bundled defaults (not provided yet)
    return None


def _get_quiz_section(cfg: dict, name: str) -> dict:
    root = cfg.get("quiz") or {}
    sec = root.get(name)
    if not isinstance(sec, dict):
        raise KeyError(f"Quiz section not found: [quiz.{name}]")
    return sec


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # type: ignore[attr-defined]
    except (
        Exception
    ):  # pragma: no cover - fallback for <3.11 if tomli is installed
        try:
            import tomli as tomllib  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "TOML support not available. Use Python 3.11+ or install "
                "'tomli'."
            ) from exc
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _read_files(files: Sequence[Path]) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    for p in files:
        try:
            text = read_text_file(p)
        except Exception:
            text = ""
        out.append((p, text))
    return out


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = _slug_re.sub("-", s).strip("-")
    return s or "topic"


def iter_quiz_files(
    paths: Sequence[Path],
    extensions: Sequence[str] = ("md", "markdown"),
    level_limit: int = 0,
) -> List[Path]:
    """Return a deterministic list of Markdown files to use as sources.

    - Include a path when it is a file that matches one of the extensions.
    - Traverse directories with optional depth control.
      ``level_limit == 1`` includes only files directly under the directory.
      ``level_limit == 2`` includes one subdirectory level, and so on.
      ``0`` means no limit.
    Files are returned in ascending name order for determinism and alignment
    with tests.
    """
    if level_limit < 0:
        raise ValueError("level_limit must be >= 0")

    exts = parse_extensions(extensions, default=extensions)
    collected: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            matches = list(iter_text_files([path], exts, level_limit))
        except FileNotFoundError:
            continue
        collected.extend(matches)
    return collected


def read_jsonl(path: Path) -> List[dict]:
    data: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")

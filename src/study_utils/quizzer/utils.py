import re
import json

from pathlib import Path
from typing import Optional, List, Sequence, Tuple, Set

try:  # optional: reuse existing OpenAI client loader
    from ..transcribe_video import load_client  # type: ignore
except Exception:  # pragma: no cover - fallback for legacy layout
    try:
        from study_utils.transcribe_video import load_client  # type: ignore
    except Exception:  # pragma: no cover
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
            text = p.read_text(encoding="utf-8", errors="replace")
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

    exts = {e.lower().lstrip(".") for e in extensions}
    collected: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        collected.extend(_collect_quiz_files_from_path(path, exts, level_limit))
    return collected


def _collect_quiz_files_from_path(
    path: Path, extensions: Set[str], level_limit: int
) -> List[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path] if _matches_extension(path, extensions) else []
    if not path.is_dir():
        return []
    return [
        file
        for file in _iter_directory_files(path, level_limit)
        if _matches_extension(file, extensions)
    ]


def _matches_extension(path: Path, extensions: Set[str]) -> bool:
    return path.is_file() and path.suffix.lower().lstrip(".") in extensions


def _iter_directory_files(base: Path, level_limit: int) -> List[Path]:
    files = sorted(
        (child for child in base.rglob("*") if child.is_file()),
        key=lambda x: x.name.lower(),
    )
    if level_limit == 0:
        return files
    limited: List[Path] = []
    for child in files:
        try:
            rel = child.relative_to(base)
        except Exception:
            continue
        if len(rel.parts) <= level_limit:
            limited.append(child)
    return limited


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

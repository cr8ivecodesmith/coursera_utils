"""Generate a Markdown document from reference files using an AI prompt.

Implements the utility described in spec.md (generate_document.py):
- Required: document type (matches a key in documents.toml)
- Required: output filename (markdown)
- Required: one or more reference files and/or directories
- Optional: --extensions, --level-limit, --config

Design notes:
- Pure helpers for parsing, discovery, and prompt building; isolate I/O in main
- Reuse discovery from text_combiner for consistency
- Use OpenAI via load_client from transcribe_video (env + .env support)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


# Discovery utilities are shared with text_combiner for consistent behavior
try:
    from .text_combiner import iter_text_files, read_text_file  # type: ignore
except Exception:  # pragma: no cover - fallback if relative import fails
    from text_combiner import iter_text_files, read_text_file  # type: ignore

try:  # Prefer relative import for src/ layout
    from .transcribe_video import load_client  # type: ignore
except Exception:  # pragma: no cover - fallback when executed differently
    from study_utils.transcribe_video import load_client  # type: ignore


# -----------------------------
# Types
# -----------------------------


@dataclass(frozen=True)
class GenerateOptions:
    extensions: Set[str]
    level_limit: int
    config_path: Path
    doc_type: str


# -----------------------------
# Parsing helpers
# -----------------------------


def parse_extensions(values: Optional[Sequence[str]]) -> Set[str]:
    """Normalize extension strings to a set without leading dots.

    Defaults to markdown-friendly set: {"txt", "md", "markdown"}.
    """
    default = {"txt", "md", "markdown"}
    if not values:
        return default
    out: Set[str] = set()
    for v in values:
        if not isinstance(v, str):
            continue
        s = v.strip().lower()
        if s.startswith("."):
            s = s[1:]
        if s:
            out.add(s)
    return out or default


def find_config_path(arg: Optional[str]) -> Path:
    """Return the path to the documents config (toml).

    Priority:
    1) --config if provided
    2) ./documents.toml in current working directory
    3) study_utils/documents.toml alongside this module
    Raises FileNotFoundError if none exist.
    """
    if arg:
        p = Path(arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        return p
    # cwd
    cwd_cfg = Path.cwd() / "documents.toml"
    if cwd_cfg.exists():
        return cwd_cfg.resolve()
    # bundled default
    bundled = Path(__file__).resolve().parent / "documents.toml"
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        "documents.toml not found (checked CWD and study_utils/)"
    )


def load_documents_config(path: Path) -> Dict[str, Dict[str, str]]:
    """Load a TOML file and return a dict of document types.

    Each top-level key should map to a dict with fields: model, description, prompt.
    """
    try:
        try:  # Python 3.11+
            import tomllib  # type: ignore
        except Exception:  # pragma: no cover - fallback for older Pythons
            import tomli as tomllib  # type: ignore
    except Exception as exc:  # no toml parser available
        raise RuntimeError(
            "Python 3.11+ required (tomllib) to read TOML config"
        ) from exc

    data: Dict[str, Dict[str, str]]
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    if not isinstance(raw, dict) or not raw:
        raise ValueError("documents.toml is empty or invalid")

    # normalize/validate
    data = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        prompt = v.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        entry = {
            "prompt": prompt.strip(),
            "description": str(v.get("description", "")).strip(),
            "model": str(
                v.get("model", os.getenv("OPENAI_TITLE_MODEL", "gpt-4o-mini"))
            ).strip(),
        }
        data[str(k).strip()] = entry

    if not data:
        raise ValueError(
            "documents.toml does not contain any valid document types"
        )
    return data


# -----------------------------
# Prompt building
# -----------------------------


def build_reference_block(files: Sequence[Tuple[Path, str]]) -> str:
    """Build a single string that contains all reference files and contents.

    Uses clear separators so the model can attribute content to files.
    """
    parts: List[str] = []
    for p, content in files:
        parts.append(
            "\n".join(
                [
                    "\n---",
                    f"File: {p.name}",
                    f"Path: {p}",
                    "Content:",
                    content,
                ]
            )
        )
    return "\n".join(parts)


def build_messages(
    doc_cfg: Dict[str, str], files: Sequence[Tuple[Path, str]]
) -> List[Dict[str, str]]:
    """Construct chat messages for the selected document type."""
    system = (
        "You are an expert writing assistant. "
        "Follow the user's instructions exactly. "
        "Only use the provided reference content. "
        "Output valid, readable Markdown for humans."
    )
    prompt = doc_cfg["prompt"]
    refs = build_reference_block(files)
    user = (
        f"Instructions:\n{prompt}\n\n"
        "Reference files are provided below. "
        "Use only these sources; do not invent facts.\n\n"
        f"{refs}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# -----------------------------
# Core generation
# -----------------------------


def generate_document(
    doc_type: str,
    output_path: Path,
    inputs: Sequence[Path],
    extensions: Set[str],
    level_limit: int,
    config_path: Path,
) -> int:
    """Generate a document and write it to `output_path`.

    Returns the number of reference files used.
    Raises on configuration or API errors.
    """
    cfg_all = load_documents_config(config_path)
    if doc_type not in cfg_all:
        known = ", ".join(sorted(cfg_all.keys()))
        raise ValueError(f"Unknown document type '{doc_type}'. Known: {known}")

    # Discover and read reference files
    discovered = list(iter_text_files(inputs, extensions, level_limit))
    if not discovered:
        raise FileNotFoundError("No matching reference files found")
    ref_pairs: List[Tuple[Path, str]] = [
        (p, read_text_file(p)) for p in discovered
    ]

    # Prepare AI call
    client = load_client()
    doc_cfg = cfg_all[doc_type]
    messages = build_messages(doc_cfg, ref_pairs)
    model = doc_cfg.get("model") or os.getenv(
        "OPENAI_TITLE_MODEL", "gpt-4o-mini"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=4096,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError(
            "AI returned empty content; check API key/model and inputs"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return len(ref_pairs)


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a Markdown document from reference files using an AI prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "DOC_TYPE", help="Document type key defined in documents.toml"
    )
    p.add_argument("OUTPUT", help="Output markdown filename")
    p.add_argument(
        "INPUTS", nargs="+", help="Reference files and/or directories to scan"
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        help="File extensions to include (e.g. txt md markdown). Defaults include txt, md, markdown",
    )
    p.add_argument(
        "--level-limit",
        type=int,
        default=0,
        help="Directory depth to traverse for directories",
    )
    p.add_argument(
        "--config",
        help="Path to documents.toml. Defaults to ./documents.toml then bundled study_utils/documents.toml",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.OUTPUT).expanduser().resolve()
    input_paths = [Path(s).expanduser().resolve() for s in args.INPUTS]
    doc_type = str(args.DOC_TYPE).strip()

    try:
        config_path = find_config_path(args.config)
        extensions = parse_extensions(args.extensions)
        if args.level_limit < 0:
            raise ValueError("--level-limit must be >= 0")
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)

    try:
        used = generate_document(
            doc_type=doc_type,
            output_path=output_path,
            inputs=input_paths,
            extensions=extensions,
            level_limit=int(args.level_limit),
            config_path=config_path,
        )
    except Exception as exc:
        print(f"Failed to generate document: {exc}")
        raise SystemExit(1)

    print(f"Generated document from {used} reference file(s) -> {output_path}")


if __name__ == "__main__":
    main()

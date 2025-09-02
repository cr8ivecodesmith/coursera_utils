"""Quizzer core helpers and CLI parser.

Implements pure, testable helpers used by tests:
- iter_quiz_files: discover Markdown files with depth control
- extract_topics: derive topics from Markdown headings (H1/H2)
- validate_mcq: strict validation for multiple-choice question dicts
- select_questions: selection strategies (balanced, random, weakness)
- aggregate_summary: compute overall and per-topic stats
- read_jsonl / write_jsonl: simple JSONL utilities
- build_arg_parser: CLI subcommands skeleton used by tests

Note: This file purposefully does not perform I/O or AI calls beyond utilities;
the CLI entry points and session runner will be added in later iterations.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional: reuse existing OpenAI client loader
    from .transcribe_video import load_client  # type: ignore
except Exception:  # pragma: no cover - fallback
    try:
        from transcribe_video import load_client  # type: ignore
    except Exception:  # pragma: no cover
        load_client = None  # type: ignore


# -----------------------------
# Discovery
# -----------------------------


def iter_quiz_files(
    paths: Sequence[Path],
    extensions: Sequence[str] = ("md", "markdown"),
    level_limit: int = 0,
) -> List[Path]:
    """Return a deterministic list of Markdown files to use as sources.

    - If a path is a file and matches an extension -> include
    - If a path is a directory -> traverse with depth control
      Depth semantics: level_limit == 1 includes only files directly under the directory.
      level_limit == 2 includes one subdirectory level, and so on. 0 means no limit.
    Files are returned in ascending name order for determinism and to align with tests.
    """
    if level_limit < 0:
        raise ValueError("level_limit must be >= 0")

    exts = {e.lower().lstrip(".") for e in extensions}
    out: List[Path] = []

    def _match(p: Path) -> bool:
        return p.is_file() and p.suffix.lower().lstrip(".") in exts

    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.is_file():
            if _match(p):
                out.append(p)
            continue
        if not p.is_dir():
            continue
        if level_limit == 0:
            files = sorted((c for c in p.rglob("*") if c.is_file()), key=lambda x: x.name.lower())
            for f in files:
                if _match(f):
                    out.append(f)
        else:
            files = sorted((c for c in p.rglob("*") if c.is_file()), key=lambda x: x.name.lower())
            for f in files:
                try:
                    rel = f.relative_to(p)
                except Exception:
                    continue
                if len(rel.parts) <= level_limit and _match(f):
                    out.append(f)
    return out


# -----------------------------
# Topic extraction
# -----------------------------


_slug_re = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = _slug_re.sub("-", s).strip("-")
    return s or "topic"


def extract_topics(
    sources: Iterable[Tuple[Path, str]],
    *,
    use_ai: bool = False,
    client: object = None,
    ai_prompt: Optional[str] = None,
    k: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Extract topics from Markdown sources.

    Strategy:
    - Use H1/H2 headings (lines starting with '#' or '##') as candidate topics
    - Deduplicate by slug; keep first occurrence
    - Record simple description (first line after heading if available)
    - Track source file path for traceability
    """
    topics: Dict[str, Dict[str, object]] = {}
    for path, text in sources:
        lines = (text or "").splitlines()
        for i, raw in enumerate(lines):
            line = raw.strip()
            if not line or not line.startswith("#"):
                continue
            m = re.match(r"^(#+)\s+(.*)$", line)
            if not m:
                continue
            level = len(m.group(1))
            if level > 2:
                continue
            name = m.group(2).strip().rstrip("# ")
            if not name:
                continue
            slug = _slugify(name)
            if slug in topics:
                sp = set(topics[slug].get("source_paths", []))  # type: ignore[assignment]
                sp.add(str(path))
                topics[slug]["source_paths"] = sorted(sp)
                continue
            desc = ""
            for j in range(i + 1, min(i + 6, len(lines))):
                nxt = lines[j].strip()
                if nxt and not nxt.startswith("#"):
                    desc = nxt
                    break
            topics[slug] = {
                "id": slug,
                "name": name,
                "description": desc,
                "source_paths": [str(path)],
                "created_at": "",
            }
    heuristic = list(topics.values())
    if not use_ai:
        return heuristic
    ai_topics = ai_extract_topics(sources, k=k, client=client, prompt=ai_prompt, seed=seed)
    merged: Dict[str, Dict[str, object]] = {t["id"]: t for t in heuristic}  # type: ignore[index]
    for t in ai_topics:
        slug = str(t.get("id") or _slugify(str(t.get("name", ""))))
        if slug in merged:
            if t.get("name") and len(str(t.get("name"))) > len(str(merged[slug].get("name", ""))):
                merged[slug]["name"] = t.get("name")
            sp = set(merged[slug].get("source_paths", [])) | set(t.get("source_paths", []) or [])  # type: ignore[arg-type]
            merged[slug]["source_paths"] = sorted(sp)
        else:
            merged[slug] = {
                "id": slug,
                "name": t.get("name") or slug,
                "description": t.get("description", ""),
                "source_paths": t.get("source_paths", []) or [],
                "created_at": "",
            }
    return list(merged.values())


# -----------------------------
# Question validation
# -----------------------------


def validate_mcq(q: Dict[str, object]) -> None:
    """Validate an MCQ question dict.

    Required keys: id, topic_id, type=='mcq', stem, choices(list of {key,text}), answer(str), explanation(str)
    - choices keys must be unique, 2–6 options, keys like 'A'..'F'
    - exactly one answer and it must be among choices
    Raises ValueError with actionable messages when invalid.
    """
    if not isinstance(q, dict):
        raise ValueError("question must be a dict")
    if q.get("type") != "mcq":
        raise ValueError("type must be 'mcq'")
    choices = q.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a non-empty list")
    keys = [str(c.get("key", "")).upper() for c in choices]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate choice keys detected")
    if not all(k and len(k) == 1 and k.isalpha() for k in keys):
        raise ValueError("choice keys must be single letters (A..Z)")
    if not all(str(c.get("text", "")).strip() for c in choices):
        raise ValueError("choice text must be non-empty")
    ans = q.get("answer")
    if isinstance(ans, list):
        raise ValueError("MCQ requires exactly one answer, not multiple")
    if not isinstance(ans, str) or not ans:
        raise ValueError("answer is required")
    if ans.upper() not in set(keys):
        raise ValueError("answer must match one of the choice keys")


# -----------------------------
# Selection strategies
# -----------------------------


def _group_by_topic(bank: Sequence[Dict[str, object]]):
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for q in bank:
        groups[str(q.get("topic_id", ""))].append(q)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: str(x.get("id", "")))
    return dict(groups)


def select_questions(
    bank: Sequence[Dict[str, object]],
    *,
    strategy: str = "balanced",
    num: int = 10,
    per_topic_stats: Optional[Dict[str, Dict[str, int]]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Select questions from a bank using a strategy.

    - balanced: round-robin across topics before repeats
    - random: uniform random over the bank
    - weakness: weighted toward topics with lower accuracy (1 - correctness)
    Deterministic when seed is provided.
    """
    if num <= 0 or not bank:
        return []
    rng = random.Random(seed)
    if strategy == "random":
        return [rng.choice(bank) for _ in range(num)]

    groups = _group_by_topic(bank)
    topics = sorted(groups.keys())
    if not topics:
        return []

    if strategy == "weakness":
        weights: List[float] = []
        for t in topics:
            s = per_topic_stats.get(t, {"asked": 0, "correct": 0}) if per_topic_stats else {"asked": 0, "correct": 0}
            asked = max(1, int(s.get("asked", 0)))
            correct = max(0, int(s.get("correct", 0)))
            acc = correct / asked
            w = max(0.05, 1.0 - acc)
            weights.append(w)
        selected: List[Dict[str, object]] = []
        for _ in range(num):
            t = rng.choices(topics, weights=weights, k=1)[0]
            selected.append(rng.choice(groups[t]))
        return selected

    # balanced
    selected: List[Dict[str, object]] = []
    topic_indices = {t: 0 for t in topics}
    i = 0
    while len(selected) < num:
        t = topics[i % len(topics)]
        g = groups[t]
        idx = topic_indices[t]
        if idx >= len(g):
            idx = rng.randrange(0, len(g)) if g else 0
        selected.append(g[idx])
        topic_indices[t] = (idx + 1) % max(1, len(g))
        i += 1
    return selected[:num]


# -----------------------------
# Summary aggregation
# -----------------------------


def aggregate_summary(responses: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total = len(responses)
    correct = sum(1 for r in responses if bool(r.get("correct")))
    per_topic: Dict[str, Dict[str, int]] = defaultdict(lambda: {"asked": 0, "correct": 0})
    for r in responses:
        t = str(r.get("topic_id", ""))
        per_topic[t]["asked"] += 1
        if r.get("correct"):
            per_topic[t]["correct"] += 1
    accuracy = (correct / total) if total else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy, "per_topic": dict(per_topic)}


# -----------------------------
# JSONL utilities
# -----------------------------


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


# -----------------------------
# CLI parser
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quizzer",
        description="Interactive quiz tool for Markdown sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp_init = sub.add_parser("init", help="Create quizzer.toml template for a quiz")
    sp_init.add_argument("name")

    sp_topics = sub.add_parser("topics", help="Topic-related commands")
    topics_sub = sp_topics.add_subparsers(dest="action", required=True)
    sp_t_gen = topics_sub.add_parser("generate", help="Extract and persist topics")
    sp_t_gen.add_argument("name")
    sp_t_gen.add_argument("--limit", type=int)
    sp_t_gen.add_argument("--extensions", nargs="+", default=["md", "markdown"], help="File extensions to include for discovery")
    sp_t_gen.add_argument("--level-limit", type=int, default=0, help="Directory depth limit (0 = no limit)")
    sp_t_gen.add_argument("--use-ai", action="store_true", help="Use AI to assist topic extraction")
    sp_t_list = topics_sub.add_parser("list", help="List topics")
    sp_t_list.add_argument("name")
    sp_t_list.add_argument("--filter")

    sp_q = sub.add_parser("questions", help="Question-related commands")
    q_sub = sp_q.add_subparsers(dest="action", required=True)
    sp_q_gen = q_sub.add_parser("generate", help="Generate questions")
    sp_q_gen.add_argument("name")
    sp_q_gen.add_argument("--per-topic", type=int)
    sp_q_list = q_sub.add_parser("list", help="List questions")
    sp_q_list.add_argument("name")
    sp_q_list.add_argument("--topics", nargs="*")

    sp_start = sub.add_parser("start", help="Start a quiz session")
    sp_start.add_argument("name")
    sp_start.add_argument("--num", type=int, default=10)
    sp_start.add_argument("--mix", choices=["balanced", "random", "weakness"], default="balanced")
    sp_start.add_argument("--resume", action="store_true")
    sp_start.add_argument("--shuffle", action="store_true")
    sp_start.add_argument("--explain", dest="explain", action="store_true")
    sp_start.add_argument("--no-explain", dest="explain", action="store_false")
    sp_start.set_defaults(explain=True)

    sp_rev = sub.add_parser("review", help="Re-quiz wrong or weak topics from a session")
    sp_rev.add_argument("name")
    sp_rev.add_argument("--session")
    sp_rep = sub.add_parser("report", help="Show session summary")
    sp_rep.add_argument("name")
    sp_rep.add_argument("--session")
    return p


def ai_extract_topics(
    sources: Iterable[Tuple[Path, str]],
    *,
    k: Optional[int] = None,
    client: object = None,
    prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    seed: Optional[int] = None,
    source_max_lines: Optional[int] = 20,
    source_max_lines_chars: Optional[int] = 400,
) -> List[Dict[str, object]]:
    """Suggest topics from raw text using an AI model.

    Will parse model output (JSON array of names or objects) and
    return a list of topic dicts with id/name/description/source_paths.
    """
    # Acquire client if not provided
    if client is None and load_client is not None:  # pragma: no cover - exercised via integration
        try:
            client = load_client()
        except Exception:
            client = None
    if client is None:
        return []

    # Build a compact prompt using a small snippet of each source
    parts: List[str] = []
    for path, text in sources:
        snippet = (text or "").strip().splitlines()
        snippet = [ln for ln in snippet if ln.strip()][:source_max_lines]
        joined = "\n".join(snippet[:source_max_lines_chars])
        parts.append(f"File: {path.name}\n{joined}\n")
    user_prompt = (
        (prompt or "Suggest concise study topics from the following notes. Output JSON array of objects with name and optional description.")
        + "\n\n" + "\n\n".join(parts)
    )

    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": "You extract clean, deduplicated topic names from text."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        content = (resp.choices[0].message.content or "").strip()  # type: ignore[index]
    except Exception:
        return []

    # Parse JSON output: accept ["Topic", ...] or [{"name": "...", "description": "..."}, ...]
    suggestions: List[Dict[str, object]] = []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data[: k or len(data)]:
                if isinstance(item, str):
                    nm = item.strip()
                    if not nm:
                        continue
                    suggestions.append({
                        "id": _slugify(nm),
                        "name": nm,
                        "description": "",
                        "source_paths": [],
                        "created_at": "",
                    })
                elif isinstance(item, dict):
                    nm = str(item.get("name", "")).strip()
                    if not nm:
                        continue
                    suggestions.append({
                        "id": _slugify(nm),
                        "name": nm,
                        "description": str(item.get("description", "")),
                        "source_paths": item.get("source_paths", []) if isinstance(item.get("source_paths", []), list) else [],
                        "created_at": "",
                    })
    except Exception:
        # If the model didn't return JSON, ignore AI suggestions
        return []

    return suggestions


__all__ = [
    "iter_quiz_files",
    "extract_topics",
    "validate_mcq",
    "select_questions",
    "aggregate_summary",
    "read_jsonl",
    "write_jsonl",
    "build_arg_parser",
    "ai_extract_topics",
]


# -----------------------------
# CLI runtime
# -----------------------------


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback for <3.11 if tomli is installed
        try:
            import tomli as tomllib  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("TOML support not available. Use Python 3.11+ or install 'tomli'.") from exc
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _find_config(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    p = Path("quizzer.toml").resolve()
    if p.exists():
        return p
    # Fallback to app defaults (not provided yet)
    return None


def _get_quiz_section(cfg: dict, name: str) -> dict:
    root = cfg.get("quiz") or {}
    sec = root.get(name)
    if not isinstance(sec, dict):
        raise KeyError(f"Quiz section not found: [quiz.{name}]")
    return sec


def _out_dir_for(name: str, cfg: Optional[dict]) -> Path:
    d = None
    if cfg:
        st = cfg.get("storage") or {}
        d = st.get("out_dir") if isinstance(st, dict) else None
    base = Path(d.replace("<name>", name)) if isinstance(d, str) else Path(".quizzer") / name
    return base.resolve()


def _read_files(files: Sequence[Path]) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        out.append((p, text))
    return out


def _cmd_init(args: argparse.Namespace) -> int:
    name: str = args.name
    path = Path("quizzer.toml").resolve()
    if path.exists():
        print(f"quizzer.toml already exists at {path}")
        return 0
    sample = (
        "# Quizzer configuration\n"
        "# Define quizzes under [quiz.<name>] sections\n\n"
        f"[quiz.{name}]\n"
        "# One or more files or directories (Markdown)\n"
        "sources = [\"./materials\"]\n"
        "types = [\"mcq\"]\n"
        "per_topic = 3\n"
        "ensure_coverage = true\n\n"
        "[storage]\n"
        "# Artifacts directory; <name> will be replaced with quiz name\n"
        "out_dir = \".quizzer/<name>\"\n\n"
        "[ai]\n"
        "model = \"gpt-4o-mini\"\n"
        "temperature = 0.2\n"
        "max_tokens = 600\n"
    )
    path.write_text(sample, encoding="utf-8")
    print(f"Created template {path}")
    return 0


def _cmd_topics_generate(args: argparse.Namespace) -> int:
    cfg_path = _find_config(getattr(args, "config", None))
    if not cfg_path:
        print("Error: quizzer.toml not found. Run 'quizzer init <name>' first.")
        return 2
    cfg = _load_toml(cfg_path)
    try:
        section = _get_quiz_section(cfg, args.name)
    except KeyError as exc:
        print(f"Error: {exc}")
        return 2
    sources = section.get("sources") or []
    if not sources:
        print(f"Error: [quiz.{args.name}] must define 'sources' list")
        return 2
    input_paths = [ Path(s).expanduser().resolve() for s in sources ]
    files = iter_quiz_files(input_paths, extensions=args.extensions, level_limit=int(args.level_limit))
    if not files:
        print("No matching Markdown files found.")
        return 1
    pairs = _read_files(files)
    topics = extract_topics(pairs, use_ai=bool(args.use_ai))
    out_dir = _out_dir_for(args.name, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    topics_path = out_dir / "topics.jsonl"
    write_jsonl(topics_path, topics)
    print(f"Wrote {len(topics)} topic(s) -> {topics_path}")
    return 0


def _cmd_topics_list(args: argparse.Namespace) -> int:
    cfg_path = _find_config(getattr(args, "config", None))
    if not cfg_path:
        print("Error: quizzer.toml not found.")
        return 2
    cfg = _load_toml(cfg_path)
    out_dir = _out_dir_for(args.name, cfg)
    topics_path = out_dir / "topics.jsonl"
    if not topics_path.exists():
        print(f"No topics found at {topics_path}. Run 'quizzer topics generate {args.name}'.")
        return 1
    topics = read_jsonl(topics_path)
    filt = (args.filter or "").lower().strip()
    shown = 0
    for t in topics:
        name = str(t.get("name", ""))
        if not filt or filt in name.lower():
            print(f"- {name}")
            shown += 1
    if shown == 0:
        print("No topics match filter.")
        return 1
    return 0


def _cmd_not_implemented(label: str) -> int:
    print(f"{label} is not implemented yet. Stay tuned.")
    return 2


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        code = _cmd_init(args)
    elif args.command == "topics" and args.action == "generate":
        code = _cmd_topics_generate(args)
    elif args.command == "topics" and args.action == "list":
        code = _cmd_topics_list(args)
    elif args.command == "questions":
        code = _cmd_not_implemented("questions subcommands")
    elif args.command == "start":
        code = _cmd_not_implemented("start")
    elif args.command == "review":
        code = _cmd_not_implemented("review")
    elif args.command == "report":
        code = _cmd_not_implemented("report")
    else:  # pragma: no cover - fallback guard
        parser.print_help()
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":  # pragma: no cover
    main()

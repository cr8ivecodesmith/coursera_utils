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


def extract_topics(sources: Iterable[Tuple[Path, str]]) -> List[Dict[str, object]]:
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
    return list(topics.values())


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


__all__ = [
    "iter_quiz_files",
    "extract_topics",
    "validate_mcq",
    "select_questions",
    "aggregate_summary",
    "read_jsonl",
    "write_jsonl",
    "build_arg_parser",
]

import re
import json
import random

from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Sequence, Tuple, Iterable, Dict

from ..utils import load_client, _slugify


def _gather_topic_context(
    topic: Dict[str, object], max_chars: int = 4000
) -> str:
    """Read source_paths for a topic and extract relevant snippets.

    Heuristics:
    - If headings matching topic name exist, include lines until next heading
    - Else include lines containing the topic name
    - Concatenate across files up to max_chars
    """
    name = str(topic.get("name", "")).strip()
    sources = topic.get("source_paths", []) or []
    if not isinstance(sources, list):
        return ""
    parts: List[str] = []
    pattern = re.compile(r"^\s*#+\s+(.+)$")
    for s in sources:
        p = Path(s)
        if not p.exists() or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = text.splitlines()
        collected: List[str] = []
        capture = False
        for ln in lines:
            m = pattern.match(ln)
            if m:
                heading = m.group(1).strip().strip("# ")
                # Start capture when heading matches name (case-insensitive)
                capture = heading.lower() == name.lower() if name else False
                if capture:
                    collected.append(ln)
                continue
            if capture:
                # Stop when next heading encountered
                if ln.strip().startswith("#"):
                    capture = False
                else:
                    collected.append(ln)
        # Fallback: lines that contain topic name
        if not collected and name:
            for ln in lines:
                if name.lower() in ln.lower():
                    collected.append(ln)
        if collected:
            snippet = "\n".join(collected).strip()
            if snippet:
                parts.append(f"From {p.name}:\n{snippet}")
        if sum(len(part) for part in parts) > max_chars:
            break
    out = "\n\n".join(parts)
    return out[:max_chars]


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
            s = (
                per_topic_stats.get(t, {"asked": 0, "correct": 0})
                if per_topic_stats
                else {"asked": 0, "correct": 0}
            )
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


def generate_questions(
    topics: Sequence[Dict[str, object]],
    *,
    per_topic: int = 3,
    qtype: str = "mcq",
    client: object = None,
    ensure_coverage: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Generate a question bank from topics.

    Call AI generation per topic and concatenate results. For now only MCQ is
    supported.
    """
    if qtype != "mcq" or per_topic <= 0:
        return []
    bank: List[Dict[str, object]] = []
    for t in topics:
        context = _gather_topic_context(t)
        qs = ai_generate_mcqs_for_topic(
            t, n=per_topic, client=client, seed=seed, context=context
        )
        if ensure_coverage and not qs:
            topic_label = t.get("name", t.get("id", "this topic"))
            stem = f"Which of the following relates to {topic_label}?"
            placeholder = {
                "id": f"{t.get('id')}-ph-1",
                "topic_id": str(t.get("id")),
                "type": "mcq",
                "stem": stem,
                "choices": [
                    {"key": "A", "text": str(t.get("name", "Concept"))},
                    {"key": "B", "text": "None of the above"},
                ],
                "answer": "A",
                "explanation": "",
            }
            try:
                validate_mcq(placeholder)
                qs = [placeholder]
            except Exception:
                qs = []
        bank.extend(qs)
    return bank


def validate_mcq(q: Dict[str, object]) -> None:
    """Validate an MCQ question dict.

    Required keys: id, topic_id, type=='mcq', stem, choices (list of
    {key,text}), answer (str), explanation (str).
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


def ai_generate_mcqs_for_topic(
    topic: Dict[str, object],
    n: int = 3,
    *,
    client: object = None,
    prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    seed: Optional[int] = None,
    context: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Generate n MCQ questions for a topic using an AI model.

    Return up to ``n`` validated MCQ dicts with normalized shape. Invalid items
    are skipped.
    """
    if n <= 0:
        return []
    if client is None and load_client is not None:  # pragma: no cover
        try:
            client = load_client()
        except Exception:
            client = None
    if client is None:
        return []

    topic_id = str(
        topic.get("id")
        or _slugify(str(topic.get("name", "topic")))
    )
    topic_name = str(topic.get("name") or topic_id)
    sys_prompt = "You generate high-quality multiple-choice study questions."
    ctx_block = (
        ("\n\nContext (from source materials):\n" + context.strip())
        if isinstance(context, str) and context.strip()
        else ""
    )
    base_instructions = (
        prompt
        or "Create concise multiple-choice questions covering the topic. "
        "Output a JSON array of objects."
    )
    schema_line = (
        '{"stem": str, "choices": [str or {"key": "A", "text": str}], '
        '"answer": str, "explanation": str}\n'
    )
    constraints = (
        "Constraints: single correct answer; plausible distractors; avoid "
        "ambiguity; keep stems under 200 chars."
    )
    user_prompt = (
        f"{base_instructions}\n\nSchema:\n{schema_line}"
        f"Topic: {topic_name}\n"
        f"Count: {n}\n"
        f"{constraints}"
        f"{ctx_block}"
    )
    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        raw_content = resp.choices[0].message.content  # type: ignore[index]
        content = (raw_content or "").strip().replace("\n", "")
    except Exception:
        return []

    items: List[Dict[str, object]] = []
    try:
        pat = re.compile(r"^(```json)(.+)(```)$")
        _data = pat.match(content)
        if _data:
            data = json.loads(_data.group(2))
        else:
            data = json.loads(content)
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    def normalize_choice_list(raw_choices) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if isinstance(raw_choices, list):
            for i, ch in enumerate(raw_choices):
                if isinstance(ch, dict):
                    key = str(ch.get("key", "")).strip() or chr(ord("A") + i)
                    text = str(ch.get("text", "")).strip()
                else:
                    key = chr(ord("A") + i)
                    text = str(ch).strip()
                if not text:
                    continue
                out.append({"key": key.upper()[:1], "text": text})
        return out

    for rec in data[:n]:
        if not isinstance(rec, dict):
            continue
        stem = str(rec.get("stem", "")).strip()
        if not stem:
            continue
        choices = normalize_choice_list(rec.get("choices", []))
        ans = rec.get("answer")
        if isinstance(ans, int) and 0 <= ans < len(choices):
            answer = choices[ans]["key"]
        else:
            answer = str(ans or "").strip().upper()
            if answer and answer not in {c["key"] for c in choices}:
                for c in choices:
                    if answer.lower() == c["text"].lower():
                        answer = c["key"]
                        break
        explanation = str(rec.get("explanation", "")).strip()
        generated_id = (
            f"{topic_id}-{random.Random(seed).randint(10000, 99999)}-"
            f"{len(items)}"
        )
        q = {
            "id": rec.get("id") or generated_id,
            "topic_id": topic_id,
            "type": "mcq",
            "stem": stem,
            "choices": choices,
            "answer": answer,
            "explanation": explanation,
        }
        try:
            validate_mcq(q)
        except Exception:
            continue
        items.append(q)
    return items[:n]


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
    if (
        client is None and load_client is not None
    ):  # pragma: no cover - exercised via integration
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
    base_prompt = (
        prompt
        or "Suggest concise study topics from the following notes. "
        "Output a JSON array of objects with name and optional description."
    )
    user_prompt = base_prompt + "\n\n" + "\n\n".join(parts)

    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract clean, deduplicated topic names."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        raw_content = resp.choices[0].message.content  # type: ignore[index]
        content = (raw_content or "").strip()
    except Exception:
        return []

    # Parse JSON output: accept ["Topic", ...] or
    # [{"name": "...", "description": "..."}, ...]
    suggestions: List[Dict[str, object]] = []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data[: k or len(data)]:
                if isinstance(item, str):
                    nm = item.strip()
                    if not nm:
                        continue
                    suggestions.append(
                        {
                            "id": _slugify(nm),
                            "name": nm,
                            "description": "",
                            "source_paths": [],
                            "created_at": "",
                        }
                    )
                elif isinstance(item, dict):
                    nm = str(item.get("name", "")).strip()
                    if not nm:
                        continue
                    suggestions.append(
                        {
                            "id": _slugify(nm),
                            "name": nm,
                            "description": str(item.get("description", "")),
                            "source_paths": item.get("source_paths", [])
                            if isinstance(item.get("source_paths", []), list)
                            else [],
                            "created_at": "",
                        }
                    )
    except Exception:
        # If the model didn't return JSON, ignore AI suggestions
        return []

    return suggestions


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
                existing_paths = topics[slug].get("source_paths", [])
                path_set = (
                    {str(p) for p in existing_paths}
                    if isinstance(existing_paths, list)
                    else set()
                )
                path_set.add(str(path))
                topics[slug]["source_paths"] = sorted(path_set)
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
    ai_topics = ai_extract_topics(
        sources, k=k, client=client, prompt=ai_prompt, seed=seed
    )
    merged: Dict[str, Dict[str, object]] = {}
    for item in heuristic:
        merged[str(item["id"])] = item
    for t in ai_topics:
        slug = str(t.get("id") or _slugify(str(t.get("name", ""))))
        if slug in merged:
            if t.get("name") and len(str(t.get("name"))) > len(
                str(merged[slug].get("name", ""))
            ):
                merged[slug]["name"] = t.get("name")
            sp = set(merged[slug].get("source_paths", [])) | set(
                t.get("source_paths", []) or []
            )  # type: ignore[arg-type]
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

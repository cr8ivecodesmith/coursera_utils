from pathlib import Path
import argparse
import pytest


def test_iter_quiz_files_discovers_markdown(tmp_path: Path):
    # Layout
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.md").write_text("# X\n")
    (tmp_path / "a" / "y.markdown").write_text("# Y\n")
    (tmp_path / "a" / "z.txt").write_text("not md\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c").mkdir()
    (tmp_path / "b" / "c" / "deep.md").write_text("# Deep\n")

    from study_utils import quizzer as qz

    # level_limit=1 should not include b/c/deep.md when starting at b
    files = qz.iter_quiz_files(
        [tmp_path / "a", tmp_path / "b"],
        extensions=("md", "markdown"),
        level_limit=1,
    )
    rel = sorted([p.relative_to(tmp_path).as_posix() for p in files])
    assert rel == ["a/x.md", "a/y.markdown"]

    # Unlimited depth will include all
    files2 = qz.iter_quiz_files(
        [tmp_path], extensions=("md", "markdown"), level_limit=0
    )
    rel2 = sorted([p.relative_to(tmp_path).as_posix() for p in files2])
    assert rel2 == ["a/x.md", "a/y.markdown", "b/c/deep.md"]


def test_extract_topics_simple_headings(tmp_path: Path):
    md = """
    # Intro
    Some context
    ## Basics
    - Bullet 1
    ## Basics
    ### Details
    """
    f = tmp_path / "notes.md"
    f.write_text(md)

    from study_utils import quizzer as qz

    topics = qz.extract_topics([(f, md)])
    names = {t["name"] for t in topics}
    ids = {t["id"] for t in topics}
    # Dedup Basics; include Intro and Basics; Details may be filtered (too
    # narrow)
    assert "Intro" in names
    assert "Basics" in names
    assert len(names) == len(ids)  # unique slugs
    # sources tracked
    assert any(str(f) in t.get("source_paths", []) for t in topics)


def test_validate_mcq_happy_and_errors():
    from study_utils import quizzer as qz

    ok = {
        "id": "q1",
        "topic_id": "intro",
        "type": "mcq",
        "stem": "What is 2+2?",
        "choices": [
            {"key": "A", "text": "3"},
            {"key": "B", "text": "4"},
            {"key": "C", "text": "5"},
            {"key": "D", "text": "22"},
        ],
        "answer": "B",
        "explanation": "2+2=4",
    }
    # Should not raise
    try:
        qz.validate_mcq(ok)
    except (
        Exception
    ) as e:  # pragma: no cover - this should not happen once implemented
        pytest.fail(f"validate_mcq raised unexpectedly: {e}")

    bad_dup_keys = {
        **ok,
        "choices": [{"key": "A", "text": "3"}, {"key": "A", "text": "4"}],
    }
    with pytest.raises(ValueError) as e1:
        qz.validate_mcq(bad_dup_keys)
    assert "duplicate" in str(e1.value).lower()

    bad_multi_answers = {**ok, "answer": ["A", "B"]}
    with pytest.raises(ValueError) as e2:
        qz.validate_mcq(bad_multi_answers)
    assert (
        "single" in str(e2.value).lower()
        or "exactly one" in str(e2.value).lower()
    )

    missing_answer = {**ok}
    missing_answer.pop("answer")
    with pytest.raises(ValueError) as e3:
        qz.validate_mcq(missing_answer)
    assert "answer" in str(e3.value).lower()


def _mk_q(topic: str, idx: int) -> dict:
    return {
        "id": f"{topic}-{idx}",
        "topic_id": topic,
        "type": "mcq",
        "stem": f"Q {topic} #{idx}",
        "choices": [
            {"key": "A", "text": "x"},
            {"key": "B", "text": "y"},
            {"key": "C", "text": "z"},
            {"key": "D", "text": "w"},
        ],
        "answer": "A",
        "explanation": "",
    }


def test_select_questions_balanced_and_seed():
    from study_utils import quizzer as qz

    bank = [_mk_q("t1", i) for i in range(3)] + [
        _mk_q("t2", i) for i in range(3)
    ]
    selected = qz.select_questions(
        bank, strategy="balanced", num=4, per_topic_stats=None, seed=42
    )
    # First two should cover both topics before repeating
    seen = [q["topic_id"] for q in selected[:2]]
    assert set(seen) == {"t1", "t2"}

    # Seeded determinism
    s1 = qz.select_questions(bank, strategy="random", num=5, seed=123)
    s2 = qz.select_questions(bank, strategy="random", num=5, seed=123)
    assert [q["id"] for q in s1] == [q["id"] for q in s2]


def test_select_questions_weakness_weighting():
    from study_utils import quizzer as qz

    bank = [_mk_q("t1", i) for i in range(10)] + [
        _mk_q("t2", i) for i in range(10)
    ]
    stats = {
        "t1": {"asked": 10, "correct": 9},
        "t2": {"asked": 10, "correct": 2},
    }
    sel = qz.select_questions(
        bank, strategy="weakness", num=10, per_topic_stats=stats, seed=7
    )
    # Expect t2 to appear more often than t1
    from collections import Counter

    c = Counter(q["topic_id"] for q in sel)
    assert c["t2"] > c["t1"]


def test_aggregate_summary():
    from study_utils import quizzer as qz

    resp = [
        {
            "question_id": "t1-1",
            "topic_id": "t1",
            "given": "A",
            "correct": True,
            "duration_sec": 3.0,
        },
        {
            "question_id": "t1-2",
            "topic_id": "t1",
            "given": "B",
            "correct": False,
            "duration_sec": 4.0,
        },
        {
            "question_id": "t2-1",
            "topic_id": "t2",
            "given": "A",
            "correct": True,
            "duration_sec": 2.0,
        },
    ]
    summary = qz.aggregate_summary(resp)
    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["per_topic"]["t1"]["asked"] == 2
    assert summary["per_topic"]["t1"]["correct"] == 1


def test_jsonl_round_trip(tmp_path: Path):
    from study_utils import quizzer as qz

    topics = [
        {
            "id": "intro",
            "name": "Intro",
            "description": "...",
            "source_paths": ["/x"],
            "created_at": "2020-01-01T00:00:00Z",
        },
        {
            "id": "basics",
            "name": "Basics",
            "description": "...",
            "source_paths": ["/y"],
            "created_at": "2020-01-01T00:00:00Z",
        },
    ]
    p = tmp_path / "topics.jsonl"
    qz.write_jsonl(p, topics)
    back = qz.read_jsonl(p)
    assert back == topics


def test_cli_parse_subcommands():
    from study_utils import quizzer as qz

    parser: argparse.ArgumentParser = qz.build_arg_parser()

    ns = parser.parse_args(["init", "myquiz"])
    assert ns.command == "init" and ns.name == "myquiz"

    ns = parser.parse_args(["topics", "generate", "myquiz", "--limit", "5"])
    assert (
        ns.command == "topics"
        and ns.action == "generate"
        and ns.name == "myquiz"
        and ns.limit == 5
    )

    ns = parser.parse_args(
        ["start", "myquiz", "--num", "10", "--mix", "balanced"]
    )
    assert ns.command == "start" and ns.num == 10 and ns.mix == "balanced"

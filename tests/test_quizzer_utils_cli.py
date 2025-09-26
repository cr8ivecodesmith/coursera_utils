from __future__ import annotations

import json
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from pathlib import Path
import builtins

import pytest

from study_utils.quizzer import utils as q_utils
from study_utils.quizzer import _main as q_cli


# ---------------------- utils._find_config / _get_quiz_section ----------------------


def test_find_config_explicit_and_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    explicit = tmp_path / "quizzer.toml"
    explicit.write_text("[quiz.example]\n", encoding="utf-8")
    assert q_utils._find_config(str(explicit)) == explicit.resolve()

    missing = tmp_path / "missing.toml"
    assert q_utils._find_config(str(missing)) is None

    monkeypatch.chdir(tmp_path)
    default = q_utils._find_config(None)
    assert default == explicit.resolve()

    explicit.unlink()
    assert q_utils._find_config(None) is None


def test_get_quiz_section_success_and_missing() -> None:
    cfg = {"quiz": {"demo": {"sources": ["/tmp"]}}}
    section = q_utils._get_quiz_section(cfg, "demo")
    assert section["sources"] == ["/tmp"]
    with pytest.raises(KeyError):
        q_utils._get_quiz_section(cfg, "missing")


# ---------------------- utils._load_toml ----------------------


def test_load_toml_with_tomllib(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[quiz.demo]\nsources=['a']\n", encoding="utf-8")
    data = q_utils._load_toml(path)
    assert data["quiz"]["demo"]["sources"] == ["a"]


def test_load_toml_falls_back_to_tomli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[doc]\nkey='value'\n", encoding="utf-8")
    real_import = builtins.__import__

    class StubTomli(ModuleType):
        @staticmethod
        def load(handle):
            return {"doc": {"key": "value"}}

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tomllib":
            raise ImportError("tomllib unavailable")
        if name == "tomli":
            return StubTomli("tomli")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    data = q_utils._load_toml(path)
    assert data["doc"]["key"] == "value"


# ---------------------- utils._read_files / iter_quiz_files / jsonl helpers ----------------------


def test_read_files_handles_read_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    files = [tmp_path / "ok.txt", tmp_path / "bad.txt"]
    for p in files:
        p.write_text("content", encoding="utf-8")

    def fake_read(path: Path) -> str:
        if path.name == "bad.txt":
            raise OSError("boom")
        return "ok"

    monkeypatch.setattr(q_utils, "read_text_file", fake_read)
    result = q_utils._read_files(files)
    assert result == [(files[0], "ok"), (files[1], "")]


def test_slugify_and_iter_quiz_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert q_utils._slugify(" Intro  Topic ! ") == "intro-topic"
    with pytest.raises(ValueError):
        q_utils.iter_quiz_files([], level_limit=-1)

    missing = tmp_path / "missing.md"
    files = q_utils.iter_quiz_files([missing], level_limit=0)
    assert files == []

    src = tmp_path / "src"
    src.mkdir()
    a = src / "a.md"
    a.write_text("# A", encoding="utf-8")
    b = src / "b.markdown"
    b.write_text("# B", encoding="utf-8")
    found = q_utils.iter_quiz_files([src], level_limit=0)
    assert found == [a, b]


def test_read_write_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    records = [{"id": 1}, {"id": 2}]
    q_utils.write_jsonl(path, records)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    back = q_utils.read_jsonl(path)
    assert back == records


# ---------------------- CLI helpers and commands ----------------------


def test_out_dir_for_defaults_and_template(tmp_path: Path) -> None:
    cfg = {"storage": {"out_dir": ".quizzer/<name>"}}
    resolved = q_cli._out_dir_for("demo", cfg)
    assert resolved.as_posix().endswith(".quizzer/demo")

    resolved_default = q_cli._out_dir_for("demo", None)
    assert resolved_default.as_posix().endswith(".quizzer/demo")


def test_cmd_init_creates_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    args = Namespace(name="demo")
    code = q_cli._cmd_init(args)
    assert code == 0
    output = capsys.readouterr().out
    assert "Created template" in output
    assert (tmp_path / "quizzer.toml").exists()

    code_again = q_cli._cmd_init(args)
    assert code_again == 0
    assert "already exists" in capsys.readouterr().out


def test_cmd_topics_generate_error_paths(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    args = Namespace(name="demo", config=None, extensions=["md"], level_limit=0, use_ai=False)

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: None)
    assert q_cli._cmd_topics_generate(args) == 2
    assert "not found" in capsys.readouterr().out

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: Path("cfg"))
    monkeypatch.setattr(q_cli, "_load_toml", lambda *_: {})
    assert q_cli._cmd_topics_generate(args) == 2
    assert "Quiz section not found" in capsys.readouterr().out

    monkeypatch.setattr(q_cli, "_load_toml", lambda *_: {"quiz": {"demo": {}}})
    monkeypatch.setattr(q_cli, "_get_quiz_section", lambda *_: {})
    assert q_cli._cmd_topics_generate(args) == 2
    assert "must define 'sources'" in capsys.readouterr().out

    monkeypatch.setattr(q_cli, "_get_quiz_section", lambda *_: {"sources": ["/missing"]})
    monkeypatch.setattr(q_cli, "iter_quiz_files", lambda *a, **kw: [])
    assert q_cli._cmd_topics_generate(args) == 1
    assert "No matching" in capsys.readouterr().out


def test_cmd_topics_generate_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    materials = tmp_path / "materials"
    materials.mkdir()
    doc = materials / "notes.md"
    doc.write_text("# Intro\n", encoding="utf-8")

    cfg_path = tmp_path / "quizzer.toml"
    cfg_path.write_text(
        "[quiz.demo]\n" "sources = ['materials']\n" "[storage]\n" "out_dir = '.quizzer/<name>'\n",
        encoding="utf-8",
    )

    topics_out = []

    def fake_extract(pairs, use_ai):
        topics_out.extend(pairs)
        return [{"id": "intro", "name": "Intro"}]

    monkeypatch.setattr(q_cli, "extract_topics", fake_extract)
    args = Namespace(
        name="demo",
        config=None,
        extensions=["md"],
        level_limit=0,
        use_ai=False,
    )
    code = q_cli._cmd_topics_generate(args)
    assert code == 0
    out_dir = tmp_path / ".quizzer" / "demo"
    assert (out_dir / "topics.jsonl").exists()
    assert "Wrote" in capsys.readouterr().out


def test_cmd_topics_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "quizzer.toml"
    cfg_path.write_text("[quiz.demo]\n", encoding="utf-8")
    args = Namespace(name="demo", config=None, filter=None)

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: None)
    assert q_cli._cmd_topics_list(args) == 2

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: cfg_path)
    monkeypatch.setattr(q_cli, "_load_toml", lambda *_: {})
    out_dir = tmp_path / ".quizzer" / "demo"
    assert q_cli._cmd_topics_list(args) == 1
    assert "No topics found" in capsys.readouterr().out

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "topics.jsonl").write_text(json.dumps({"name": "Intro"}) + "\n", encoding="utf-8")
    assert q_cli._cmd_topics_list(args) == 0
    assert "Intro" in capsys.readouterr().out

    args_filter = Namespace(name="demo", config=None, filter="missing")
    assert q_cli._cmd_topics_list(args_filter) == 1
    assert "No topics match" in capsys.readouterr().out


def test_cmd_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    assert q_cli._cmd_not_implemented("review") == 2
    assert "not implemented" in capsys.readouterr().out


def test_cmd_questions_generate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "quizzer.toml"
    cfg_path.write_text(
        "[quiz.demo]\nper_topic = 2\nensure_coverage = true\n[storage]\nout_dir = '.quizzer/<name>'\n",
        encoding="utf-8",
    )
    args = Namespace(name="demo", config=None, per_topic=None, ensure_coverage=True)

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: None)
    assert q_cli._cmd_questions_generate(args) == 2

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: cfg_path)
    monkeypatch.setattr(q_cli, "_load_toml", lambda *_: {})
    assert q_cli._cmd_questions_generate(args) == 2
    assert "Quiz section not found" in capsys.readouterr().out

    monkeypatch.setattr(q_cli, "_load_toml", lambda *_: {"quiz": {"demo": {"per_topic": 2, "ensure_coverage": True}}})
    assert q_cli._cmd_questions_generate(args) == 1
    assert "No topics found" in capsys.readouterr().out

    topics_dir = tmp_path / ".quizzer" / "demo"
    topics_dir.mkdir(parents=True, exist_ok=True)
    topics_file = topics_dir / "topics.jsonl"
    topics_file.write_text(json.dumps({"id": "intro"}) + "\n", encoding="utf-8")

    def raising_client():
        raise RuntimeError("no client")

    monkeypatch.setattr(q_cli, "load_client", raising_client)

    def fake_generate(topics, per_topic, client, ensure_coverage):
        return []

    monkeypatch.setattr(q_cli, "generate_questions", fake_generate)
    assert q_cli._cmd_questions_generate(args) == 1
    assert "No questions generated" in capsys.readouterr().out

    monkeypatch.setattr(q_cli, "load_client", lambda: SimpleNamespace())

    def fake_generate_success(topics, per_topic, client, ensure_coverage):
        return [{"id": "intro-1", "topic_id": "intro", "type": "mcq", "stem": "Q", "choices": [], "answer": "A", "explanation": ""}]

    monkeypatch.setattr(q_cli, "generate_questions", fake_generate_success)
    assert q_cli._cmd_questions_generate(args) == 0
    assert (topics_dir / "questions.jsonl").exists()


def test_cmd_questions_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "quizzer.toml"
    cfg_path.write_text("[quiz.demo]\n", encoding="utf-8")
    args = Namespace(name="demo", config=None, topics=None)

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: None)
    assert q_cli._cmd_questions_list(args) == 2

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: cfg_path)
    monkeypatch.setattr(q_cli, "_load_toml", lambda *_: {})
    out_dir = tmp_path / ".quizzer" / "demo"
    assert q_cli._cmd_questions_list(args) == 1
    assert "No questions found" in capsys.readouterr().out

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.jsonl").write_text(json.dumps({"id": "1", "topic_id": "intro", "stem": "Q", "answer": "A"}) + "\n", encoding="utf-8")
    assert q_cli._cmd_questions_list(args) == 0
    assert "[intro]" in capsys.readouterr().out

    filt_args = Namespace(name="demo", config=None, topics=["missing"])
    assert q_cli._cmd_questions_list(filt_args) == 1
    assert "No questions to show" in capsys.readouterr().out


def test_cmd_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "quizzer.toml"
    cfg_path.write_text("[quiz.demo]\n", encoding="utf-8")
    args = Namespace(name="demo", config=None, shuffle=True, num=1)

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: None)
    assert q_cli._cmd_start(args) == 2

    monkeypatch.setattr(q_cli, "_find_config", lambda *_: cfg_path)
    monkeypatch.setattr(q_cli, "_load_toml", lambda *_: {})
    out_dir = tmp_path / ".quizzer" / "demo"
    assert q_cli._cmd_start(args) == 1
    assert "No questions found" in capsys.readouterr().out

    out_dir.mkdir(parents=True, exist_ok=True)
    questions_file = out_dir / "questions.jsonl"
    questions_file.write_text("", encoding="utf-8")
    assert q_cli._cmd_start(args) == 1
    assert "Question bank is empty" in capsys.readouterr().out

    questions_file.write_text(json.dumps({"id": "1", "stem": "Q", "answer": "A", "choices": []}) + "\n", encoding="utf-8")

    recorded = {}

    class StubQuizApp:
        def __init__(self, questions):
            recorded["questions"] = questions

        def run(self):
            recorded["ran"] = True

    monkeypatch.setattr(q_cli, "QuizApp", StubQuizApp)
    assert q_cli._cmd_start(args) == 0
    assert recorded["ran"] and len(recorded["questions"]) == 1


def test_main_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_init(args):
        called["init"] = True
        return 0

    monkeypatch.setattr(q_cli, "_cmd_init", fake_init)
    with pytest.raises(SystemExit) as exc:
        q_cli.main(["init", "demo"])
    assert exc.value.code == 0 and called["init"]

    with pytest.raises(SystemExit) as exc2:
        q_cli.main(["review", "demo"])
    assert exc2.value.code == 2

    with pytest.raises(SystemExit) as exc3:
        q_cli.main(["report", "demo"])
    assert exc3.value.code == 2


def test_main_dispatch_other_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(q_cli, "_cmd_topics_generate", lambda args: calls.append("tg") or 0)
    monkeypatch.setattr(q_cli, "_cmd_topics_list", lambda args: calls.append("tl") or 0)
    monkeypatch.setattr(q_cli, "_cmd_questions_generate", lambda args: calls.append("qg") or 0)
    monkeypatch.setattr(q_cli, "_cmd_questions_list", lambda args: calls.append("ql") or 0)
    monkeypatch.setattr(q_cli, "_cmd_start", lambda args: calls.append("start") or 0)
    with pytest.raises(SystemExit) as exc:
        q_cli.main(["topics", "generate", "demo"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit):
        q_cli.main(["topics", "list", "demo"])
    with pytest.raises(SystemExit):
        q_cli.main(["questions", "generate", "demo"])
    with pytest.raises(SystemExit):
        q_cli.main(["questions", "list", "demo"])
    with pytest.raises(SystemExit):
        q_cli.main(["start", "demo"])
    assert calls == ["tg", "tl", "qg", "ql", "start"]


def test_quizzer_dunder_main_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import study_utils.quizzer.__main__ as quiz_dunder

    def fake_main() -> None:
        raise SystemExit(0)

    monkeypatch.setattr(q_cli, "main", fake_main)
    module = importlib.reload(quiz_dunder)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0

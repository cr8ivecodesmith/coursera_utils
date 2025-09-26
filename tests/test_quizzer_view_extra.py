from __future__ import annotations

from types import SimpleNamespace

import pytest

from study_utils.quizzer.view import quiz as qv


class StubContainer:
    def __init__(self, *_, **kwargs):
        self.id = kwargs.get("id")
        self.children = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def remove_children(self) -> None:
        self.children.clear()

    def mount(self, widget) -> None:
        self.children.append(widget)


class StubVertical(StubContainer):
    pass


class StubStatic:
    def __init__(self, text: str, id: str | None = None):
        self.text = text
        self.id = id

    def update(self, new: str) -> None:
        self.text = new


class StubButton:
    def __init__(self, label: str, id: str | None = None):
        self.label = label
        self.id = id

    def add_class(self, name: str) -> None:
        raise RuntimeError("no add_class")


def test_quiz_app_compose_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qv, "Container", StubContainer)
    monkeypatch.setattr(qv, "Static", StubStatic)
    empty_app = qv.QuizApp([])
    rendered = list(empty_app.compose())
    empty_app._update_stage()
    assert rendered and rendered[0].text == "No questions."


def test_quiz_app_compose_and_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qv, "Container", StubContainer)
    monkeypatch.setattr(qv, "Vertical", StubVertical)
    monkeypatch.setattr(qv, "Static", StubStatic)
    monkeypatch.setattr(qv, "Button", StubButton)

    question = {"id": "1", "stem": "Q1", "answer": "A", "choices": [{"key": "A", "text": "Ans"}]}
    app = qv.QuizApp([question])

    stage = StubContainer(id="stage")
    answered = StubStatic("", id="answered")

    def failing_query(selector: str, _type):
        if selector == "#stage":
            return stage
        if selector == "#answered":
            raise LookupError
        raise LookupError

    monkeypatch.setattr(app, "query_one", failing_query)
    list(app.compose())
    assert not app.select_answer("")
    assert app.select_answer("A")
    app.action_next()
    app.action_prev()
    app.action_submit()
    app.action_select_b()
    app.action_select_c()
    app.action_select_d()

    monkeypatch.setattr(app, "query_one", lambda selector, _type: stage if selector == "#stage" else answered)
    app._update_stage()
    assert answered.text.startswith("Answered")


def test_question_view_compose_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qv.Container, "__enter__", lambda self: self, raising=False)
    monkeypatch.setattr(qv.Container, "__exit__", lambda self, exc_type, exc, tb: False, raising=False)
    monkeypatch.setattr(qv.Vertical, "__enter__", lambda self: self, raising=False)
    monkeypatch.setattr(qv.Vertical, "__exit__", lambda self, exc_type, exc, tb: False, raising=False)
    monkeypatch.setattr(qv, "Button", StubButton)
    monkeypatch.setattr(qv, "Static", StubStatic)

    view = qv.QuestionView(
        {"stem": "Q", "choices": ["Option", {"key": "b", "text": "Choice"}], "answer": "B", "explanation": "Because"},
        index=1,
        total=2,
        selected="?",
    )
    elements = list(view.compose())
    ids = {getattr(elem, "id", "") for elem in elements}
    assert {"stem", "progress", "feedback"}.issubset(ids)
    assert view.feedback_text("B").startswith("Correct")
    assert "Incorrect" in view.feedback_text("A")


def test_summarize_results_handles_unknown_and_unanswered() -> None:
    questions = [{"id": "1", "stem": "Q1", "answer": "A"}]
    summary = qv.summarize_results(questions, {"missing": "B"})
    assert summary[0]["selected"] is None


def test_quiz_app_summary_helpers() -> None:
    app = qv.QuizApp([
        {"id": "1", "stem": "Q1", "answer": "A", "choices": [{"key": "A", "text": "Ans"}]}
    ])
    app.action_select_a()
    app._summary = [{"id": str(i)} for i in range(6)]
    assert app.summary_page_count(page_size=4) == 2
    assert len(app.summary_items_for_page(1, page_size=4)) == 2


class StubEvent:
    def __init__(self, button_id: str):
        self.button = SimpleNamespace(id=button_id)


def test_on_button_pressed_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qv, "Container", StubContainer)
    monkeypatch.setattr(qv, "Vertical", StubVertical)
    monkeypatch.setattr(qv, "Static", StubStatic)
    monkeypatch.setattr(qv, "Button", StubButton)
    app = qv.QuizApp([
        {"id": "1", "stem": "Q1", "answer": "A", "choices": [{"key": "A", "text": "Ans"}]}
    ])
    monkeypatch.setattr(app, "query_one", lambda selector, _type: StubContainer(id=selector))
    list(app.compose())
    app.on_button_pressed(StubEvent("choice-A"))
    app.on_button_pressed(StubEvent("submit"))
    app.on_button_pressed(StubEvent("next"))
    app.on_button_pressed(StubEvent("prev"))

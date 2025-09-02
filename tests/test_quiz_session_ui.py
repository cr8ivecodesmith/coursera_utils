from app import quizzer as qz
from textual.widgets import Button, Static


def test_question_view_composes_expected_widgets():
    question = {
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
        "explanation": "",
    }
    view = qz.QuestionView(question, index=1, total=10)
    widgets = list(view.compose())
    # First widget is stem
    assert isinstance(widgets[0], Static)
    assert "2+2" in str(widgets[0].renderable)
    # The Vertical context yields Buttons directly in our compose
    buttons = [b for b in widgets if isinstance(b, Button)]
    assert len(buttons) == 4
    assert any("A)" in str(b.label) for b in buttons)
    # Progress present
    assert isinstance(widgets[-1], Static)
    assert widgets[-1].id == "progress"


def test_quiz_app_initial_state():
    q = {
        "id": "q1",
        "topic_id": "intro",
        "type": "mcq",
        "stem": "Q",
        "choices": [{"key": "A", "text": "x"}, {"key": "B", "text": "y"}],
        "answer": "A",
        "explanation": "",
    }
    app = qz.QuizApp([q])
    assert app._index == 0
    assert len(app._questions) == 1


from app import quizzer as qz
from textual.widgets import Button, Static


def test_question_view_feedback_text():
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
        "explanation": "because 4",
    }
    view = qz.QuestionView(question, index=1, total=10)
    assert view.feedback_text("B").startswith("Correct")
    assert view.feedback_text("A").startswith("Incorrect")


def test_quiz_app_initial_state():
    q = {
        "id": "q1",
        "topic_id": "intro",
        "type": "mcq",
        "stem": "Q",
        "choices": [{"key": "A", "text": "x"}, {"key": "B", "text": "y"}],
        "answer": "A",
        "explanation": "because 4",
    }
    app = qz.QuizApp([q])
    assert app._index == 0
    assert len(app._questions) == 1


def test_quiz_app_navigation_and_selection():
    qs = [
        {
            "id": "q1",
            "topic_id": "intro",
            "type": "mcq",
            "stem": "Q1",
            "choices": [{"key": "A", "text": "x"}, {"key": "B", "text": "y"}],
            "answer": "B",
            "explanation": "because 4",
        },
        {
            "id": "q2",
            "topic_id": "intro",
            "type": "mcq",
            "stem": "Q2",
            "choices": [{"key": "A", "text": "x"}, {"key": "B", "text": "y"}],
            "answer": "A",
            "explanation": "because 4",
        },
    ]
    app = qz.QuizApp(qs)
    assert app.current_question()["id"] == "q1"
    assert app.answered_count() == 0
    assert app.select_answer("A") is True
    assert app.answered_count() == 1
    app.action_select_b()
    qid = app.current_question()["id"]
    assert app._selected.get(qid) == "B"
    app.next_question()
    assert app.current_question()["id"] == "q2"
    app.prev_question()
    assert app.current_question()["id"] == "q1"


def test_summarize_results_and_pagination():
    from app import quizzer as qz
    qs = [
        {"id": f"q{i}", "topic_id": "t", "type": "mcq", "stem": f"Q{i}", "choices": [{"key":"A","text":"x"},{"key":"B","text":"y"}], "answer": "A", "explanation": ""}
        for i in range(1, 13)
    ]
    selected = {"q1": "A", "q2": "B", "q3": "A"}
    summary = qz.summarize_results(qs, selected)
    m = {it["id"]: it for it in summary}
    assert m["q1"]["correct"] is True
    assert m["q2"]["correct"] is False
    assert m["q4"]["selected"] is None
    app = qz.QuizApp(qs)
    app._summary = summary
    assert app.summary_page_count() == 3
    assert len(app.summary_items_for_page(0)) == 5
    assert len(app.summary_items_for_page(2)) == 2

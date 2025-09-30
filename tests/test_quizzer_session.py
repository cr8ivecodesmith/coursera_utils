from __future__ import annotations

from rich.console import Console

from study_utils.quizzer.session import (
    QuizSessionResult,
    QuizSessionState,
    SessionCommand,
    TopicSummary,
    _apply_command,
    parse_session_command,
    run_quiz_session,
)
from study_utils.quizzer.session import Choice, QuestionRecord


def make_provider(commands: list[str]):
    iterator = iter(commands)

    def _provider() -> str:
        return next(iterator)

    return _provider


def test_parse_session_command_variants() -> None:
    assert parse_session_command("a") == SessionCommand("select", "A")
    assert parse_session_command("  Next ") == SessionCommand("next")
    assert parse_session_command("p") == SessionCommand("prev")
    assert parse_session_command("submit") == SessionCommand("submit")
    assert parse_session_command("quit") == SessionCommand("quit")
    assert parse_session_command(None) is None
    assert parse_session_command("") is None
    assert parse_session_command("?unknown") is None


def test_run_quiz_session_submit_flow() -> None:
    console = Console(record=True, width=80, force_terminal=True)
    questions = [
        {
            "id": "q1",
            "stem": "What is the capital of France?",
            "choices": [
                {"key": "A", "text": "Paris"},
                {"key": "B", "text": "London"},
            ],
            "answer": "A",
            "topic_id": "geography",
            "explanation": "Paris is the capital city of France.",
        },
        {
            "id": "q2",
            "stem": "Select the even number.",
            "choices": [
                {"key": "A", "text": "2"},
                {"key": "B", "text": "3"},
            ],
            "answer": "A",
            "topic_id": "math",
            "explanation": "2 is divisible by 2.",
        },
    ]
    provider = make_provider(["b", "n", "a", "submit"])

    result = run_quiz_session(
        questions,
        console,
        provider,
        show_explanations=True,
    )

    assert isinstance(result, QuizSessionResult)
    assert result.exit_action == "submitted"
    assert result.summary.total_questions == 2
    assert result.summary.correct_answers == 1
    assert result.summary.answered_questions == 2
    explanations = [response.explanation for response in result.responses]
    assert any(explanations)
    rendered = console.export_text()
    assert "Quiz Summary" in rendered
    assert "Paris" in rendered
    assert "Ending session" not in rendered


def test_run_quiz_session_quit_flow_hides_explanations() -> None:
    console = Console(record=True, width=80, force_terminal=True)
    questions = [
        {
            "id": "only",
            "stem": "Pick A",
            "choices": [
                {"key": "A", "text": "A"},
                {"key": "B", "text": "B"},
            ],
            "answer": "A",
            "topic_id": "letters",
            "explanation": "Because it says so.",
        }
    ]
    provider = make_provider(["a", "quit"])

    result = run_quiz_session(
        questions,
        console,
        provider,
        show_explanations=False,
    )

    assert result.exit_action == "quit"
    assert result.summary.total_questions == 1
    assert result.summary.answered_questions == 1
    assert result.responses[0].explanation is None
    assert "Ending session" in console.export_text()


def test_run_quiz_session_handles_unanswered_and_invalid_choice() -> None:
    console = Console(record=True, width=80, force_terminal=True)
    questions = [
        {
            "id": "q1",
            "stem": "First question",
            "choices": [
                {"key": "A", "text": "Alpha"},
                {"key": "B", "text": "Beta"},
            ],
            "answer": "B",
            "topic_id": "letters",
        },
        {
            "id": "q2",
            "stem": "Second question",
            "choices": ["Option 1", "Option 2"],
            "answer": "Z",
            "topic_id": "numbers",
        },
        {
            "id": "q3",
            "stem": "Third question",
            "choices": None,
            "answer": "A",
            "topic_id": "misc",
        },
    ]
    provider = make_provider(["   ", "n", "p", "z", "a", "submit"])

    result = run_quiz_session(
        questions,
        console,
        provider,
        show_explanations=True,
    )

    output = console.export_text()
    assert "not a valid choice" in output
    assert result.exit_action == "submitted"
    assert result.summary.total_questions == 3
    assert result.summary.answered_questions == 1
    assert any(resp.selected is None for resp in result.responses)


def test_topic_summary_accuracy_handles_zero() -> None:
    summary = TopicSummary(topic_id="t", asked=0, correct=0)
    assert summary.accuracy == 0.0


def test_run_quiz_session_empty_question_bank() -> None:
    console = Console(record=True, width=80, force_terminal=True)

    result = run_quiz_session(
        [],
        console,
        lambda: "",
        show_explanations=True,
    )

    assert result.exit_action == "empty"
    assert result.summary.total_questions == 0
    assert "Question bank is empty" in console.export_text()


def test_run_quiz_session_handles_stop_iteration() -> None:
    console = Console(record=True, width=80, force_terminal=True)
    questions = [
        {
            "id": "q1",
            "stem": "A question",
            "choices": [
                {"key": "A", "text": "Answer"},
            ],
            "answer": "A",
        }
    ]

    result = run_quiz_session(
        questions,
        console,
        iter(()).__next__,
        show_explanations=True,
    )

    assert result.exit_action == "quit"
    assert result.summary.total_questions == 1
    assert "Session interrupted" in console.export_text()


def test_apply_command_ignores_select_without_choice() -> None:
    console = Console(record=True, width=80, force_terminal=True)
    record = QuestionRecord(
        id="q1",
        stem="Stem",
        choices=[Choice("A", "Choice A")],
        answer="A",
        explanation=None,
        topic_id=None,
        source={"id": "q1", "stem": "Stem", "choices": [], "answer": "A"},
    )
    state = QuizSessionState([record])

    result = _apply_command(SessionCommand("select", None), state, console)

    assert result is None
    assert state.selections == {}
    assert console.export_text().strip() == ""

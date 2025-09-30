"""Rich-powered quiz session controller and supporting data structures.

This module provides a synchronous session loop that renders multiple-choice
questions using Rich, captures user commands, and returns a structured
`QuizSessionResult` summarizing the interaction. The implementation keeps
stateful logic small and testable so the CLI can orchestrate quiz sessions
without depending on Textual artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Callable, Literal

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# NOTE: These helpers live in the legacy Textual view module today. They will
# be relocated into this module in a follow-up step so callers depend on the
# Rich session boundary instead of Textual.
from .view.quiz import aggregate_summary, summarize_results

InputProvider = Callable[[], str]
ExitAction = Literal["submitted", "quit", "empty"]


@dataclass(frozen=True)
class Choice:
    """Normalized multiple-choice option."""

    key: str
    text: str


@dataclass(frozen=True)
class QuestionRecord:
    """Immutable representation of a quiz question used during a session."""

    id: str
    stem: str
    choices: list[Choice]
    answer: str
    explanation: str | None
    topic_id: str | None
    source: dict[str, object]

    def choice_for(self, key: str | None) -> Choice | None:
        if not key:
            return None
        normalized = str(key).strip().upper()[:1]
        for choice in self.choices:
            if choice.key == normalized:
                return choice
        return None


@dataclass(frozen=True)
class QuestionResponse:
    """A user's response to a specific question."""

    question_id: str
    stem: str
    selected: str | None
    selected_text: str | None
    answer: str
    answer_text: str | None
    is_correct: bool
    topic_id: str | None = None
    explanation: str | None = None


@dataclass(frozen=True)
class TopicSummary:
    """Aggregate performance metrics for a single topic."""

    topic_id: str
    asked: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.asked == 0:
            return 0.0
        return self.correct / self.asked


@dataclass(frozen=True)
class QuizSummary:
    """Overall session summary derived from question responses."""

    total_questions: int
    correct_answers: int
    accuracy: float
    answered_questions: int
    per_topic: dict[str, TopicSummary] = field(default_factory=dict)


@dataclass(frozen=True)
class QuizSessionResult:
    """Return value from ``run_quiz_session``."""

    responses: list[QuestionResponse]
    summary: QuizSummary
    exit_action: ExitAction


@dataclass(frozen=True)
class SessionCommand:
    """Normalized user command parsed from console input."""

    type: Literal["next", "prev", "submit", "quit", "select"]
    choice: str | None = None


@dataclass
class QuizSessionState:
    """Mutable session state shared by the Rich UI loop."""

    questions: list[QuestionRecord]
    show_explanations: bool = True
    index: int = 0
    selections: dict[str, str] = field(default_factory=dict)

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    @property
    def current(self) -> QuestionRecord:
        return self.questions[self.index]

    def answered_count(self) -> int:
        return len(self.selections)

    def select(self, choice_key: str) -> bool:
        choice_key = choice_key.strip().upper()[:1]
        question = self.current
        if not question.choice_for(choice_key):
            return False
        self.selections[question.id] = choice_key
        return True

    def next(self) -> None:
        if self.index + 1 < self.total_questions:
            self.index += 1

    def previous(self) -> None:
        if self.index > 0:
            self.index -= 1

    def selected_for(
        self, question: QuestionRecord | None = None
    ) -> str | None:
        target = question or self.current
        return self.selections.get(target.id)

    def available_choice_keys(self) -> list[str]:
        return [choice.key for choice in self.current.choices]


def parse_session_command(raw: str | None) -> SessionCommand | None:
    """Parse raw user input into a structured command."""

    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"n", "next"}:
        return SessionCommand("next")
    if lowered in {"p", "prev", "previous"}:
        return SessionCommand("prev")
    if lowered in {"submit", "s"}:
        return SessionCommand("submit")
    if lowered in {"quit", "q", "exit"}:
        return SessionCommand("quit")
    key = text[0].upper()
    if key.isalnum():
        return SessionCommand("select", key)
    return None


def run_quiz_session(
    questions: Sequence[dict[str, object]],
    console: Console,
    input_provider: InputProvider,
    *,
    show_explanations: bool = True,
) -> QuizSessionResult:
    """Run an interactive quiz session using Rich-rendered prompts."""

    records = [
        _build_question_record(question, idx)
        for idx, question in enumerate(questions)
    ]
    state = QuizSessionState(records, show_explanations=show_explanations)

    if not state.questions:
        console.print(
            Panel(
                "Question bank is empty.",
                title="Quiz Session",
                border_style="yellow",
            )
        )
        summary = QuizSummary(
            total_questions=0,
            correct_answers=0,
            accuracy=0.0,
            answered_questions=0,
        )
        return QuizSessionResult([], summary, "empty")

    exit_action: ExitAction = "quit"
    while True:
        _render_question(console, state)
        try:
            raw = input_provider()
        except (EOFError, KeyboardInterrupt, StopIteration):
            console.print("\n[bold yellow]Session interrupted.[/]")
            exit_action = "quit"
            break
        command = parse_session_command(raw)
        if command is None:
            console.print("[red]Unrecognized command. Try again.[/]")
            continue
        exit_candidate = _apply_command(command, state, console)
        if exit_candidate:
            exit_action = exit_candidate
            break

    responses, summary = _build_responses_and_summary(state)
    result = QuizSessionResult(responses, summary, exit_action)

    if exit_action == "submitted":
        _render_summary(console, result, show_explanations=show_explanations)

    return result


def _apply_command(
    command: SessionCommand,
    state: QuizSessionState,
    console: Console,
) -> ExitAction | None:
    if command.type == "select" and command.choice:
        if state.select(command.choice):
            console.print(f"Selected [bold]{command.choice}[/].")
        else:
            console.print(
                "[red]'%s' is not a valid choice for this question.[/red]"
                % command.choice,
            )
        return None
    if command.type == "next":
        state.next()
        return None
    if command.type == "prev":
        state.previous()
        return None
    if command.type == "quit":
        console.print("\n[bold yellow]Ending session without submission.[/]")
        return "quit"
    if command.type == "submit":
        return "submitted"
    return None


def _build_question_record(
    data: dict[str, object],
    index: int,
) -> QuestionRecord:
    identifier = str(data.get("id", index))
    stem = str(data.get("stem", "")).strip()
    answer = str(data.get("answer", "")).strip().upper()[:1]
    explanation = data.get("explanation")
    topic = data.get("topic_id")
    choices = list(_iter_choices(data.get("choices"), answer))
    return QuestionRecord(
        id=identifier,
        stem=stem,
        choices=choices,
        answer=answer,
        explanation=str(explanation) if explanation is not None else None,
        topic_id=str(topic) if topic is not None else None,
        source=dict(data),
    )


def _iter_choices(choices_field: object, answer: str) -> Iterable[Choice]:
    if not isinstance(choices_field, Iterable) or isinstance(
        choices_field, (str, bytes)
    ):
        return [Choice(answer or "A", str(choices_field or ""))]
    normalized: list[Choice] = []
    for item in choices_field:
        if isinstance(item, dict):
            key = str(item.get("key", "")).strip().upper()[:1] or None
            text = str(item.get("text", "")).strip()
        else:
            key = None
            text = str(item)
        if not key:
            key = chr(ord("A") + len(normalized))
        normalized.append(Choice(key, text))
    return normalized


def _render_question(console: Console, state: QuizSessionState) -> None:
    question = state.current
    header = Text.assemble(
        (f"Question {state.index + 1}", "bold cyan"),
        (f" / {state.total_questions}", "dim"),
    )
    console.print()
    console.rule(header)
    console.print(Text(question.stem, style="bold"))

    table = Table(show_header=False, box=box.SIMPLE, expand=True)
    table.add_column("Key", justify="center", style="cyan")
    table.add_column("Choice")

    selected = state.selected_for(question)
    for choice in question.choices:
        indicator = "•" if choice.key == selected else " "
        choice_text = (
            Text(choice.text) if choice.text else Text("", style="dim")
        )
        if choice.key == selected:
            choice_text.stylize("bold green")
        row_text = Text(indicator + " ")
        row_text += choice_text
        table.add_row(choice.key, row_text)

    console.print(table)
    choice_hint = ", ".join(state.available_choice_keys())
    command_hint = (
        f"Commands: choices [{choice_hint}], n (next), p (prev), submit, quit"
    )
    console.print(
        Text(
            f"Answered {state.answered_count()}/{state.total_questions} | "
            f"{command_hint}",
            style="dim",
        )
    )


def _build_responses_and_summary(
    state: QuizSessionState,
) -> tuple[list[QuestionResponse], QuizSummary]:
    raw_responses = summarize_results(
        [record.source for record in state.questions],
        state.selections,
    )
    responses: list[QuestionResponse] = []
    aggregate_input: list[dict[str, object]] = []
    question_by_id = {record.id: record for record in state.questions}

    for item in raw_responses:
        qid = str(item.get("id", ""))
        record = question_by_id.get(qid)
        selected = item.get("selected")
        answer = item.get("answer")
        is_correct = bool(item.get("correct"))
        choice_obj = record.choice_for(selected) if record else None
        answer_obj = record.choice_for(answer) if record else None
        explanation = (
            record.explanation if (record and state.show_explanations) else None
        )
        topic_id = record.topic_id if record else None

        responses.append(
            QuestionResponse(
                question_id=qid,
                stem=str(item.get("stem", "")),
                selected=str(selected) if selected is not None else None,
                selected_text=choice_obj.text if choice_obj else None,
                answer=str(answer) if answer is not None else "",
                answer_text=answer_obj.text if answer_obj else None,
                is_correct=is_correct,
                topic_id=topic_id,
                explanation=explanation,
            )
        )

        aggregate_input.append(
            {
                "question_id": qid,
                "topic_id": topic_id or "",
                "correct": is_correct,
            }
        )

    aggregate = aggregate_summary(aggregate_input)
    answered_count = sum(
        1 for response in responses if response.selected is not None
    )

    summary = QuizSummary(
        total_questions=int(aggregate.get("total", 0)),
        correct_answers=int(aggregate.get("correct", 0)),
        accuracy=float(aggregate.get("accuracy", 0.0)),
        answered_questions=answered_count,
        per_topic={
            topic_id: TopicSummary(
                topic_id=topic_id,
                asked=int(values.get("asked", 0)),
                correct=int(values.get("correct", 0)),
            )
            for topic_id, values in (
                aggregate.get("per_topic", {}) or {}
            ).items()
        },
    )
    return responses, summary


def _render_summary(
    console: Console,
    result: QuizSessionResult,
    *,
    show_explanations: bool,
) -> None:
    console.print()
    console.rule(Text("Quiz Summary", style="bold magenta"))

    summary = result.summary
    overview = Table(
        show_header=False,
        box=box.MINIMAL_DOUBLE_HEAD,
        expand=False,
    )
    overview.add_column("Metric", style="bold")
    overview.add_column("Value", justify="right")
    overview.add_row("Total questions", str(summary.total_questions))
    overview.add_row("Answered", str(summary.answered_questions))
    overview.add_row("Correct", str(summary.correct_answers))
    overview.add_row("Accuracy", f"{summary.accuracy * 100:.1f}%")
    console.print(overview)

    if summary.per_topic:
        per_topic = Table(title="Per topic", box=box.SIMPLE, expand=False)
        per_topic.add_column("Topic")
        per_topic.add_column("Asked", justify="right")
        per_topic.add_column("Correct", justify="right")
        per_topic.add_column("Accuracy", justify="right")
        for topic_id, metrics in summary.per_topic.items():
            per_topic.add_row(
                topic_id or "(unknown)",
                str(metrics.asked),
                str(metrics.correct),
                f"{metrics.accuracy * 100:.1f}%",
            )
        console.print(per_topic)

    response_table = Table(title="Responses", box=box.SIMPLE, expand=True)
    response_table.add_column("#", justify="right")
    response_table.add_column("Question", overflow="fold")
    response_table.add_column("Your answer")
    response_table.add_column("Correct answer")
    response_table.add_column("Result", justify="center")

    for idx, response in enumerate(result.responses, start=1):
        your = response.selected or "—"
        correct = response.answer or "—"
        outcome = "✅" if response.is_correct else "❌"
        response_table.add_row(
            str(idx),
            response.stem or f"Question {idx}",
            your,
            correct,
            outcome,
        )
    console.print(response_table)

    if show_explanations:
        for response in result.responses:
            if not response.explanation:
                continue
            border = "green" if response.is_correct else "red"
            console.print(
                Panel(
                    response.explanation,
                    title=f"Explanation — {response.question_id}",
                    border_style=border,
                )
            )

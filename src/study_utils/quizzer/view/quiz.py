from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button
from textual.containers import Vertical, Container


def aggregate_summary(
    responses: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    total = len(responses)
    correct = sum(1 for r in responses if bool(r.get("correct")))
    per_topic: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"asked": 0, "correct": 0}
    )
    for r in responses:
        t = str(r.get("topic_id", ""))
        per_topic[t]["asked"] += 1
        if r.get("correct"):
            per_topic[t]["correct"] += 1
    accuracy = (correct / total) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_topic": dict(per_topic),
    }


def summarize_results(
    questions: Sequence[Dict[str, object]], selected: Dict[str, str]
) -> List[Dict[str, object]]:
    """Summarize results comparing selected answers to correct answers.

    Returns a list with one entry per question with fields:
    id, stem, selected, answer, correct (bool)
    """
    items: List[Dict[str, object]] = []
    qmap = {str(q.get("id")): q for q in questions}
    # Add answered first
    for qid, choice in selected.items():
        q = qmap.get(str(qid))
        if not q:
            continue
        ans = str(q.get("answer", "")).strip().upper()
        items.append(
            {
                "id": str(qid),
                "stem": q.get("stem", ""),
                "selected": str(choice).strip().upper(),
                "answer": ans,
                "correct": str(choice).strip().upper() == ans,
            }
        )
    # Then unanswered
    for q in questions:
        qid = str(q.get("id"))
        if qid in selected:
            continue
        items.append(
            {
                "id": qid,
                "stem": q.get("stem", ""),
                "selected": None,
                "answer": str(q.get("answer", "")).strip().upper(),
                "correct": False,
            }
        )
    return items


class QuizApp(App):
    CSS_PATH = None
    CSS = """
#choices Button.selected { background: $accent; color: black; }
#nav { color: $text; }
"""
    BINDINGS = [
        ("n", "next", "Next"),
        ("p", "prev", "Prev"),
        ("a", "select_a", "Select A"),
        ("b", "select_b", "Select B"),
        ("c", "select_c", "Select C"),
        ("d", "select_d", "Select D"),
        ("enter", "submit", "Submit"),
        ("s", "submit", "Submit"),
    ]

    def __init__(self, questions: Sequence[Dict[str, object]]):
        super().__init__()
        self._questions = list(questions)
        self._index = 0
        self._selected: Dict[str, str] = {}

    def compose(self) -> ComposeResult:
        if not self._questions:
            yield Static("No questions.", id="empty")
            return
        with Container(id="stage"):
            q = self._questions[self._index]
            qid = str(q.get("id", self._index))
            sel = self._selected.get(qid)
            yield QuestionView(
                q,
                index=self._index + 1,
                total=len(self._questions),
                selected=sel,
            )
        with Container(id="footer"):
            yield Button("Prev", id="prev")
            yield Button("Next", id="next")
            yield Button("Submit", id="submit")
            yield Static(self._answered_text(), id="answered")
            yield Static("", id="confirm")

    # Pure helpers for navigation and selection (testable without running App)
    def current_question(self) -> Dict[str, object]:
        return self._questions[self._index]

    def next_question(self) -> int:
        if self._index + 1 < len(self._questions):
            self._index += 1
        self._update_stage()
        return self._index

    def prev_question(self) -> int:
        if self._index > 0:
            self._index -= 1
        self._update_stage()
        return self._index

    def select_answer(self, key: str) -> bool:
        q = self.current_question()
        k = str(key).strip().upper()[:1]
        if not k:
            return False
        qid = str(q.get("id", self._index))
        self._selected[qid] = k
        self._update_stage()
        return True

    def _update_stage(self) -> None:
        if not self._questions:
            return
        try:
            stage = self.query_one("#stage", Container)
        except Exception:
            return
        stage.remove_children()
        q = self._questions[self._index]
        qid = str(q.get("id", self._index))
        sel = self._selected.get(qid)
        stage.mount(
            QuestionView(
                q,
                index=self._index + 1,
                total=len(self._questions),
                selected=sel,
            )
        )
        try:
            ans = self.query_one("#answered", Static)
            ans.update(self._answered_text())
        except Exception:
            pass

    def action_next(self) -> None:
        self.next_question()

    def action_prev(self) -> None:
        self.prev_question()

    def action_select_a(self) -> None:
        self.select_answer("A")

    def action_select_b(self) -> None:
        self.select_answer("B")

    def action_select_c(self) -> None:
        self.select_answer("C")

    def action_select_d(self) -> None:
        self.select_answer("D")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = getattr(event.button, "id", "") or ""
        if bid.startswith("choice-") and len(bid) >= 8:
            self.select_answer(bid[-1])
        elif bid == "submit":
            self.action_submit()
        elif bid == "next":
            self.action_next()
        elif bid == "prev":
            self.action_prev()

    def action_submit(self) -> None:
        # No-op for now; will evaluate/save later
        pass

    def _answered_text(self) -> str:
        return f"Answered: {len(self._selected)}/{len(self._questions)}"

    def answered_count(self) -> int:
        return len(self._selected)

    def summary_page_count(self, page_size: int = 5) -> int:
        total = len(getattr(self, "_summary", []))
        return (total + page_size - 1) // page_size if total else 1

    def summary_items_for_page(
        self, page: int, page_size: int = 5
    ) -> List[Dict[str, object]]:
        items = list(getattr(self, "_summary", []))
        start = page * page_size
        end = start + page_size
        return items[start:end]


class QuestionView(Widget):
    """A simple view that renders a single MCQ with choices, progress and status."""

    def __init__(
        self,
        question: Dict[str, object],
        index: int,
        total: int,
        *,
        selected: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.question = question
        self.index = index
        self.total = total
        self.selected = (selected or "").strip().upper() or None

    def compose(self) -> ComposeResult:
        stem = str(self.question.get("stem", "")).strip()
        yield Static(stem, id="stem")
        choices = self.question.get("choices") or []
        with Vertical(id="choices"):
            for ch in choices:  # type: ignore[assignment]
                if isinstance(ch, dict):
                    key = str(ch.get("key", "")).upper()[:1] or "?"
                    text = str(ch.get("text", ""))
                else:
                    key = "?"
                    text = str(ch)
                label = f"{key}) {text}"
                btn = Button(label, id=f"choice-{key}")
                if self.selected and key == self.selected:
                    try:
                        btn.add_class("selected")
                    except Exception:
                        pass
                yield btn
        prog = f"{self.index}/{self.total}"
        yield Static(prog, id="progress")
        status = f"Selected: {self.selected}" if self.selected else ""
        yield Static(status, id="feedback")

    def compute_correct(self, key: str) -> tuple[bool, str]:
        """Evaluate a choice key against the question's answer.

        Returns (is_correct, explanation).
        """
        answer = str(self.question.get("answer", "")).strip().upper()
        explanation = str(self.question.get("explanation", ""))
        return (key.strip().upper() == answer, explanation)

    def feedback_text(self, key: str) -> str:
        ok, expl = self.compute_correct(key)
        return (
            "Correct."
            if ok
            else (f"Incorrect. {expl}" if expl else "Incorrect.")
        )

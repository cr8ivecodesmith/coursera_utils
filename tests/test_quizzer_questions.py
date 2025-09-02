import json
import pytest


class _FakeClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                # capture prompt for assertions
                _FakeClient.last_messages = kwargs.get("messages")
                payload = [
                    {
                        "stem": "What is 2+2?",
                        "choices": ["3", "4", "5", "22"],
                        "answer": "B",
                        "explanation": "2+2=4",
                    },
                    {
                        "stem": "Select a prime number.",
                        "choices": ["4", {"key": "b", "text": "5"}, "6", "8"],
                        "answer": "b",
                        "explanation": "5 is prime",
                    },
                ]

                class _Resp:
                    class _Choice:
                        class _Msg:
                            content = json.dumps(payload)
                        message = _Msg()
                    choices = [_Choice()]
                return _Resp()


def test_ai_generate_mcqs_for_topic_parses_and_validates():
    from app import quizzer as qz

    topic = {"id": "intro", "name": "Intro"}
    res = qz.ai_generate_mcqs_for_topic(topic, n=2, client=_FakeClient())
    assert len(res) == 2
    for q in res:
        assert q["type"] == "mcq"
        assert q["topic_id"] == "intro"
        keys = [c["key"] for c in q["choices"]]
        assert all(len(k) == 1 and k.isalpha() and k == k.upper() for k in keys)
        assert q["answer"] in set(keys)


def test_generate_questions_aggregates_by_topic():
    from app import quizzer as qz

    topics = [
        {"id": "intro", "name": "Intro"},
        {"id": "basics", "name": "Basics"},
    ]
    res = qz.generate_questions(topics, per_topic=2, client=_FakeClient())
    assert len(res) == 4
    from collections import Counter
    c = Counter(q["topic_id"] for q in res)
    assert c["intro"] == 2 and c["basics"] == 2


def test_generate_questions_passes_context(tmp_path):
    from app import quizzer as qz

    # Create a source file with a matching heading and distinctive token
    token = "EULER_IDENTITY"
    text = f"# Intro\nSome background...\n{token} appears here.\n"
    src = tmp_path / "notes.md"
    src.write_text(text)
    topics = [
        {"id": "intro", "name": "Intro", "source_paths": [str(src)]},
    ]
    _ = qz.generate_questions(topics, per_topic=1, client=_FakeClient())
    # Ensure the captured messages include our token in the user content
    msgs = getattr(_FakeClient, "last_messages", []) or []
    user_contents = "\n\n".join(m.get("content", "") for m in msgs if m.get("role") == "user")
    assert token in user_contents

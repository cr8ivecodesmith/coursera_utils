import json
from pathlib import Path
from types import SimpleNamespace


def test_extract_topics_with_ai_merges_suggestions(tmp_path: Path):
    from study_utils import quizzer as qz

    md = """
    # Intro
    Some context
    ## Basics
    Details here
    """
    f = tmp_path / "notes.md"
    f.write_text(md)

    class FakeClient:
        def __init__(self):
            payload = [
                {
                    "name": "Advanced Techniques",
                    "description": "Deepdive",
                }
            ]

            def create(**kwargs):
                message = SimpleNamespace(content=json.dumps(payload))
                choice = SimpleNamespace(message=message)
                return SimpleNamespace(choices=[choice])

            completions = SimpleNamespace(create=create)
            self.chat = SimpleNamespace(completions=completions)

    # Once implemented, this should return Intro,
    # Basics (heuristics) + Advanced Techniques (AI)
    topics = qz.extract_topics([(f, md)], use_ai=True, client=FakeClient(), k=5)
    names = {t["name"] for t in topics}
    assert {"Intro", "Basics", "Advanced Techniques"}.issubset(names)

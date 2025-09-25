from pathlib import Path
import pytest


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
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    # The real implementation will return JSON with topic names
                    class _Resp:
                        class _Choice:
                            class _Msg:
                                # Pretend AI suggests an extra topic beyond headings
                                content = '[{"name": "Advanced Techniques", "description": "Deep dive"}]'
                            message = _Msg()
                        choices = [_Choice()]
                    return _Resp()

    # Once implemented, this should return Intro, Basics (heuristics) + Advanced Techniques (AI)
    topics = qz.extract_topics([(f, md)], use_ai=True, client=FakeClient(), k=5)
    names = {t["name"] for t in topics}
    assert {"Intro", "Basics", "Advanced Techniques"}.issubset(names)


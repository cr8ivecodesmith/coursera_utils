import sys
import types
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub out heavy/optional deps at import time so tests can import transcribe_video
if 'pydub' not in sys.modules:
    pydub = types.ModuleType('pydub')
    class _AudioSegment:
        @classmethod
        def from_file(cls, *a, **k):
            raise RuntimeError('AudioSegment.from_file should be stubbed in tests that use it')
    pydub.AudioSegment = _AudioSegment
    sys.modules['pydub'] = pydub
if 'pydub.utils' not in sys.modules:
    utils = types.ModuleType('pydub.utils')
    def make_chunks(*a, **k):
        return []
    utils.make_chunks = make_chunks
    sys.modules['pydub.utils'] = utils

if 'openai' not in sys.modules:
    openai = types.ModuleType('openai')
    class OpenAI:
        def __init__(self, *a, **k):
            pass
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class _Resp:
                        class _Choice:
                            class _Msg:
                                content = ""
                            message = _Msg()
                        choices = [_Choice()]
                    return _Resp()
    openai.OpenAI = OpenAI
    sys.modules['openai'] = openai

if 'dotenv' not in sys.modules:
    dotenv = types.ModuleType('dotenv')
    def load_dotenv(*a, **k):
        return None
    dotenv.load_dotenv = load_dotenv
    sys.modules['dotenv'] = dotenv

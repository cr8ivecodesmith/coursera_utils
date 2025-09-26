"""Simple stubs for the pydub modules used in tests."""

from __future__ import annotations

from types import ModuleType
from typing import Dict, Optional


class AudioSegmentStub:
    @classmethod
    def from_file(cls, *args, **kwargs):
        raise RuntimeError(
            "AudioSegment.from_file should be patched within tests before use"
        )


def _build_pydub_modules() -> Dict[str, ModuleType]:
    audio_module = ModuleType("pydub")
    audio_module.AudioSegment = AudioSegmentStub

    utils_module = ModuleType("pydub.utils")

    def make_chunks(*args, **kwargs):
        return []

    utils_module.make_chunks = make_chunks
    return {"pydub": audio_module, "pydub.utils": utils_module}


def install_pydub_stub_modules(target: Optional[Dict[str, ModuleType]] = None) -> None:
    import sys

    modules = _build_pydub_modules()
    target_dict = target if target is not None else sys.modules
    target_dict.update(modules)

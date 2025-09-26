"""Minimal WeasyPrint stubs for offline testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional


@dataclass
class GeneratedPDF:
    target: Path
    stylesheets: List[Any]
    html: str
    base_url: str


class HTMLStub:
    """Collects invocation arguments for ``write_pdf`` calls."""

    _calls: List[GeneratedPDF] = []

    def __init__(self, *, string: str, base_url: str) -> None:
        self._string = string
        self._base_url = base_url

    def write_pdf(self, *, target: str, stylesheets: Optional[List[Any]] = None) -> None:
        stylesheets = stylesheets or []
        record = GeneratedPDF(
            target=Path(target),
            stylesheets=list(stylesheets),
            html=self._string,
            base_url=self._base_url,
        )
        self.__class__._calls.append(record)

    @classmethod
    def pop_calls(cls) -> List[GeneratedPDF]:
        calls = list(cls._calls)
        cls._calls.clear()
        return calls


class CSSStub:
    def __init__(self, *, string: Optional[str] = None, filename: Optional[str] = None) -> None:
        self.string = string
        self.filename = filename


def _build_weasyprint_module() -> ModuleType:
    module = ModuleType("weasyprint")
    module.HTML = HTMLStub
    module.CSS = CSSStub
    return module


def install_weasyprint_stub_module(target: Optional[Dict[str, ModuleType]] = None) -> None:
    import sys

    module = _build_weasyprint_module()
    target_dict = target if target is not None else sys.modules
    target_dict["weasyprint"] = module

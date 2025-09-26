"""Simple dotenv stub so modules can import without optional dependency."""

from __future__ import annotations

from types import ModuleType
from typing import Dict, Optional


def _build_dotenv_module() -> ModuleType:
    module = ModuleType("dotenv")

    def load_dotenv(*args, **kwargs):
        return None

    module.load_dotenv = load_dotenv
    return module


def install_dotenv_stub_module(target: Optional[Dict[str, ModuleType]] = None) -> None:
    import sys

    module = _build_dotenv_module()
    target_dict = target if target is not None else sys.modules
    target_dict["dotenv"] = module

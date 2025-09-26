"""Shared AI helper utilities."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

try:  # Allow module import even when the OpenAI dependency is absent.
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None  # type: ignore

__all__ = ["load_client"]


def load_client() -> Any:
    """Initialize an OpenAI client using environment-derived credentials."""
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is required to create a client. "
            "Install it and retry."
        )
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment. Set it or add to .env"
        )
    return OpenAI(api_key=api_key)

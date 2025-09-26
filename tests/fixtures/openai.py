"""OpenAI client stubs shared across tests.

The production code imports ``from openai import OpenAI`` and expects the
returned object to expose ``chat.completions.create``.  The test stub mirrors the
minimal surface area so modules can run offline while exposing hooks to inspect
requests and queue deterministic responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Choice:
    """Represents a single completion choice returned by the stub."""

    content: str

    @property
    def message(self) -> SimpleNamespace:
        return SimpleNamespace(content=self.content)


class OpenAIStub:
    """Lightweight stand-in for ``OpenAI`` chat completion client."""

    def __init__(self, *, side_effect: Optional[Callable[[Dict[str, Any]], Any]] = None):
        self.side_effect = side_effect
        self.calls: List[Dict[str, Any]] = []
        self.responses: List[str] = []
        self._chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    @property
    def chat(self) -> SimpleNamespace:
        return self._chat

    def queue_response(self, content: str) -> None:
        """Append a response string returned on the next call."""

        self.responses.append(content)

    def _create_completion(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.side_effect:
            result = self.side_effect(kwargs)
            if result is not None:
                return result
        content = self.responses.pop(0) if self.responses else ""
        choice = Choice(content)
        return SimpleNamespace(choices=[choice])


class OpenAIStubFactory:
    """Factory installed as the ``OpenAI`` symbol in ``sys.modules``."""

    def __init__(self) -> None:
        self.instances: List[OpenAIStub] = []

    def __call__(self, *args: Any, **kwargs: Any) -> OpenAIStub:
        stub = OpenAIStub()
        stub.init_args = args  # type: ignore[attr-defined]
        stub.init_kwargs = kwargs  # type: ignore[attr-defined]
        self.instances.append(stub)
        return stub

    def reset(self) -> None:
        self.instances.clear()

    @property
    def last(self) -> Optional[OpenAIStub]:
        return self.instances[-1] if self.instances else None


def _build_openai_module(factory: OpenAIStubFactory) -> ModuleType:
    module = ModuleType("openai")
    module.OpenAI = factory  # type: ignore[attr-defined]
    module.BadRequestError = RuntimeError
    module.APIConnectionError = RuntimeError
    module.OpenAIError = RuntimeError
    module.RateLimitError = RuntimeError
    module.__dict__["_factory"] = factory
    return module


def install_openai_stub_module(target: Optional[Dict[str, ModuleType]] = None) -> OpenAIStubFactory:
    """Install the stub ``openai`` module into ``sys.modules``.

    Returns the factory so callers can inspect the created client instances.
    ``target`` defaults to ``sys.modules`` and is pluggable for tests.
    """

    import sys

    factory = OpenAIStubFactory()
    module = _build_openai_module(factory)
    target_dict = target if target is not None else sys.modules
    target_dict["openai"] = module
    return factory

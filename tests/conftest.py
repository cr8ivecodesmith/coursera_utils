from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from fixtures import (  # noqa: E402
    WorkspaceBuilder,
    install_dotenv_stub_module,
    install_openai_stub_module,
    install_pydub_stub_modules,
    install_weasyprint_stub_module,
)
from fixtures.openai import OpenAIStubFactory  # noqa: E402
from fixtures.weasyprint import HTMLStub  # noqa: E402

# Ensure project root and src/ are importable when tests spawn subprocesses
ROOT = TESTS_DIR.parent
for extra in (ROOT, ROOT / "src", ROOT / "src" / "study_utils" / "quizzer"):
    path_str = str(extra)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Install lightweight stub modules so imports succeed without heavy deps
OPENAI_FACTORY: OpenAIStubFactory = install_openai_stub_module()
install_dotenv_stub_module()
install_pydub_stub_modules()
install_weasyprint_stub_module()


@pytest.fixture(autouse=True)
def _reset_openai_factory() -> Iterator[None]:
    OPENAI_FACTORY.reset()
    yield
    OPENAI_FACTORY.reset()


@pytest.fixture
def openai_factory() -> OpenAIStubFactory:
    """Access the OpenAI stub factory to inspect calls or queue responses."""

    return OPENAI_FACTORY


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceBuilder:
    """Provide a helper bound to pytest's per-test tmp directory."""

    return WorkspaceBuilder(tmp_path)


@pytest.fixture(autouse=True)
def _clear_weasyprint_calls() -> Iterator[None]:
    HTMLStub.pop_calls()
    yield
    HTMLStub.pop_calls()

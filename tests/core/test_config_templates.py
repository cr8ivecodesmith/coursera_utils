from __future__ import annotations

from pathlib import Path

import pytest

from study_utils.core import config_templates
from study_utils.core.config_templates import (
    ConfigTemplate,
    ConfigTemplateError,
)


def test_get_template_returns_convert_markdown_template(tmp_path: Path) -> None:
    template = config_templates.get_template("convert_markdown")
    assert isinstance(template, ConfigTemplate)

    contents = template.read_text()
    assert "[paths]" in contents
    assert "extensions" in contents

    target = tmp_path / "config.toml"
    written = template.write(target)
    assert written == target
    assert target.read_text(encoding="utf-8") == contents

    with pytest.raises(ConfigTemplateError):
        template.write(target)

    updated = template.write(target, overwrite=True)
    assert updated == target


def test_iter_templates_returns_registered_templates() -> None:
    names = {template.name for template in config_templates.iter_templates()}
    assert "convert_markdown" in names


@pytest.mark.parametrize("unknown", ["missing", "", "rag"])
def test_get_template_unknown_raises(unknown: str) -> None:
    with pytest.raises(ConfigTemplateError):
        config_templates.get_template(unknown)

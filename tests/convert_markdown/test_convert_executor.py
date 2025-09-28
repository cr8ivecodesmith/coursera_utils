from __future__ import annotations

import logging
from pathlib import Path

import pytest

from study_utils.convert_markdown import config as cfg
from study_utils.convert_markdown import converter
from study_utils.convert_markdown import executor


@pytest.fixture(name="logger")
def _logger_fixture() -> logging.Logger:
    logger = logging.getLogger("study_utils.tests.convert_executor")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def _deps(
    markdown_body: str = "body",
    epub_body: str = "epub",
) -> converter.ConverterDependencies:
    return converter.ConverterDependencies(
        markitdown=lambda _: markdown_body,
        epub=lambda _: epub_body,
    )


def _config(
    tmp_path: Path, extensions: tuple[str, ...]
) -> cfg.ConvertMarkdownConfig:
    return cfg.ConvertMarkdownConfig(
        extensions=extensions,
        output_dir=tmp_path / "out",
        collision=cfg.CollisionPolicy.OVERWRITE,
        log_level="INFO",
    )


def test_run_conversion_processes_directory_in_order(tmp_path, logger):
    source_dir = tmp_path / "inputs"
    nested = source_dir / "nested"
    nested.mkdir(parents=True)

    first = source_dir / "b.docx"
    second = nested / "a.pdf"
    third = nested / "c.epub"
    for path in (first, second, third):
        path.write_text("data", encoding="utf-8")

    config = _config(tmp_path, ("docx", "pdf", "epub"))
    deps = _deps()

    summary = executor.run_conversion(
        [source_dir],
        config=config,
        dependencies=deps,
        logger=logger,
    )

    assert summary.success_count == 3
    assert summary.failure_count == 0
    assert summary.skipped_count == 0
    # Paths are processed in deterministic order using string-based sorting.
    assert summary.processed == tuple(
        sorted(summary.processed, key=lambda path: str(path))
    )
    for outcome in summary.outcomes:
        assert outcome.output_path is not None
        assert outcome.output_path.exists()


def test_run_conversion_skips_extension_not_in_config(tmp_path, logger):
    source = tmp_path / "notes.txt"
    source.write_text("data", encoding="utf-8")

    config = _config(tmp_path, ("pdf",))

    summary = executor.run_conversion(
        [source],
        config=config,
        dependencies=_deps(),
        logger=logger,
    )

    assert summary.success_count == 0
    assert summary.skipped_count == 1
    assert summary.failure_count == 0
    outcome = summary.outcomes[0]
    assert outcome.status is converter.ConversionStatus.SKIPPED
    assert "not enabled" in (outcome.reason or "")


def test_run_conversion_reports_missing_sources(tmp_path, logger):
    missing = tmp_path / "missing.pdf"

    config = _config(tmp_path, ("pdf",))

    summary = executor.run_conversion(
        [missing],
        config=config,
        dependencies=_deps(),
        logger=logger,
    )

    assert summary.success_count == 0
    assert summary.failure_count == 1
    assert missing in summary.processed
    outcome = summary.outcomes[0]
    assert outcome.status is converter.ConversionStatus.FAILED
    assert "not found" in (outcome.reason or "")


def test_run_conversion_deduplicates_inputs(tmp_path, logger):
    source = tmp_path / "file.pdf"
    source.write_text("data", encoding="utf-8")

    config = _config(tmp_path, ("pdf",))

    summary = executor.run_conversion(
        [source, source],
        config=config,
        dependencies=_deps(),
        logger=logger,
    )

    assert summary.success_count == 1
    assert len(summary.processed) == 1
    assert summary.processed[0] == source.resolve()


def test_run_conversion_logs_skip_from_converter(tmp_path, logger):
    source = tmp_path / "doc.pdf"
    source.write_text("content", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    target = output_dir / "doc.md"
    target.write_text("existing", encoding="utf-8")

    config = cfg.ConvertMarkdownConfig(
        extensions=("pdf",),
        output_dir=output_dir,
        collision=cfg.CollisionPolicy.SKIP,
        log_level="INFO",
    )

    summary = executor.run_conversion(
        [source],
        config=config,
        dependencies=_deps(),
        logger=logger,
    )

    assert summary.success_count == 0
    assert summary.skipped_count == 1
    assert summary.outcomes[0].status is converter.ConversionStatus.SKIPPED


def test_normalize_inputs_handles_resolution_failure(
    tmp_path, logger, monkeypatch
):
    original_resolve = Path.resolve

    def fake_resolve(self, strict=False):
        if self.name == "special":
            raise FileNotFoundError("boom")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)

    special = Path("special")
    config = _config(tmp_path, ("pdf",))

    summary = executor.run_conversion(
        [special],
        config=config,
        dependencies=_deps(),
        logger=logger,
    )

    assert summary.failure_count == 1
    outcome = summary.outcomes[0]
    assert outcome.status is converter.ConversionStatus.FAILED


def test_run_conversion_ignores_directory_files_without_extension(
    tmp_path, logger
):
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    (source_dir / "README").write_text("notes", encoding="utf-8")

    config = _config(tmp_path, ("pdf",))

    summary = executor.run_conversion(
        [source_dir],
        config=config,
        dependencies=_deps(),
        logger=logger,
    )

    assert summary.outcomes == ()
    assert summary.processed == ()


from __future__ import annotations

from datetime import datetime, timezone

from study_utils.convert_markdown import output


def test_render_document_with_body_formats_metadata():
    metadata = {
        "source_path": "/tmp/source.pdf",
        "converted_at": datetime(2025, 3, 18, 12, 30, tzinfo=timezone.utc),
        "source_modified_at": datetime(
            2025,
            3,
            17,
            9,
            15,
            tzinfo=timezone.utc,
        ),
    }
    body = "# Heading\nContent line\n"

    document = output.render_document(metadata, body)

    assert document.startswith("---\n")
    assert 'source_path: "/tmp/source.pdf"' in document
    assert 'converted_at: "2025-03-18T12:30:00Z"' in document
    assert 'source_modified_at: "2025-03-17T09:15:00Z"' in document
    assert document.rstrip().endswith("Content line")
    assert document.endswith("\n")


def test_render_document_handles_empty_body():
    metadata = {
        "source_path": "/tmp/source.html",
        "converted_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "source_modified_at": datetime(2024, 12, 31, tzinfo=timezone.utc),
    }

    document = output.render_document(metadata, "")

    assert document == (
        "---\n"
        'source_path: "/tmp/source.html"\n'
        'converted_at: "2025-01-01T00:00:00Z"\n'
        'source_modified_at: "2024-12-31T00:00:00Z"\n'
        "---\n\n"
    )

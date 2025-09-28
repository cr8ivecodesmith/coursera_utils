"""Output helpers for the document-to-Markdown converter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Mapping


def render_document(
    metadata: Mapping[str, datetime | str],
    body: str,
) -> str:
    """Render Markdown with YAML front matter built from ``metadata``."""

    serialized = {
        key: _serialize_value(value)
        for key, value in metadata.items()
    }

    front_matter_lines = ["---"]
    for key, value in serialized.items():
        front_matter_lines.append(
            f"{key}: {json.dumps(value, ensure_ascii=False)}"
        )
    front_matter_lines.append("---")

    front_matter = "\n".join(front_matter_lines)
    normalized_body = body.rstrip("\n")
    if normalized_body:
        return f"{front_matter}\n\n{normalized_body}\n"
    return f"{front_matter}\n\n"


def _serialize_value(value: datetime | str) -> str:
    if isinstance(value, datetime):
        timestamp = value.astimezone(timezone.utc).replace(microsecond=0)
        return timestamp.isoformat().replace("+00:00", "Z")
    return str(value)


__all__ = ["render_document"]

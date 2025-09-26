"""Filesystem helpers shared by tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Union

TreeValue = Union[str, bytes, "Tree", None]
Tree = Mapping[str, TreeValue]


def build_tree(base: Path, tree: Tree) -> None:
    """Create files/directories under ``base`` from a nested mapping.

    ``tree`` maps names to either strings/bytes (file content), ``None``
    (directories), or nested mappings for subdirectories.
    """

    for name, value in tree.items():
        path = base / name
        if isinstance(value, (str, bytes)):
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, bytes):
                path.write_bytes(value)
            else:
                path.write_text(value, encoding="utf-8")
            continue
        if isinstance(value, Mapping):
            path.mkdir(parents=True, exist_ok=True)
            build_tree(path, value)  # type: ignore[arg-type]
            continue
        if value is None:
            path.mkdir(parents=True, exist_ok=True)
            continue
        raise TypeError(f"Unsupported tree value for {path}: {type(value)!r}")


@dataclass
class WorkspaceBuilder:
    """Helper bound to a tmp directory for concise tree creation."""

    root: Path

    def create(self, tree: Tree) -> Path:
        build_tree(self.root, tree)
        return self.root

    def write(
        self, relative: Union[str, Path], content: Union[str, bytes]
    ) -> Path:
        path = self.root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def files(self, pattern: str = "**/*") -> Iterator[Path]:
        return (p for p in self.root.glob(pattern) if p.is_file())

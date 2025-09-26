"""Session persistence for the Study RAG chat runtime (Milestone 3)."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

__all__ = [
    "SessionError",
    "ChatMessage",
    "ChatSession",
    "SessionStore",
]


_SESSION_FILENAME = "session.json"
_LOCK_FILENAME = ".lock"
_LOCK_TIMEOUT_SECONDS = 5.0


class SessionError(RuntimeError):
    """Raised when session persistence or validation fails."""


@dataclass(frozen=True)
class ChatMessage:
    """A single chat transcript entry."""

    role: str
    content: str
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChatMessage":
        role = str(payload.get("role"))
        content = str(payload.get("content"))
        created_at = str(payload.get("created_at"))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SessionError(
                "Chat message metadata must be a mapping when present."
            )
        return cls(
            role=role,
            content=content,
            created_at=created_at,
            metadata=dict(metadata),
        )


@dataclass
class ChatSession:
    """In-memory representation of a chat session."""

    session_id: str
    directory: Path
    created_at: str
    updated_at: str
    vector_dbs: tuple[str, ...]
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    chat_model: str
    messages: list[ChatMessage]

    def add_message(
        self,
        role: str,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a message to the transcript and update timestamps."""

        timestamp = _timestamp()
        entry = ChatMessage(
            role=role,
            content=content,
            created_at=timestamp,
            metadata=dict(metadata or {}),
        )
        self.messages.append(entry)
        self.updated_at = timestamp

    def enforce_history_limit(self, max_messages: int) -> None:
        """Trim the transcript to the most recent ``max_messages`` entries."""

        if max_messages <= 0:
            return
        if len(self.messages) <= max_messages:
            return
        excess = len(self.messages) - max_messages
        del self.messages[0:excess]
        self.updated_at = _timestamp()

    def merge_vector_dbs(self, names: tuple[str, ...]) -> None:
        """Add new vector DB names, keeping the tuple sorted and unique."""

        merged = set(self.vector_dbs)
        merged.update(names)
        self.vector_dbs = tuple(sorted(merged))
        self.updated_at = _timestamp()

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "vector_dbs": list(self.vector_dbs),
            "embedding": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "dimension": self.embedding_dimension,
            },
            "chat_model": self.chat_model,
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(
        cls, directory: Path, payload: Mapping[str, Any]
    ) -> "ChatSession":
        try:
            session_id = str(payload["session_id"])
            created_at = str(payload["created_at"])
            updated_at = str(payload["updated_at"])
            vector_dbs = tuple(str(name) for name in payload["vector_dbs"])
            embedding = payload["embedding"]
            embedding_provider = str(embedding["provider"])
            embedding_model = str(embedding["model"])
            embedding_dimension = int(embedding["dimension"])
            chat_model = str(payload["chat_model"])
            messages = [
                ChatMessage.from_dict(item)
                for item in payload.get("messages", [])
            ]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise SessionError(
                f"Session payload missing required field: {exc}"
            ) from exc
        return cls(
            session_id=session_id,
            directory=directory,
            created_at=created_at,
            updated_at=updated_at,
            vector_dbs=tuple(sorted(vector_dbs)),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chat_model=chat_model,
            messages=messages,
        )


class SessionStore:
    """Manage persistence for chat sessions under the sessions directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, session_id: str) -> Path:
        return self._root / session_id

    def create(
        self,
        *,
        vector_dbs: tuple[str, ...],
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        chat_model: str,
    ) -> ChatSession:
        session_id = _generate_session_id()
        directory = self.path_for(session_id)
        tries = 0
        while directory.exists():  # pragma: no cover - extremely unlikely
            session_id = _generate_session_id()
            directory = self.path_for(session_id)
            tries += 1
            if tries > 5:
                raise SessionError(
                    "Failed to allocate unique session directory."
                )
        directory.mkdir(parents=True, exist_ok=False)
        created = _timestamp()
        session = ChatSession(
            session_id=session_id,
            directory=directory,
            created_at=created,
            updated_at=created,
            vector_dbs=tuple(sorted(vector_dbs)),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chat_model=chat_model,
            messages=[],
        )
        self.save(session)
        return session

    def load(self, session_id: str) -> ChatSession:
        directory = self.path_for(session_id)
        payload = self._read_payload(directory)
        return ChatSession.from_dict(directory, payload)

    def save(self, session: ChatSession) -> None:
        directory = session.directory
        if not directory.exists():
            raise SessionError(
                f"Session directory missing on save: {directory}"
            )
        payload = session.to_dict()
        target = directory / _SESSION_FILENAME
        lock_path = directory / _LOCK_FILENAME
        with _SessionLock(lock_path):
            _atomic_write_json(target, payload)

    def _read_payload(self, directory: Path) -> Mapping[str, Any]:
        target = directory / _SESSION_FILENAME
        if not target.is_file():
            raise SessionError(f"Session file missing: {target}")
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SessionError(
                f"Failed to parse session file: {target}"
            ) from exc


class _SessionLock:
    """Simple filesystem lock using exclusive file creation."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def __enter__(self) -> "_SessionLock":
        deadline = time.time() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fd = os.open(
                    self._path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.close(fd)
                break
            except FileExistsError:
                if time.time() > deadline:
                    raise SessionError(
                        f"Timed out waiting for session lock: {self._path}"
                    )
                time.sleep(0.05)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self._path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        encoding="utf-8",
        dir=str(path.parent),
    )
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
    try:
        path.chmod(0o600)
    except PermissionError:  # pragma: no cover - depends on filesystem
        pass


def _generate_session_id() -> str:
    return uuid.uuid4().hex


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

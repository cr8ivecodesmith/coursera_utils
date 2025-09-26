from __future__ import annotations

import pytest

from study_utils.rag import session


def test_session_store_roundtrip(tmp_path):
    store = session.SessionStore(tmp_path / "sessions")
    chat_session = store.create(
        vector_dbs=("physics",),
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimension=2,
        chat_model="gpt-4o-mini",
    )

    chat_session.add_message("user", "Hello", metadata={"foo": "bar"})
    chat_session.add_message("assistant", "Hi there")
    chat_session.merge_vector_dbs(("math",))
    chat_session.enforce_history_limit(1)

    store.save(chat_session)

    loaded = store.load(chat_session.session_id)
    assert loaded.vector_dbs == ("math", "physics")
    assert len(loaded.messages) == 1
    assert loaded.messages[0].role == "assistant"
    entry = loaded.messages[0]
    reconstructed = session.ChatMessage.from_dict(entry.to_dict())
    assert reconstructed.content == entry.content


def test_chat_message_from_dict_requires_mapping_metadata():
    payload = {
        "role": "user",
        "content": "hello",
        "created_at": "now",
        "metadata": "oops",
    }
    with pytest.raises(session.SessionError):
        session.ChatMessage.from_dict(payload)


def test_enforce_history_limit_zero_noop(tmp_path):
    chat_session = session.ChatSession(
        session_id="sess",
        directory=tmp_path,
        created_at="now",
        updated_at="now",
        vector_dbs=("db",),
        embedding_provider="openai",
        embedding_model="model",
        embedding_dimension=2,
        chat_model="gpt",
        messages=[
            session.ChatMessage(
                role="user",
                content="hi",
                created_at="now",
                metadata={},
            )
        ],
    )
    chat_session.enforce_history_limit(0)
    assert len(chat_session.messages) == 1


def test_session_store_root_property(tmp_path):
    store = session.SessionStore(tmp_path / "sessions")
    assert store.root == tmp_path / "sessions"


def test_session_store_save_requires_directory(tmp_path):
    store = session.SessionStore(tmp_path / "sessions")
    chat_session = session.ChatSession(
        session_id="sess",
        directory=tmp_path / "missing",
        created_at="now",
        updated_at="now",
        vector_dbs=("db",),
        embedding_provider="openai",
        embedding_model="model",
        embedding_dimension=2,
        chat_model="gpt",
        messages=[],
    )
    with pytest.raises(session.SessionError):
        store.save(chat_session)


def test_read_payload_handles_missing_and_invalid(tmp_path):
    store = session.SessionStore(tmp_path / "sessions")
    target_dir = tmp_path / "sessions" / "sess"
    target_dir.mkdir(parents=True)
    with pytest.raises(session.SessionError):
        store._read_payload(target_dir)

    payload_path = target_dir / "session.json"
    payload_path.write_text("not json", encoding="utf-8")
    with pytest.raises(session.SessionError):
        store._read_payload(target_dir)


def test_session_lock_retries(monkeypatch, tmp_path):
    lock_path = tmp_path / ".lock"
    lock = session._SessionLock(lock_path)
    call_count = {"count": 0}
    real_open = session.os.open

    def fake_open(path, flags, *args):  # noqa: D401, ANN001
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise FileExistsError
        return real_open(path, flags, *args)

    monkeypatch.setattr(session.os, "open", fake_open)
    monkeypatch.setattr(session.time, "sleep", lambda _: None)

    with lock:
        pass

    assert call_count["count"] == 2
    assert not lock_path.exists()


def test_session_lock_times_out(monkeypatch, tmp_path):
    lock_path = tmp_path / ".lock"
    lock = session._SessionLock(lock_path)

    def always_fail(*args, **kwargs):  # noqa: D401, ANN001
        raise FileExistsError

    monkeypatch.setattr(session.os, "open", always_fail)
    monkeypatch.setattr(session.time, "sleep", lambda _: None)

    times = iter([0.0, 6.0])
    monkeypatch.setattr(session.time, "time", lambda: next(times))

    with pytest.raises(session.SessionError):
        with lock:
            pass

    assert not lock_path.exists()

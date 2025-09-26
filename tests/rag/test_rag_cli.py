from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pytest

from study_utils import cli as root_cli
from study_utils.rag import cli as rag_cli
from study_utils.rag import config as config_mod
from study_utils.rag import vector_store
from study_utils.rag import chat as chat_mod


class CLIStubEmbedder:
    def embed_documents(self, texts):
        return [[float(len(text)), 0.0] for text in texts]


class CLIStubChatClient:
    def __init__(self):
        self.calls = []

    def complete(
        self,
        *,
        model,
        messages,
        temperature,
        max_tokens,
        stream,
        timeout,
    ) -> str:
        self.calls.append([dict(item) for item in messages])
        return "CLI stub response"


@pytest.fixture()
def rag_cli_env(tmp_path, monkeypatch):
    data_home = tmp_path / "data"
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(data_home))
    config_mod.write_template(data_home / "config" / "rag.toml")
    monkeypatch.setattr(
        rag_cli,
        "_build_backend",
        lambda: vector_store.InMemoryVectorStoreBackend(),
    )
    monkeypatch.setattr(
        rag_cli,
        "_build_embedder",
        lambda cfg: CLIStubEmbedder(),
    )
    return data_home


def test_rag_command_registered():
    assert "rag" in root_cli.COMMANDS


def test_config_init_creates_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    target = tmp_path / "custom" / "rag.toml"
    exit_code = rag_cli.main(["config", "init", "--path", str(target)])

    assert exit_code == 0
    assert target.exists()
    out = capsys.readouterr().out
    assert "Wrote config template" in out


def test_config_init_requires_force(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    target = tmp_path / "rag.toml"
    config_mod.write_template(target)

    exit_code = rag_cli.main(["config", "init", "--path", str(target)])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Config already exists" in err


def test_config_validate_reports_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "dh"))
    target = tmp_path / "rag.toml"
    config_mod.write_template(target)

    exit_code = rag_cli.main(["config", "validate", "--path", str(target)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Configuration OK" in output
    assert "chunk_tokens" in output


def test_config_validate_propagates_errors(tmp_path, capsys):
    target = tmp_path / "bad.toml"
    target.write_text("not valid", encoding="utf-8")

    exit_code = rag_cli.main(["config", "validate", "--path", str(target)])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Failed to parse" in err


def test_config_path_prints_resolved_value(tmp_path, capsys):
    target = tmp_path / "custom" / "rag.toml"
    exit_code = rag_cli.main(["config", "path", "--path", str(target)])

    assert exit_code == 0
    output = capsys.readouterr().out.strip()
    assert Path(output) == target.resolve()


def test_config_init_uses_default_path(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    exit_code = rag_cli.main(["config", "init", "--force"])
    assert exit_code == 0


def test_handle_config_unknown_command():
    with pytest.raises(RuntimeError):
        rag_cli._handle_config(argparse.Namespace(config_command="oops"))


def test_config_path_handles_resolve_error(monkeypatch, capsys):
    def raise_error(**kwargs):  # noqa: D401, ARG001
        raise config_mod.ConfigError("boom")

    monkeypatch.setattr(config_mod, "resolve_config_path", raise_error)
    exit_code = rag_cli.main(["config", "path"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "boom" in err


def test_main_reports_unhandled_command(monkeypatch):
    class DummyParser:
        def __init__(self):
            self.error_message = None

        def parse_args(self, argv=None):  # noqa: ARG002
            return argparse.Namespace(command="todo")

        def error(self, message):
            self.error_message = message

    dummy = DummyParser()
    monkeypatch.setattr(rag_cli, "_build_parser", lambda: dummy)

    exit_code = rag_cli.main(["todo"])
    assert exit_code == 2
    assert dummy.error_message == "Command not implemented yet."


def test_to_path_none_returns_none():
    assert rag_cli._to_path(None) is None


def test_ingest_command_creates_vector_store(tmp_path, rag_cli_env, capsys):
    source = tmp_path / "doc.txt"
    source.write_text("sample text", encoding="utf-8")

    exit_code = rag_cli.main(["ingest", "--name", "physics", str(source)])

    assert exit_code == 0
    manifest = rag_cli_env / "rag_dbs" / "physics" / "manifest.json"
    assert manifest.exists()
    out = capsys.readouterr().out
    assert "Vector database 'physics' ready." in out


def test_ingest_requires_force_for_existing(tmp_path, rag_cli_env, capsys):
    source = tmp_path / "doc.txt"
    source.write_text("sample text", encoding="utf-8")

    first = rag_cli.main(["ingest", "--name", "physics", str(source)])
    assert first == 0
    second = rag_cli.main(["ingest", "--name", "physics", str(source)])
    assert second == 2
    err = capsys.readouterr().err
    assert "Use --force" in err


def test_list_export_import_delete_flow(tmp_path, rag_cli_env, capsys):
    source = tmp_path / "doc.txt"
    source.write_text("sample text", encoding="utf-8")
    assert rag_cli.main(["ingest", "--name", "physics", str(source)]) == 0

    assert rag_cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "physics" in out

    archive = tmp_path / "physics.zip"
    assert (
        rag_cli.main(
            [
                "export",
                "--name",
                "physics",
                "--out",
                str(archive),
            ]
        )
        == 0
    )
    assert archive.exists()

    assert rag_cli.main(["delete", "--name", "physics"]) == 0
    store_dir = rag_cli_env / "rag_dbs" / "physics"
    assert not store_dir.exists()

    assert (
        rag_cli.main(
            [
                "import",
                "--name",
                "physics",
                "--archive",
                str(archive),
            ]
        )
        == 0
    )
    assert store_dir.exists()


def test_inspect_command_prints_manifest(tmp_path, rag_cli_env, capsys):
    source = tmp_path / "doc.txt"
    source.write_text("sample text", encoding="utf-8")
    assert rag_cli.main(["ingest", "--name", "physics", str(source)]) == 0
    capsys.readouterr()

    exit_code = rag_cli.main(["inspect", "--name", "physics"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Name: physics" in output
    assert "Documents:" in output
    assert str(source) in output


def test_inspect_handles_empty_documents(monkeypatch, capsys):
    manifest = vector_store.build_manifest(
        name="physics",
        embedding=vector_store.EmbeddingMetadata(
            provider="openai",
            model="test",
            dimension=2,
        ),
        chunking=vector_store.ChunkingMetadata(
            tokenizer="tiktoken",
            encoding="cl100k_base",
            tokens_per_chunk=10,
            token_overlap=0,
            fallback_delimiter="\n\n",
        ),
        dedupe=vector_store.DedupMetadata(
            strategy="checksum",
            checksum_algorithm="sha256",
        ),
        documents=(),
    )

    class StubRepo:
        def load_manifest(self, name):  # noqa: D401
            assert name == "physics"
            return manifest

    monkeypatch.setattr(rag_cli, "_build_repository", lambda: StubRepo())

    exit_code = rag_cli.main(["inspect", "--name", "physics"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Documents: 0" in out
    assert "- " not in out


def test_ingest_handles_config_error(monkeypatch, capsys):
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: (_ for _ in ()).throw(config_mod.ConfigError("bad")),
    )
    exit_code = rag_cli.main(["ingest", "--name", "physics", "file.txt"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "bad" in err


def test_ingest_handles_runtime_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    config_mod.write_template(tmp_path / "data" / "config" / "rag.toml")
    monkeypatch.setattr(rag_cli, "_build_backend", lambda: None)

    def raise_runtime(*args, **kwargs):  # noqa: D401, ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(rag_cli, "_build_embedder", raise_runtime)
    exit_code = rag_cli.main(["ingest", "--name", "physics", "file.txt"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "boom" in err


def test_list_handles_empty(monkeypatch, capsys):
    monkeypatch.setattr(
        rag_cli,
        "_build_repository",
        lambda: vector_store.VectorStoreRepository(Path("unused")),
    )
    exit_code = rag_cli.main(["list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No vector databases" in out


def test_inspect_handles_missing_manifest(monkeypatch, capsys):
    repo = vector_store.VectorStoreRepository(Path("unused"))
    monkeypatch.setattr(rag_cli, "_build_repository", lambda: repo)

    exit_code = rag_cli.main(["inspect", "--name", "physics"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Manifest missing" in err


def test_delete_missing_store(monkeypatch, capsys):
    repo = vector_store.VectorStoreRepository(Path("unused"))
    monkeypatch.setattr(rag_cli, "_build_repository", lambda: repo)
    exit_code = rag_cli.main(["delete", "--name", "physics"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "does not exist" in err


def test_export_missing_store(monkeypatch, capsys):
    repo = vector_store.VectorStoreRepository(Path("unused"))
    monkeypatch.setattr(rag_cli, "_build_repository", lambda: repo)
    exit_code = rag_cli.main(
        ["export", "--name", "physics", "--out", "archive.zip"]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "does not exist" in err or "missing" in err


def test_import_missing_archive(monkeypatch, capsys):
    repo = vector_store.VectorStoreRepository(Path("unused"))
    monkeypatch.setattr(rag_cli, "_build_repository", lambda: repo)
    exit_code = rag_cli.main(
        [
            "import",
            "--name",
            "physics",
            "--archive",
            "missing.zip",
        ]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "Archive not found" in err


def test_build_embedder_requires_openai_provider():
    cfg = types.SimpleNamespace(
        providers=types.SimpleNamespace(default="azure")
    )
    with pytest.raises(vector_store.VectorStoreError):
        rag_cli._build_embedder(cfg)


def test_select_embedding_model_rejects_unknown():
    cfg = types.SimpleNamespace(
        providers=types.SimpleNamespace(default="azure")
    )
    with pytest.raises(vector_store.VectorStoreError):
        rag_cli._select_embedding_model(cfg)


def test_helper_builders_use_config(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    config_mod.write_template(tmp_path / "data" / "config" / "rag.toml")
    cfg = config_mod.load_config()
    backend = rag_cli._build_backend()
    assert isinstance(backend, vector_store.FaissVectorStoreBackend)
    chunker = rag_cli._build_chunker(cfg)
    assert chunker.chunk("text")
    dedupe = rag_cli._build_dedupe(cfg)
    assert dedupe.checksum_algorithm == "sha256"
    chunking = rag_cli._build_chunking(cfg)
    assert chunking.fallback_delimiter


def test_build_embedder_creates_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(tmp_path / "data"))
    config_mod.write_template(tmp_path / "data" / "config" / "rag.toml")
    cfg = config_mod.load_config()
    created = {}

    class DummyAdapter:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(
        rag_cli.ingest_mod,
        "OpenAIEmbeddingClient",
        DummyAdapter,
    )
    adapter = rag_cli._build_embedder(cfg)
    assert isinstance(adapter, DummyAdapter)
    assert created["model"] == cfg.providers.openai.embedding_model


def test_print_chat_answer_handles_no_context(capsys):
    answer = chat_mod.ChatAnswer(
        session_id="s",
        prompt="p",
        response="No context",
        contexts=[],
    )
    rag_cli._print_chat_answer(answer)
    output = capsys.readouterr().out
    assert "No retrieval context" in output


def test_chat_command_question(tmp_path, rag_cli_env, monkeypatch, capsys):
    source = tmp_path / "doc.txt"
    source.write_text("sample text", encoding="utf-8")
    assert rag_cli.main(["ingest", "--name", "physics", str(source)]) == 0
    capsys.readouterr()

    stub_client = CLIStubChatClient()
    monkeypatch.setattr(rag_cli, "_build_chat_client", lambda cfg: stub_client)

    exit_code = rag_cli.main(
        ["chat", "--db", "physics", "--question", "Explain this?"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "CLI stub response" in output
    sessions_dir = rag_cli_env / "rag_sessions"
    assert any(sessions_dir.iterdir())


def test_chat_command_resume_merges_new_db(
    tmp_path, rag_cli_env, monkeypatch, capsys
):
    src1 = tmp_path / "doc1.txt"
    src1.write_text("first", encoding="utf-8")
    src2 = tmp_path / "doc2.txt"
    src2.write_text("second", encoding="utf-8")

    assert rag_cli.main(["ingest", "--name", "physics", str(src1)]) == 0
    assert rag_cli.main(["ingest", "--name", "algebra", str(src2)]) == 0
    capsys.readouterr()

    stub_client = CLIStubChatClient()
    monkeypatch.setattr(rag_cli, "_build_chat_client", lambda cfg: stub_client)

    assert (
        rag_cli.main(["chat", "--db", "physics", "--question", "First?"]) == 0
    )
    capsys.readouterr()

    sessions_dir = rag_cli_env / "rag_sessions"
    session_dirs = sorted(sessions_dir.iterdir())
    assert session_dirs
    session_id = session_dirs[0].name

    assert (
        rag_cli.main(
            [
                "chat",
                "--resume",
                session_id,
                "--db",
                "algebra",
                "--question",
                "Second?",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "CLI stub response" in output
    payload = json.loads((session_dirs[0] / "session.json").read_text())
    assert sorted(payload["vector_dbs"]) == ["algebra", "physics"]


def test_chat_command_requires_db(monkeypatch, capsys, rag_cli_env):
    stub_client = CLIStubChatClient()
    monkeypatch.setattr(rag_cli, "_build_chat_client", lambda cfg: stub_client)
    exit_code = rag_cli.main(["chat", "--question", "Hi?"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "At least one --db" in err


def test_handle_chat_handles_config_error(monkeypatch, capsys):
    def raise_config():  # noqa: D401
        raise config_mod.ConfigError("bad config")

    monkeypatch.setattr(config_mod, "load_config", raise_config)
    args = argparse.Namespace(dbs=None, resume=None, question=None)
    exit_code = rag_cli._handle_chat(args)
    assert exit_code == 2
    assert "bad config" in capsys.readouterr().err


def _prepare_chat_environment(tmp_path, monkeypatch):
    data_home = tmp_path / "data"
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(data_home))
    config_mod.write_template(data_home / "config" / "rag.toml")
    return config_mod.load_config()


def test_handle_chat_handles_embedder_error(tmp_path, monkeypatch, capsys):
    _prepare_chat_environment(tmp_path, monkeypatch)

    def fail_embedder(cfg):  # noqa: D401, ANN001
        raise vector_store.VectorStoreError("no embed")

    monkeypatch.setattr(rag_cli, "_build_embedder", fail_embedder)
    args = argparse.Namespace(dbs=None, resume=None, question=None)
    exit_code = rag_cli._handle_chat(args)
    assert exit_code == 2
    assert "no embed" in capsys.readouterr().err


def test_handle_chat_handles_chat_client_error(tmp_path, monkeypatch, capsys):
    _prepare_chat_environment(tmp_path, monkeypatch)

    monkeypatch.setattr(
        rag_cli, "_build_embedder", lambda cfg: CLIStubEmbedder()
    )

    def fail_client(cfg):  # noqa: D401
        raise RuntimeError("client boom")

    monkeypatch.setattr(rag_cli, "_build_chat_client", fail_client)
    args = argparse.Namespace(dbs=None, resume=None, question=None)
    exit_code = rag_cli._handle_chat(args)
    assert exit_code == 2
    assert "client boom" in capsys.readouterr().err


def test_handle_chat_handles_prepare_session_error(
    tmp_path, monkeypatch, capsys
):
    _prepare_chat_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        rag_cli, "_build_embedder", lambda cfg: CLIStubEmbedder()
    )
    monkeypatch.setattr(
        rag_cli, "_build_chat_client", lambda cfg: CLIStubChatClient()
    )

    def raise_prepare(self, **kwargs):  # noqa: D401, ANN001
        raise chat_mod.ChatError("prep failed")

    monkeypatch.setattr(chat_mod.ChatRuntime, "prepare_session", raise_prepare)
    args = argparse.Namespace(dbs=None, resume=None, question=None)
    exit_code = rag_cli._handle_chat(args)
    assert exit_code == 2
    assert "prep failed" in capsys.readouterr().err


def test_handle_chat_reports_question_error(tmp_path, monkeypatch, capsys):
    _prepare_chat_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        rag_cli, "_build_embedder", lambda cfg: CLIStubEmbedder()
    )
    monkeypatch.setattr(
        rag_cli, "_build_chat_client", lambda cfg: CLIStubChatClient()
    )

    class DummyRuntime(chat_mod.ChatRuntime):
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(chat_mod, "ChatRuntime", DummyRuntime)

    def return_session(self, resume_id=None, vector_dbs=()):  # noqa: D401, ANN001
        return types.SimpleNamespace()

    def raise_ask(self, sess, question):  # noqa: D401, ANN001
        raise chat_mod.ChatError("ask nope")

    monkeypatch.setattr(DummyRuntime, "prepare_session", return_session)
    monkeypatch.setattr(DummyRuntime, "ask", raise_ask)

    args = argparse.Namespace(dbs=None, resume=None, question="hi")
    exit_code = rag_cli._handle_chat(args)
    assert exit_code == 2
    assert "ask nope" in capsys.readouterr().err


def test_handle_chat_invokes_interactive_loop(tmp_path, monkeypatch):
    _prepare_chat_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        rag_cli, "_build_embedder", lambda cfg: CLIStubEmbedder()
    )
    monkeypatch.setattr(
        rag_cli, "_build_chat_client", lambda cfg: CLIStubChatClient()
    )

    called = {}

    class DummyRuntime(chat_mod.ChatRuntime):
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
            super().__init__(*args, **kwargs)

    def return_session(self, resume_id=None, vector_dbs=()):  # noqa: D401, ANN001
        return types.SimpleNamespace()

    def interactive(self, sess, console=None):  # noqa: D401, ANN001
        called["interactive"] = True

    monkeypatch.setattr(chat_mod, "ChatRuntime", DummyRuntime)
    monkeypatch.setattr(DummyRuntime, "prepare_session", return_session)
    monkeypatch.setattr(DummyRuntime, "ask", lambda self, sess, q: None)
    monkeypatch.setattr(DummyRuntime, "interactive_loop", interactive)

    args = argparse.Namespace(dbs=None, resume=None, question=None)
    exit_code = rag_cli._handle_chat(args)
    assert exit_code == 0
    assert called["interactive"] is True


def test_build_chat_client_uses_openai_settings(tmp_path, monkeypatch):
    cfg = _prepare_chat_environment(tmp_path, monkeypatch)
    captured = {}

    def factory(**kwargs):  # noqa: D401, ANN001
        captured.update(kwargs)
        return "client"

    monkeypatch.setattr(chat_mod, "OpenAIChatClient", factory)
    client = rag_cli._build_chat_client(cfg)
    assert client == "client"
    assert captured["model"] == cfg.providers.openai.chat_model
    assert captured["temperature"] == cfg.providers.openai.temperature
    assert captured["api_base"] == cfg.providers.openai.api_base

from __future__ import annotations

import argparse
import types
from pathlib import Path

import pytest

from study_utils import cli as root_cli
from study_utils.rag import cli as rag_cli
from study_utils.rag import config as config_mod
from study_utils.rag import vector_store


class CLIStubEmbedder:
    def embed_documents(self, texts):
        return [[float(len(text)), 0.0] for text in texts]


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

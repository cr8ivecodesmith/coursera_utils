"""Command-line entry points for the Study RAG workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import config as config_mod
from . import data_dir
from . import ingest as ingest_mod
from . import vector_store


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study rag",
        description="Manage Study RAG configuration and resources.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser(
        "config",
        help="Manage Study RAG configuration files.",
    )
    _build_config_subcommands(config_parser)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest study materials into a named vector database.",
    )
    ingest_parser.add_argument(
        "--name",
        required=True,
        help="Logical name for the vector database (letters, numbers, -,_).",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing vector database with the same name.",
    )
    ingest_parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to ingest (recursively when directories).",
    )

    subparsers.add_parser(
        "list",
        help="List available vector databases.",
    )

    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a vector database from the data directory.",
    )
    delete_parser.add_argument(
        "--name",
        required=True,
        help="Name of the vector database to remove.",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Export a vector database as a portable archive.",
    )
    export_parser.add_argument(
        "--name",
        required=True,
        help="Name of the vector database to export.",
    )
    export_parser.add_argument(
        "--out",
        required=True,
        help="Destination zip archive path.",
    )

    import_parser = subparsers.add_parser(
        "import",
        help="Import a vector database from an archive.",
    )
    import_parser.add_argument(
        "--name",
        required=True,
        help="Target name for the imported vector database.",
    )
    import_parser.add_argument(
        "--archive",
        required=True,
        help="Source zip archive path.",
    )
    import_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing database with the same name.",
    )

    return parser


def _build_config_subcommands(parent: argparse.ArgumentParser) -> None:
    subparsers = parent.add_subparsers(dest="config_command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Write the default configuration template.",
    )
    init_parser.add_argument(
        "--path",
        type=str,
        help=(
            "Optional destination for the config TOML (defaults to data home)."
        ),
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file if present.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the active configuration file.",
    )
    validate_parser.add_argument(
        "--path",
        type=str,
        help="Path to the config TOML (defaults to resolved data home).",
    )
    validate_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress success output; errors still print to stderr.",
    )

    path_parser = subparsers.add_parser(
        "path",
        help="Print the resolved config path.",
    )
    path_parser.add_argument(
        "--path",
        type=str,
        help="Optional path override to resolve/normalise.",
    )


def _to_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _handle_config(args: argparse.Namespace) -> int:
    command = args.config_command
    if command == "init":
        return _handle_config_init(args)
    if command == "validate":
        return _handle_config_validate(args)
    if command == "path":
        return _handle_config_path(args)
    raise RuntimeError(f"Unhandled config command: {command}")


def _handle_config_init(args: argparse.Namespace) -> int:
    explicit_path = _to_path(args.path)
    try:
        target = config_mod.resolve_config_path(explicit_path=explicit_path)
        config_mod.write_template(target, overwrite=args.force)
    except config_mod.ConfigError as exc:
        _print_error(str(exc))
        return 2
    print(f"Wrote config template to {target}")
    return 0


def _handle_config_validate(args: argparse.Namespace) -> int:
    explicit_path = _to_path(args.path)
    try:
        cfg = config_mod.load_config(explicit_path=explicit_path)
    except config_mod.ConfigError as exc:
        _print_error(str(exc))
        return 2
    if not args.quiet:
        print("Configuration OK")
        print(f"  data_home: {cfg.data_home}")
        provider = cfg.providers.openai
        print(f"  chat_model: {provider.chat_model}")
        print(f"  embedding_model: {provider.embedding_model}")
        print(f"  chunk_tokens: {cfg.ingestion.chunking.tokens_per_chunk}")
    return 0


def _handle_config_path(args: argparse.Namespace) -> int:
    explicit_path = _to_path(args.path)
    try:
        path = config_mod.resolve_config_path(explicit_path=explicit_path)
    except config_mod.ConfigError as exc:
        _print_error(str(exc))
        return 2
    print(path)
    return 0


def _handle_ingest(args: argparse.Namespace) -> int:
    try:
        cfg = config_mod.load_config()
    except config_mod.ConfigError as exc:
        _print_error(str(exc))
        return 2

    repo = _build_repository()
    backend = _build_backend()
    try:
        embedder = _build_embedder(cfg)
        chunker = _build_chunker(cfg)
        report = ingest_mod.ingest_sources(
            args.name,
            inputs=[Path(p) for p in args.paths],
            repository=repo,
            backend=backend,
            embedder=embedder,
            chunker=chunker,
            embedding_provider=cfg.providers.default,
            embedding_model=_select_embedding_model(cfg),
            dedupe=_build_dedupe(cfg),
            chunking=_build_chunking(cfg),
            overwrite=args.force,
        )
    except vector_store.VectorStoreError as exc:
        _print_error(str(exc))
        return 2
    except RuntimeError as exc:
        _print_error(str(exc))
        return 2

    manifest_path = repo.manifest_path(report.name)
    _print_ingest_report(report, manifest_path)
    return 0


def _handle_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    repo = _build_repository()
    manifests = repo.list_manifests()
    if not manifests:
        print("No vector databases found.")
        return 0
    name_width = max(len(item.name) for item in manifests)
    print("Name".ljust(name_width), "Created", "Docs", "Chunks", "Model")
    for manifest in manifests:
        docs = len(manifest.documents)
        line = "{name:<{width}} {created} {docs:>4} {chunks:>6} {model}".format(
            name=manifest.name,
            width=name_width,
            created=manifest.created_at,
            docs=docs,
            chunks=manifest.total_chunks,
            model=manifest.embedding.model,
        )
        print(line)
    return 0


def _handle_delete(args: argparse.Namespace) -> int:
    repo = _build_repository()
    try:
        repo.delete(args.name)
    except vector_store.VectorStoreError as exc:
        _print_error(str(exc))
        return 2
    print(f"Deleted vector database '{args.name}'.")
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    repo = _build_repository()
    destination = Path(args.out).expanduser().resolve()
    try:
        repo.export_store(args.name, destination)
    except vector_store.VectorStoreError as exc:
        _print_error(str(exc))
        return 2
    print(
        "Exported vector database '{0}' to {1}.".format(
            args.name,
            destination,
        )
    )
    return 0


def _handle_import(args: argparse.Namespace) -> int:
    repo = _build_repository()
    archive = Path(args.archive).expanduser().resolve()
    try:
        manifest = repo.import_store(
            args.name,
            archive,
            overwrite=args.force,
        )
    except vector_store.VectorStoreError as exc:
        _print_error(str(exc))
        return 2
    print(
        "Imported vector database '{0}' ({1} chunks).".format(
            manifest.name,
            manifest.total_chunks,
        )
    )
    return 0


def _build_repository() -> vector_store.VectorStoreRepository:
    root = data_dir.vector_db_dir()
    return vector_store.VectorStoreRepository(root)


def _build_backend() -> vector_store.VectorStoreBackend:
    return vector_store.FaissVectorStoreBackend()


def _build_embedder(cfg: config_mod.RagConfig) -> ingest_mod.EmbeddingClient:
    provider = cfg.providers.default.lower()
    if provider != "openai":
        raise vector_store.VectorStoreError(
            "Only the 'openai' provider is supported for embeddings."
        )
    openai_cfg = cfg.providers.openai
    return ingest_mod.OpenAIEmbeddingClient(
        model=openai_cfg.embedding_model,
        api_base=openai_cfg.api_base,
        request_timeout=openai_cfg.request_timeout_seconds,
    )


def _build_chunker(cfg: config_mod.RagConfig) -> ingest_mod.TextChunker:
    chunk_cfg = cfg.ingestion.chunking
    return ingest_mod.TextChunker(
        tokenizer=chunk_cfg.tokenizer,
        encoding=chunk_cfg.encoding,
        tokens_per_chunk=chunk_cfg.tokens_per_chunk,
        token_overlap=chunk_cfg.token_overlap,
        fallback_delimiter=chunk_cfg.fallback_delimiter,
    )


def _build_dedupe(cfg: config_mod.RagConfig) -> vector_store.DedupMetadata:
    dedupe_cfg = cfg.ingestion.dedupe
    return vector_store.DedupMetadata(
        strategy=dedupe_cfg.strategy,
        checksum_algorithm=dedupe_cfg.checksum_algorithm,
    )


def _build_chunking(cfg: config_mod.RagConfig) -> vector_store.ChunkingMetadata:
    chunk_cfg = cfg.ingestion.chunking
    return vector_store.ChunkingMetadata(
        tokenizer=chunk_cfg.tokenizer,
        encoding=chunk_cfg.encoding,
        tokens_per_chunk=chunk_cfg.tokens_per_chunk,
        token_overlap=chunk_cfg.token_overlap,
        fallback_delimiter=chunk_cfg.fallback_delimiter,
    )


def _select_embedding_model(cfg: config_mod.RagConfig) -> str:
    provider = cfg.providers.default.lower()
    if provider == "openai":
        return cfg.providers.openai.embedding_model
    raise vector_store.VectorStoreError(
        f"Unknown embedding provider '{cfg.providers.default}'."
    )


def _print_ingest_report(
    report: ingest_mod.IngestionReport,
    manifest_path: Path,
) -> None:
    print(f"Vector database '{report.name}' ready.")
    print(f"  documents ingested: {report.documents_ingested}")
    print(f"  documents skipped: {report.documents_skipped}")
    print(f"  chunks ingested: {report.chunks_ingested}")
    print(f"  manifest: {manifest_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # pragma: no cover - argparse already handles
        return int(exc.code)

    handlers = {
        "config": _handle_config,
        "ingest": _handle_ingest,
        "list": _handle_list,
        "delete": _handle_delete,
        "export": _handle_export,
        "import": _handle_import,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error("Command not implemented yet.")
        return 2
    return handler(args)


def _print_error(message: str) -> None:
    sys.stderr.write(message + "\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Study RAG workspace package."""

from .config import (  # noqa: F401
    CONFIG_PATH_ENV,
    ConfigError,
    RagConfig,
    config_template,
    default_tree,
    load_config,
    resolve_config_path,
    write_template,
)
from .ingest import (  # noqa: F401
    EmbeddingClient,
    IngestionReport,
    OpenAIEmbeddingClient,
    TextChunker,
    ingest_sources,
)
from .vector_store import (  # noqa: F401
    EmbeddingMetadata,
    SourceDocument,
    VectorStoreBackend,
    VectorStoreError,
    VectorStoreManifest,
    VectorStoreRepository,
    build_manifest,
)

__all__ = [
    "CONFIG_PATH_ENV",
    "ConfigError",
    "RagConfig",
    "config_template",
    "default_tree",
    "load_config",
    "resolve_config_path",
    "write_template",
    "EmbeddingClient",
    "OpenAIEmbeddingClient",
    "TextChunker",
    "IngestionReport",
    "ingest_sources",
    "EmbeddingMetadata",
    "SourceDocument",
    "VectorStoreBackend",
    "VectorStoreError",
    "VectorStoreManifest",
    "VectorStoreRepository",
    "build_manifest",
]

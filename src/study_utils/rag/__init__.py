"""Study RAG workspace package."""

from .chat import (  # noqa: F401
    ChatAnswer,
    ChatClient,
    ChatError,
    ChatRuntime,
    OpenAIChatClient,
    RetrievalResult,
)
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
from .session import (  # noqa: F401
    ChatMessage,
    ChatSession,
    SessionError,
    SessionStore,
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
    "ChatError",
    "ChatClient",
    "ChatRuntime",
    "OpenAIChatClient",
    "ChatAnswer",
    "RetrievalResult",
    "ChatMessage",
    "ChatSession",
    "SessionError",
    "SessionStore",
    "EmbeddingMetadata",
    "SourceDocument",
    "VectorStoreBackend",
    "VectorStoreError",
    "VectorStoreManifest",
    "VectorStoreRepository",
    "build_manifest",
]

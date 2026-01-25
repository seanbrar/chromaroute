"""chromaroute: Provider-agnostic embedding functions for ChromaDB."""

from importlib.metadata import PackageNotFoundError, version

from chromaroute.config import EmbedConfig, load_config
from chromaroute.embedding import OpenRouterEmbeddingFunction, build_embedding_function
from chromaroute.vector_store import VectorStore

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


__all__ = [
    "OpenRouterEmbeddingFunction",
    "build_embedding_function",
    "EmbedConfig",
    "load_config",
    "VectorStore",
    "__version__",
]

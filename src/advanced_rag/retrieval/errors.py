"""Dense retrieval exceptions with actionable messages."""


class DenseIndexError(Exception):
    """Base error for dense index operations."""


class EmbeddingConfigurationError(DenseIndexError):
    """The configured embedder is missing credentials or is index-incompatible."""


class EmbeddingResponseError(DenseIndexError):
    """An embedding provider returned malformed output."""


class SparseIndexError(Exception):
    """A persisted sparse index is missing, corrupt, or incompatible."""


class HybridRetrievalError(Exception):
    """Hybrid retrieval could not obtain candidates from either backend."""


class RerankerError(Exception):
    """A reranker returned malformed output or could not execute."""


class ChatModelError(Exception):
    """The configured answer-generation model could not produce valid output."""

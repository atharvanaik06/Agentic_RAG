"""Provider-neutral text embedding adapters."""

import math
import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Protocol

from openai import OpenAI
from pydantic import SecretStr

from advanced_rag.config import Settings
from advanced_rag.retrieval.errors import EmbeddingConfigurationError, EmbeddingResponseError

_TOKEN = re.compile(r"[\w-]+", re.UNICODE)


class EmbeddingProvider(Protocol):
    """Minimal interface used by dense indexing and search."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingProvider:
    """Batched adapter around the official OpenAI embeddings client."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
        client: OpenAI | None = None,
    ) -> None:
        self._model_name = model
        self._dimensions = dimensions
        self.batch_size = batch_size
        self._api_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self.client = client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        client = self._client()
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = client.embeddings.create(
                input=batch,
                model=self.model_name,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        _validate_vectors(vectors, expected_count=len(texts), dimensions=self.dimensions)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding query must not be empty")
        return self.embed_documents([text])[0]

    def _client(self) -> OpenAI:
        if self.client is None:
            if self._api_key is None:
                raise EmbeddingConfigurationError(
                    "OpenAI embeddings require RAG_OPENAI_API_KEY in the environment or .env file"
                )
            self.client = OpenAI(api_key=self._api_key)
        return self.client


class DeterministicEmbeddingProvider:
    """Local hash embedding for tests and pipeline development, not production."""

    provider_name = "deterministic"

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return "hash-embedding-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding query must not be empty")
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN.findall(text.casefold()) or [text]
        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def _validate_vectors(
    vectors: Sequence[Sequence[float]], *, expected_count: int, dimensions: int
) -> None:
    if len(vectors) != expected_count:
        raise EmbeddingResponseError(
            f"Embedding provider returned {len(vectors)} vectors for {expected_count} inputs"
        )
    if any(len(vector) != dimensions for vector in vectors):
        raise EmbeddingResponseError(f"Embedding vectors must have exactly {dimensions} dimensions")


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the configured embedding adapter."""
    if settings.embedding_provider == "deterministic":
        return DeterministicEmbeddingProvider(dimensions=settings.embedding_dimensions)
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )

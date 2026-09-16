from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from advanced_rag.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from advanced_rag.retrieval.errors import EmbeddingConfigurationError, EmbeddingResponseError


def test_deterministic_embeddings_are_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)

    first = provider.embed_query("Token authentication")
    second = provider.embed_query("Token authentication")

    assert first == second
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_openai_adapter_batches_and_restores_response_order() -> None:
    client = MagicMock()
    client.embeddings.create.side_effect = [
        SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        ),
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.5, 0.5])]),
    ]
    provider = OpenAIEmbeddingProvider(
        api_key=None,
        model="test-model",
        dimensions=2,
        batch_size=2,
        client=cast(OpenAI, client),
    )

    vectors = provider.embed_documents(["one", "two", "three"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert client.embeddings.create.call_count == 2


def test_openai_adapter_requires_key_only_when_embedding() -> None:
    provider = OpenAIEmbeddingProvider(api_key=None, dimensions=2)

    with pytest.raises(EmbeddingConfigurationError, match="RAG_OPENAI_API_KEY"):
        provider.embed_query("query")


def test_embedding_adapter_rejects_malformed_dimensions() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[1.0])]
    )
    provider = OpenAIEmbeddingProvider(
        api_key=None,
        dimensions=2,
        client=cast(OpenAI, client),
    )

    with pytest.raises(EmbeddingResponseError, match="exactly 2"):
        provider.embed_documents(["text"])

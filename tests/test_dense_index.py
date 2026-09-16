from collections.abc import Sequence
from pathlib import Path

import pytest

from advanced_rag.config import Settings
from advanced_rag.ingestion.pipeline import IngestionPipeline
from advanced_rag.retrieval.dense import ChromaDenseIndex
from advanced_rag.retrieval.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from advanced_rag.retrieval.errors import EmbeddingConfigurationError
from advanced_rag.retrieval.models import DenseSearchFilters


class CountingEmbedder(DeterministicEmbeddingProvider):
    def __init__(self, dimensions: int = 32) -> None:
        super().__init__(dimensions)
        self.document_calls = 0
        self.embedded_documents = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        self.embedded_documents += len(texts)
        return super().embed_documents(texts)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "raw",
        index_dir=tmp_path / "indexes",
        chroma_dir=tmp_path / "indexes" / "chroma",
        chroma_collection="test-collection",
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=5,
        embedding_provider="deterministic",
        embedding_model="hash-embedding-v1",
        embedding_dimensions=32,
        embedding_batch_size=2,
    )


def _index(
    tmp_path: Path, embedder: EmbeddingProvider | None = None
) -> tuple[ChromaDenseIndex, Settings]:
    settings = _settings(tmp_path)
    provider = embedder or CountingEmbedder(settings.embedding_dimensions)
    return (
        ChromaDenseIndex(
            path=settings.chroma_dir,
            collection_name=settings.chroma_collection,
            embedder=provider,
            write_batch_size=settings.embedding_batch_size,
        ),
        settings,
    )


def test_index_is_idempotent_searchable_filterable_and_persistent(tmp_path: Path) -> None:
    index, settings = _index(tmp_path)
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "auth.txt").write_text(
        "Authentication requires a secure access token.", encoding="utf-8"
    )
    (settings.data_dir / "garden.txt").write_text(
        "Tomatoes grow well with sunlight and water.", encoding="utf-8"
    )
    pipeline = IngestionPipeline(settings)

    first, ingestion = index.index_path(settings.data_dir, pipeline=pipeline)
    second, _ = index.index_path(settings.data_dir, pipeline=pipeline)

    assert ingestion.failed_files == 0
    assert first.embedded_chunks == 2
    assert first.total_chunks == 2
    assert second.embedded_chunks == 0
    assert second.unchanged_chunks == 2

    results = index.search("secure authentication token", top_k=2)
    assert results[0].chunk.metadata.filename == "auth.txt"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0 - results[0].distance)

    filtered = index.search(
        "sunlight",
        filters=DenseSearchFilters(filename="garden.txt", file_type="text"),
    )
    assert [result.chunk.metadata.filename for result in filtered] == ["garden.txt"]

    reopened, _ = _index(tmp_path)
    assert reopened.info().count == 2
    assert reopened.search("authentication", top_k=1)[0].chunk.metadata.filename == "auth.txt"


def test_changed_and_deleted_sources_remove_stale_chunks(tmp_path: Path) -> None:
    index, settings = _index(tmp_path)
    settings.data_dir.mkdir(parents=True)
    changing = settings.data_dir / "changing.txt"
    removed = settings.data_dir / "removed.txt"
    changing.write_text("Original authentication behavior.", encoding="utf-8")
    removed.write_text("A temporary document.", encoding="utf-8")
    pipeline = IngestionPipeline(settings)
    first, _ = index.index_path(settings.data_dir, pipeline=pipeline)

    changing.write_text("Updated authorization behavior.", encoding="utf-8")
    removed.unlink()
    second, _ = index.index_path(settings.data_dir, pipeline=pipeline)

    assert first.total_chunks == 2
    assert second.embedded_chunks == 1
    assert second.deleted_chunks == 2
    assert second.total_chunks == 1
    assert index.search("updated authorization", top_k=3)[0].chunk.text.startswith("Updated")


def test_no_prune_keeps_sources_missing_from_scan(tmp_path: Path) -> None:
    index, settings = _index(tmp_path)
    settings.data_dir.mkdir(parents=True)
    source = settings.data_dir / "kept.txt"
    source.write_text("Keep this indexed record.", encoding="utf-8")
    pipeline = IngestionPipeline(settings)
    index.index_path(settings.data_dir, pipeline=pipeline)
    source.unlink()

    result, _ = index.index_path(settings.data_dir, pipeline=pipeline, prune_missing=False)

    assert result.deleted_chunks == 0
    assert result.total_chunks == 1


def test_collection_rejects_incompatible_embedding_configuration(tmp_path: Path) -> None:
    index, _ = _index(tmp_path)
    assert index.info().embedding_dimensions == 32

    with pytest.raises(EmbeddingConfigurationError, match="incompatible"):
        ChromaDenseIndex(
            path=index.path,
            collection_name=index.collection_name,
            embedder=DeterministicEmbeddingProvider(dimensions=16),
        )


def test_empty_index_search_and_reset(tmp_path: Path) -> None:
    index, _ = _index(tmp_path)

    assert index.search("anything") == []
    with pytest.raises(ValueError, match="must not be empty"):
        index.search(" ")
    with pytest.raises(ValueError, match="positive"):
        index.search("query", top_k=0)

    index.reset()
    assert index.info().count == 0

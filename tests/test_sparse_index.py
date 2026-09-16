from pathlib import Path

import pytest

from advanced_rag.config import Settings
from advanced_rag.ingestion import IngestionPipeline
from advanced_rag.retrieval.errors import SparseIndexError
from advanced_rag.retrieval.models import RetrievalFilters
from advanced_rag.retrieval.sparse import BM25SparseIndex


def _settings(raw: Path, index: Path) -> Settings:
    return Settings(
        data_dir=raw,
        index_dir=index.parent,
        chroma_dir=index.parent / "chroma",
        bm25_dir=index,
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_size=1,
        embedding_provider="deterministic",
        embedding_dimensions=32,
    )


def test_sparse_index_persists_exact_terms_and_filters(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bis-1047.txt").write_text(
        "BIS Working Paper 1047 studies monetary policy and bank liquidity.",
        encoding="utf-8",
    )
    (raw / "inflation.md").write_text(
        "# Outlook\n\nInflation expectations and interest rates guide policy decisions.",
        encoding="utf-8",
    )
    settings = _settings(raw, tmp_path / "indexes" / "bm25")
    pipeline = IngestionPipeline(settings)
    index = BM25SparseIndex(path=settings.bm25_dir)

    first, ingestion = index.index_path(raw, pipeline=pipeline)

    assert first.rebuilt is True
    assert first.total_chunks == len(ingestion.chunks) == 2
    result = index.search("BIS Working Paper 1047", top_k=2)
    assert result[0].chunk.metadata.filename == "bis-1047.txt"
    assert result[0].score > 0

    reopened = BM25SparseIndex(path=settings.bm25_dir)
    repeated = reopened.search("BIS Working Paper 1047", top_k=2)
    assert [item.chunk.chunk_id for item in repeated] == [item.chunk.chunk_id for item in result]
    filtered = reopened.search(
        "policy",
        top_k=2,
        filters=RetrievalFilters(filename="inflation.md", file_type="markdown"),
    )
    assert len(filtered) == 1
    assert filtered[0].chunk.metadata.filename == "inflation.md"

    unchanged, _ = reopened.index_path(raw, pipeline=pipeline)
    assert unchanged.rebuilt is False
    assert unchanged.indexed_chunks == 0


def test_sparse_index_rebuilds_when_corpus_changes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "rates.txt"
    source.write_text("The policy rate is 4.25 percent.", encoding="utf-8")
    (raw / "alternative.txt").write_text(
        "An alternative scenario has a 4.75 percent policy rate.", encoding="utf-8"
    )
    settings = _settings(raw, tmp_path / "bm25")
    index = BM25SparseIndex(path=settings.bm25_dir)
    pipeline = IngestionPipeline(settings)
    initial, _ = index.index_path(raw, pipeline=pipeline)

    source.write_text("The policy rate is 3.75 percent after easing.", encoding="utf-8")
    changed, _ = index.index_path(raw, pipeline=pipeline)

    assert changed.rebuilt is True
    assert changed.fingerprint != initial.fingerprint
    assert index.search("3.75 easing")[0].chunk.metadata.filename == "rates.txt"
    assert index.search("4.75")[0].chunk.metadata.filename == "alternative.txt"
    assert index.search("term-does-not-exist") == []


def test_sparse_index_missing_corrupt_empty_and_reset(tmp_path: Path) -> None:
    path = tmp_path / "bm25"
    index = BM25SparseIndex(path=path)
    assert index.info().count == 0
    with pytest.raises(SparseIndexError, match="index-sparse"):
        index.search("policy")

    empty_raw = tmp_path / "raw"
    empty_raw.mkdir()
    settings = _settings(empty_raw, path)
    result, _ = index.index_path(empty_raw, pipeline=IngestionPipeline(settings))
    assert result.total_chunks == 0
    assert index.search("policy") == []

    (path / "manifest.json").write_text("not json", encoding="utf-8")
    with pytest.raises(SparseIndexError, match="manifest"):
        BM25SparseIndex(path=path).search("policy")

    index.reset()
    assert not path.exists()


@pytest.mark.parametrize(
    ("k1", "b", "message"),
    [(0.0, 0.75, "k1"), (1.5, 1.1, "b")],
)
def test_sparse_index_validates_parameters(
    tmp_path: Path, k1: float, b: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BM25SparseIndex(path=tmp_path / "bm25", k1=k1, b=b)

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from advanced_rag.ingestion.models import ChunkMetadata, DocumentChunk, FileType
from advanced_rag.retrieval.errors import HybridRetrievalError, RerankerError
from advanced_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from advanced_rag.retrieval.hybrid_models import FusedCandidate, RerankScore
from advanced_rag.retrieval.models import (
    DenseSearchFilters,
    DenseSearchResult,
    RetrievalFilters,
    SparseSearchResult,
)
from advanced_rag.retrieval.rerankers import FlashRankReranker, NoOpReranker
from advanced_rag.tools import create_search_knowledge_base_tool


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source: str = "source-a",
    page: int = 1,
    tokens: int = 20,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        token_count=tokens,
        metadata=ChunkMetadata(
            source_id=source,
            source_path=f"{source}.pdf",
            filename=f"{source}.pdf",
            file_type=FileType.PDF,
            content_hash=f"hash-{source}",
            title=source,
            page_number=page,
            chunk_index=page,
        ),
    )


class FakeDense:
    def __init__(
        self, results: list[DenseSearchResult] | None = None, error: Exception | None = None
    ):
        self.results = results or []
        self.error = error

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: DenseSearchFilters | None = None,
    ) -> list[DenseSearchResult]:
        del query, filters
        if self.error:
            raise self.error
        return self.results[:top_k]


class FakeSparse:
    def __init__(
        self, results: list[SparseSearchResult] | None = None, error: Exception | None = None
    ):
        self.results = results or []
        self.error = error

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: RetrievalFilters | None = None,
    ) -> list[SparseSearchResult]:
        del query, filters
        if self.error:
            raise self.error
        return self.results[:top_k]


class ReverseReranker:
    @property
    def name(self) -> str:
        return "reverse-test"

    def rerank(self, query: str, candidates: Sequence[FusedCandidate]) -> list[RerankScore]:
        del query
        return [
            RerankScore(chunk_id=candidate.chunk.chunk_id, score=float(position))
            for position, candidate in enumerate(candidates, start=1)
        ]


class FailingReranker:
    @property
    def name(self) -> str:
        return "failing-test"

    def rerank(self, query: str, candidates: Sequence[FusedCandidate]) -> list[RerankScore]:
        del query, candidates
        raise RerankerError("planned failure")


def _dense(chunk: DocumentChunk, rank: int) -> DenseSearchResult:
    return DenseSearchResult(chunk=chunk, rank=rank, distance=0.1 * rank, score=1 - 0.1 * rank)


def _sparse(chunk: DocumentChunk, rank: int) -> SparseSearchResult:
    return SparseSearchResult(chunk=chunk, rank=rank, score=10.0 / rank)


def test_rrf_deduplicates_and_rewards_cross_retriever_agreement() -> None:
    agreed = _chunk("agreed", "shared evidence")
    dense_only = _chunk("dense", "semantic evidence")
    sparse_only = _chunk("sparse", "keyword evidence")

    fused = reciprocal_rank_fusion(
        [_dense(dense_only, 1), _dense(agreed, 2)],
        [_sparse(agreed, 1), _sparse(sparse_only, 2)],
        rrf_k=60,
    )

    assert [item.chunk.chunk_id for item in fused] == ["agreed", "dense", "sparse"]
    assert fused[0].retrieval_sources == ("dense", "sparse")
    assert fused[0].dense_score == pytest.approx(0.8)
    assert fused[0].sparse_score == pytest.approx(10.0)
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([], [], rrf_k=0)


def test_hybrid_reranks_diversifies_and_enforces_budget() -> None:
    first = _chunk("first", "inflation demand evidence", tokens=30)
    second = _chunk("second", "supply shock evidence", page=2, tokens=30)
    third = _chunk("third", "foreign policy evidence", source="source-b", tokens=30)
    retriever = HybridRetriever(
        dense=FakeDense([_dense(first, 1), _dense(second, 2), _dense(third, 3)]),
        sparse=FakeSparse([_sparse(first, 1), _sparse(second, 2), _sparse(third, 3)]),
        reranker=ReverseReranker(),
        context_token_budget=60,
        max_chunks_per_source=1,
    )

    response = retriever.search("inflation", top_k=3)

    assert [item.chunk.chunk_id for item in response.results] == ["third", "second"]
    assert response.results[0].reranker_score == pytest.approx(3.0)
    assert response.diagnostics.evidence_tokens == 60
    assert response.diagnostics.distinct_sources == 2
    assert response.diagnostics.rerank_applied is True
    assert response.diagnostics.confidence == "high"


def test_hybrid_falls_back_when_one_backend_or_reranker_fails() -> None:
    chunk = _chunk("only", "monetary policy transmission")
    retriever = HybridRetriever(
        dense=FakeDense(error=RuntimeError("dense offline")),
        sparse=FakeSparse([_sparse(chunk, 1)]),
        reranker=FailingReranker(),
    )

    response = retriever.search("transmission")

    assert response.results[0].chunk.chunk_id == "only"
    assert response.results[0].reranker_score is None
    assert response.diagnostics.rerank_applied is False
    assert len(response.diagnostics.warnings) == 2
    assert response.diagnostics.confidence == "low"

    broken = HybridRetriever(
        dense=FakeDense(error=RuntimeError("dense offline")),
        sparse=FakeSparse(error=RuntimeError("sparse offline")),
        reranker=NoOpReranker(),
    )
    with pytest.raises(HybridRetrievalError, match="Both retrieval backends failed"):
        broken.search("transmission")


def test_hybrid_validates_inputs_and_suppresses_near_duplicates() -> None:
    original = _chunk("original", "one two three four five six seven eight nine ten")
    duplicate = _chunk("duplicate", "one two three four five six seven eight nine ten extra")
    retriever = HybridRetriever(
        dense=FakeDense([_dense(original, 1), _dense(duplicate, 2)]),
        sparse=FakeSparse(),
        reranker=NoOpReranker(),
    )

    assert len(retriever.search("numbers", top_k=2).results) == 1
    with pytest.raises(ValueError, match="must not be empty"):
        retriever.search(" ")
    with pytest.raises(ValueError, match="positive"):
        retriever.search("query", top_k=0)
    with pytest.raises(ValueError, match="positive"):
        HybridRetriever(
            dense=FakeDense(), sparse=FakeSparse(), reranker=NoOpReranker(), candidate_k=0
        )


def test_knowledge_base_tool_returns_structured_citations() -> None:
    chunk = _chunk("tool-result", "central bank evidence")
    retriever = HybridRetriever(
        dense=FakeDense([_dense(chunk, 1)]),
        sparse=FakeSparse([_sparse(chunk, 1)]),
        reranker=NoOpReranker(),
    )
    tool = create_search_knowledge_base_tool(retriever)

    output = tool.invoke({"query": "central bank", "top_k": 1, "file_type": "pdf"})

    assert tool.name == "search_knowledge_base"
    assert output["results"][0]["chunk"]["metadata"]["page_number"] == 1
    assert output["diagnostics"]["agreement_count"] == 1


def test_flashrank_adapter_parses_and_validates_results(tmp_path: Path) -> None:
    chunk = _chunk("flash", "inflation evidence")
    candidate = reciprocal_rank_fusion([_dense(chunk, 1)], [], rrf_k=60)[0]

    class FakeRanker:
        def rerank(self, request: Any) -> list[dict[str, object]]:
            del request
            return [{"id": "flash", "text": "inflation evidence", "score": 0.9}]

    class BadRanker:
        def rerank(self, request: Any) -> list[dict[str, object]]:
            del request
            return []

    reranker = FlashRankReranker(model_name="test-model", cache_dir=tmp_path)
    reranker._ranker = FakeRanker()
    assert reranker.rerank("inflation", [candidate])[0].score == pytest.approx(0.9)

    reranker._ranker = BadRanker()
    with pytest.raises(RerankerError, match="missing"):
        reranker.rerank("inflation", [candidate])

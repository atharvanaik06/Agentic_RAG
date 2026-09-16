"""Hybrid dense/BM25 retrieval with RRF, reranking, and diversity selection."""

import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import NotRequired, Protocol, TypedDict

from advanced_rag.ingestion.models import DocumentChunk
from advanced_rag.retrieval.errors import HybridRetrievalError, RerankerError
from advanced_rag.retrieval.hybrid_models import (
    ConfidenceLevel,
    FusedCandidate,
    HybridSearchResponse,
    HybridSearchResult,
    RetrievalDiagnostics,
)
from advanced_rag.retrieval.models import (
    DenseSearchFilters,
    DenseSearchResult,
    RetrievalFilters,
    SparseSearchResult,
)
from advanced_rag.retrieval.rerankers import Reranker

_WORD_PATTERN = re.compile(r"(?u)\b\w+\b")


class _FusionEntry(TypedDict):
    chunk: DocumentChunk
    rrf: float
    dense_rank: NotRequired[int]
    dense_score: NotRequired[float]
    sparse_rank: NotRequired[int]
    sparse_score: NotRequired[float]


class DenseRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: DenseSearchFilters | None = None,
    ) -> list[DenseSearchResult]: ...


class SparseRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: RetrievalFilters | None = None,
    ) -> list[SparseSearchResult]: ...


class HybridRetriever:
    """Coordinate candidate retrieval and return budgeted, citation-ready evidence."""

    def __init__(
        self,
        *,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        reranker: Reranker,
        candidate_k: int = 15,
        rerank_k: int = 15,
        rrf_k: int = 60,
        context_token_budget: int = 3000,
        max_chunks_per_source: int = 3,
        max_chunks_per_page: int = 2,
    ) -> None:
        values = {
            "candidate_k": candidate_k,
            "rerank_k": rerank_k,
            "rrf_k": rrf_k,
            "context_token_budget": context_token_budget,
            "max_chunks_per_source": max_chunks_per_source,
            "max_chunks_per_page": max_chunks_per_page,
        }
        if any(value < 1 for value in values.values()):
            raise ValueError("Hybrid retrieval limits must all be positive")
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k
        self.context_token_budget = context_token_budget
        self.max_chunks_per_source = max_chunks_per_source
        self.max_chunks_per_page = max_chunks_per_page

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: RetrievalFilters | None = None,
    ) -> HybridSearchResponse:
        """Retrieve in parallel, fuse, rerank, diversify, and enforce a token budget."""
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        dense_results, sparse_results, warnings = self._retrieve(query, filters)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=self.rrf_k)
        rerank_pool = fused[: self.rerank_k]
        rerank_applied = self.reranker.name != "none" and bool(rerank_pool)
        try:
            rerank_scores = self.reranker.rerank(query, rerank_pool)
            expected_ids = {item.chunk.chunk_id for item in rerank_pool}
            returned_ids = {item.chunk_id for item in rerank_scores}
            if len(rerank_scores) != len(rerank_pool) or returned_ids != expected_ids:
                raise RerankerError("Reranker returned missing, duplicate, or unknown chunk IDs")
            score_by_id = {item.chunk_id: item.score for item in rerank_scores}
            ordered = sorted(
                rerank_pool,
                key=lambda item: (
                    -score_by_id[item.chunk.chunk_id],
                    -item.rrf_score,
                    item.chunk.chunk_id,
                ),
            )
        except Exception as exc:
            rerank_applied = False
            score_by_id = {}
            ordered = rerank_pool
            warnings.append(f"Reranker unavailable; used fused ranking: {exc}")

        selected = self._select(ordered, score_by_id, top_k)
        agreement = sum(
            candidate.dense_rank is not None and candidate.sparse_rank is not None
            for candidate in fused
        )
        distinct_sources = len({item.chunk.metadata.source_id for item in selected})
        evidence_tokens = sum(item.chunk.token_count for item in selected)
        confidence = _confidence(selected, agreement, distinct_sources, warnings)
        diagnostics = RetrievalDiagnostics(
            dense_candidates=len(dense_results),
            sparse_candidates=len(sparse_results),
            fused_candidates=len(fused),
            reranked_candidates=len(rerank_pool),
            agreement_count=agreement,
            distinct_sources=distinct_sources,
            evidence_tokens=evidence_tokens,
            confidence=confidence,
            reranker=self.reranker.name,
            rerank_applied=rerank_applied,
            warnings=tuple(warnings),
        )
        return HybridSearchResponse(query=query, results=tuple(selected), diagnostics=diagnostics)

    def _retrieve(
        self, query: str, filters: RetrievalFilters | None
    ) -> tuple[list[DenseSearchResult], list[SparseSearchResult], list[str]]:
        dense_filters = DenseSearchFilters.model_validate(filters.model_dump()) if filters else None
        warnings: list[str] = []
        dense_failed = False
        sparse_failed = False
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as executor:
            dense_future = executor.submit(
                self.dense.search,
                query,
                top_k=self.candidate_k,
                filters=dense_filters,
            )
            sparse_future = executor.submit(
                self.sparse.search,
                query,
                top_k=self.candidate_k,
                filters=filters,
            )
            try:
                dense_results = dense_future.result()
            except Exception as exc:
                dense_results = []
                dense_failed = True
                warnings.append(f"Dense retrieval failed: {exc}")
            try:
                sparse_results = sparse_future.result()
            except Exception as exc:
                sparse_results = []
                sparse_failed = True
                warnings.append(f"Sparse retrieval failed: {exc}")
        if dense_failed and sparse_failed:
            raise HybridRetrievalError("Both retrieval backends failed: " + "; ".join(warnings))
        return dense_results, sparse_results, warnings

    def _select(
        self,
        candidates: Sequence[FusedCandidate],
        score_by_id: dict[str, float],
        top_k: int,
    ) -> list[HybridSearchResult]:
        selected: list[HybridSearchResult] = []
        source_counts: Counter[str] = Counter()
        page_counts: Counter[tuple[str, int]] = Counter()
        used_tokens = 0
        for candidate in candidates:
            chunk = candidate.chunk
            source_id = chunk.metadata.source_id
            page = chunk.metadata.page_number
            page_key = (source_id, page) if page is not None else None
            if source_counts[source_id] >= self.max_chunks_per_source:
                continue
            if page_key is not None and page_counts[page_key] >= self.max_chunks_per_page:
                continue
            if used_tokens + chunk.token_count > self.context_token_budget:
                continue
            if any(_near_duplicate(chunk.text, item.chunk.text) for item in selected):
                continue
            selected.append(
                HybridSearchResult(
                    **candidate.model_dump(),
                    final_rank=len(selected) + 1,
                    reranker_score=score_by_id.get(chunk.chunk_id),
                )
            )
            source_counts[source_id] += 1
            if page_key is not None:
                page_counts[page_key] += 1
            used_tokens += chunk.token_count
            if len(selected) == top_k:
                break
        return selected


def reciprocal_rank_fusion(
    dense: Sequence[DenseSearchResult],
    sparse: Sequence[SparseSearchResult],
    *,
    rrf_k: int = 60,
) -> list[FusedCandidate]:
    """Fuse heterogeneous ranks without comparing their backend-specific scores."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    by_id: dict[str, _FusionEntry] = {}
    for dense_result in dense:
        entry = by_id.setdefault(
            dense_result.chunk.chunk_id, {"chunk": dense_result.chunk, "rrf": 0.0}
        )
        entry["dense_rank"] = dense_result.rank
        entry["dense_score"] = dense_result.score
        entry["rrf"] += 1.0 / (rrf_k + dense_result.rank)
    for sparse_result in sparse:
        entry = by_id.setdefault(
            sparse_result.chunk.chunk_id, {"chunk": sparse_result.chunk, "rrf": 0.0}
        )
        entry["sparse_rank"] = sparse_result.rank
        entry["sparse_score"] = sparse_result.score
        entry["rrf"] += 1.0 / (rrf_k + sparse_result.rank)

    candidates = [
        FusedCandidate(
            chunk=entry["chunk"],
            rrf_score=entry["rrf"],
            dense_rank=entry.get("dense_rank"),
            dense_score=entry.get("dense_score"),
            sparse_rank=entry.get("sparse_rank"),
            sparse_score=entry.get("sparse_score"),
            retrieval_sources=tuple(
                source
                for source, key in (("dense", "dense_rank"), ("sparse", "sparse_rank"))
                if key in entry
            ),
        )
        for entry in by_id.values()
    ]
    return sorted(
        candidates,
        key=lambda item: (
            -item.rrf_score,
            min(item.dense_rank or 10**9, item.sparse_rank or 10**9),
            item.chunk.chunk_id,
        ),
    )


def _near_duplicate(left: str, right: str, threshold: float = 0.9) -> bool:
    left_tokens = set(_WORD_PATTERN.findall(left.lower()))
    right_tokens = set(_WORD_PATTERN.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= threshold


def _confidence(
    results: Sequence[HybridSearchResult],
    agreement: int,
    distinct_sources: int,
    warnings: Sequence[str],
) -> ConfidenceLevel:
    if not results or warnings:
        return "low"
    if agreement >= 2 and distinct_sources >= 2:
        return "high"
    return "medium"

"""Comparative dense, sparse, RRF, and reranked retrieval evaluation."""

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Protocol

from advanced_rag.evaluation.metrics import mean, retrieval_metrics
from advanced_rag.evaluation.models import (
    BenchmarkCase,
    MethodSummary,
    RerankerSummary,
    RetrievalCaseResult,
    RetrievalEvaluationReport,
    RetrievalMethod,
)
from advanced_rag.ingestion.models import DocumentChunk
from advanced_rag.retrieval.hybrid import DenseRetriever, SparseRetriever
from advanced_rag.retrieval.hybrid_models import HybridSearchResponse
from advanced_rag.retrieval.models import DenseSearchFilters, RetrievalFilters


class HybridSearchRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: RetrievalFilters | None = None,
    ) -> HybridSearchResponse: ...


class RetrievalEvaluator:
    """Run a benchmark independently through every retrieval stage."""

    def __init__(
        self,
        *,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        hybrid: HybridSearchRetriever,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.hybrid = hybrid
        self.rrf_k = rrf_k

    def evaluate(
        self,
        cases: Sequence[BenchmarkCase],
        *,
        benchmark_name: str,
        top_k: int = 5,
    ) -> RetrievalEvaluationReport:
        """Evaluate answerable cases and return auditable per-query rankings."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        results: list[RetrievalCaseResult] = []
        rank_changes: list[tuple[int | None, int | None]] = []
        answerable = [case for case in cases if case.answerable]
        for case in answerable:
            dense_hits, dense_latency, dense_error = self._dense(case, top_k)
            sparse_hits, sparse_latency, sparse_error = self._sparse(case, top_k)
            fused_hits = _fuse_records(dense_hits, sparse_hits, self.rrf_k)
            fused_latency = dense_latency + sparse_latency
            hybrid_hits, hybrid_latency, hybrid_error = self._hybrid(case, top_k)

            method_data: tuple[
                tuple[RetrievalMethod, list[tuple[DocumentChunk, float | None]], float, str | None],
                ...,
            ] = (
                ("dense", dense_hits, dense_latency, dense_error),
                ("sparse", sparse_hits, sparse_latency, sparse_error),
                (
                    "hybrid_rrf",
                    fused_hits,
                    fused_latency,
                    dense_error or sparse_error,
                ),
                ("hybrid_reranked", hybrid_hits, hybrid_latency, hybrid_error),
            )
            case_results: dict[RetrievalMethod, RetrievalCaseResult] = {}
            for method, hits, latency, error in method_data:
                metrics = retrieval_metrics(hits, case, top_k=top_k)
                case_result = RetrievalCaseResult(
                    case_id=case.id,
                    category=case.category,
                    method=method,
                    recall_at_k=metrics[0],
                    precision_at_k=metrics[1],
                    reciprocal_rank=metrics[2],
                    average_precision=metrics[3],
                    ndcg_at_k=metrics[4],
                    source_hit=metrics[5],
                    page_hit=metrics[6],
                    latency_ms=latency,
                    records=metrics[7],
                    error=error,
                )
                results.append(case_result)
                case_results[method] = case_result
            rank_changes.append(
                (
                    _best_relevant_rank(case_results["hybrid_rrf"]),
                    _best_relevant_rank(case_results["hybrid_reranked"]),
                )
            )

        methods: tuple[RetrievalMethod, ...] = (
            "dense",
            "sparse",
            "hybrid_rrf",
            "hybrid_reranked",
        )
        summaries = tuple(_summarize(method, results) for method in methods)
        return RetrievalEvaluationReport(
            benchmark=benchmark_name,
            top_k=top_k,
            cases=len(answerable),
            results=tuple(results),
            summaries=summaries,
            reranker=_reranker_summary(rank_changes),
        )

    def _dense(
        self, case: BenchmarkCase, top_k: int
    ) -> tuple[list[tuple[DocumentChunk, float | None]], float, str | None]:
        started = perf_counter()
        try:
            results = self.dense.search(
                case.question,
                top_k=top_k,
                filters=DenseSearchFilters(),
            )
            return (
                [(item.chunk, item.score) for item in results],
                (perf_counter() - started) * 1000,
                None,
            )
        except Exception as exc:
            return [], (perf_counter() - started) * 1000, str(exc)

    def _sparse(
        self, case: BenchmarkCase, top_k: int
    ) -> tuple[list[tuple[DocumentChunk, float | None]], float, str | None]:
        started = perf_counter()
        try:
            results = self.sparse.search(
                case.question,
                top_k=top_k,
                filters=RetrievalFilters(),
            )
            return (
                [(item.chunk, item.score) for item in results],
                (perf_counter() - started) * 1000,
                None,
            )
        except Exception as exc:
            return [], (perf_counter() - started) * 1000, str(exc)

    def _hybrid(
        self, case: BenchmarkCase, top_k: int
    ) -> tuple[list[tuple[DocumentChunk, float | None]], float, str | None]:
        started = perf_counter()
        try:
            response = self.hybrid.search(case.question, top_k=top_k)
            return (
                [(item.chunk, item.reranker_score) for item in response.results],
                (perf_counter() - started) * 1000,
                None,
            )
        except Exception as exc:
            return [], (perf_counter() - started) * 1000, str(exc)


def _fuse_records(
    dense: list[tuple[DocumentChunk, float | None]],
    sparse: list[tuple[DocumentChunk, float | None]],
    rrf_k: int,
) -> list[tuple[DocumentChunk, float | None]]:
    by_id: dict[str, tuple[DocumentChunk, float]] = {}
    for results in (dense, sparse):
        for rank, (chunk, _score) in enumerate(results, start=1):
            existing = by_id.get(chunk.chunk_id)
            score = (existing[1] if existing else 0.0) + 1.0 / (rrf_k + rank)
            by_id[chunk.chunk_id] = (chunk, score)
    fused: list[tuple[DocumentChunk, float | None]] = list(by_id.values())
    fused.sort(
        key=lambda item: (
            -(item[1] if item[1] is not None else 0.0),
            item[0].chunk_id,
        )
    )
    return fused


def _summarize(method: RetrievalMethod, results: Sequence[RetrievalCaseResult]) -> MethodSummary:
    selected = [result for result in results if result.method == method]
    page_results = [result.page_hit for result in selected if result.page_hit is not None]
    return MethodSummary(
        method=method,
        cases=len(selected),
        recall_at_k=mean([item.recall_at_k for item in selected]),
        precision_at_k=mean([item.precision_at_k for item in selected]),
        mrr=mean([item.reciprocal_rank for item in selected]),
        map=mean([item.average_precision for item in selected]),
        ndcg_at_k=mean([item.ndcg_at_k for item in selected]),
        source_hit_rate=mean([float(item.source_hit) for item in selected]),
        page_hit_rate=mean([float(value) for value in page_results]) if page_results else None,
        average_latency_ms=mean([item.latency_ms for item in selected]),
    )


def _best_relevant_rank(result: RetrievalCaseResult) -> int | None:
    return next((record.rank for record in result.records if record.relevant), None)


def _reranker_summary(changes: Sequence[tuple[int | None, int | None]]) -> RerankerSummary:
    improved = unchanged = worsened = missing = 0
    numeric_changes: list[float] = []
    for before, after in changes:
        if before is None or after is None:
            missing += 1
        elif after < before:
            improved += 1
            numeric_changes.append(float(before - after))
        elif after > before:
            worsened += 1
            numeric_changes.append(float(before - after))
        else:
            unchanged += 1
            numeric_changes.append(0.0)
    return RerankerSummary(
        improved=improved,
        unchanged=unchanged,
        worsened=worsened,
        missing=missing,
        average_rank_change=mean(numeric_changes),
    )


def benchmark_name(path: Path | str) -> str:
    """Use a stable filename in persisted reports without exposing absolute paths."""
    return Path(path).name

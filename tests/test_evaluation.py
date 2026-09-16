from collections.abc import Sequence
from pathlib import Path

import pytest

from advanced_rag.evaluation.agent import AgentEvaluator
from advanced_rag.evaluation.dataset import BenchmarkError, load_benchmark
from advanced_rag.evaluation.metrics import retrieval_metrics
from advanced_rag.evaluation.models import (
    BenchmarkCase,
    EntailmentResult,
    GoldTarget,
    RegressionThresholds,
)
from advanced_rag.evaluation.reporting import (
    load_agent_report,
    load_retrieval_report,
    regression_result,
    render_markdown,
    save_report,
)
from advanced_rag.evaluation.retrieval import RetrievalEvaluator
from advanced_rag.generation.models import (
    AgentAnswer,
    CitationSource,
    CitationValidation,
    GraphTraceEvent,
    ModelUsage,
)
from advanced_rag.ingestion.models import ChunkMetadata, DocumentChunk, FileType
from advanced_rag.retrieval.hybrid_models import (
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


def _chunk(chunk_id: str, filename: str, page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=f"Evidence from {filename} about inflation demand and supply.",
        token_count=10,
        metadata=ChunkMetadata(
            source_id=filename,
            source_path=filename,
            filename=filename,
            file_type=FileType.PDF,
            content_hash=f"hash-{filename}",
            page_number=page,
            chunk_index=page,
        ),
    )


def _case(case_id: str = "case-1", *, answerable: bool = True) -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        question="What drove inflation?",
        category="single_document" if answerable else "unanswerable",
        answerable=answerable,
        gold_targets=(GoldTarget(filename="gold.pdf", pages=(2,)),) if answerable else (),
        must_include=("demand", "supply") if answerable else (),
    )


class FakeDense:
    def __init__(self, results: list[DenseSearchResult]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: DenseSearchFilters | None = None,
    ) -> list[DenseSearchResult]:
        del query, filters
        return self.results[:top_k]


class FakeSparse:
    def __init__(self, results: list[SparseSearchResult]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: RetrievalFilters | None = None,
    ) -> list[SparseSearchResult]:
        del query, filters
        return self.results[:top_k]


class FakeHybrid:
    def __init__(self, result: HybridSearchResult) -> None:
        self.result = result

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: RetrievalFilters | None = None,
    ) -> HybridSearchResponse:
        del top_k, filters
        return HybridSearchResponse(
            query=query,
            results=(self.result,),
            diagnostics=_diagnostics(),
        )


def _diagnostics() -> RetrievalDiagnostics:
    return RetrievalDiagnostics(
        dense_candidates=2,
        sparse_candidates=1,
        fused_candidates=2,
        reranked_candidates=2,
        agreement_count=1,
        distinct_sources=1,
        evidence_tokens=10,
        confidence="high",
        reranker="test",
        rerank_applied=True,
    )


def test_benchmark_loader_validates_jsonl_and_seed_dataset(tmp_path: Path) -> None:
    seed = load_benchmark("evaluations/monetary_policy.jsonl")
    assert len(seed) == 30
    assert sum(case.answerable for case in seed) == 27

    duplicate = tmp_path / "duplicate.jsonl"
    record = _case().model_dump_json()
    duplicate.write_text(record + "\n" + record + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="Duplicate"):
        load_benchmark(duplicate)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="empty"):
        load_benchmark(empty)


def test_metrics_use_distinct_gold_targets_and_rank_positions() -> None:
    case = _case()
    irrelevant = _chunk("x", "other.pdf")
    relevant = _chunk("g", "gold.pdf", page=2)

    metrics = retrieval_metrics([(irrelevant, 0.9), (relevant, 0.8)], case, top_k=2)

    assert metrics[0] == 1.0
    assert metrics[1] == 0.5
    assert metrics[2] == 0.5
    assert metrics[3] == 0.5
    assert metrics[5] is True
    assert metrics[6] is True
    assert metrics[7][1].relevant is True


def test_retrieval_evaluator_compares_all_four_methods(tmp_path: Path) -> None:
    gold = _chunk("g", "gold.pdf", page=2)
    other = _chunk("x", "other.pdf")
    dense = FakeDense(
        [
            DenseSearchResult(chunk=other, rank=1, distance=0.1, score=0.9),
            DenseSearchResult(chunk=gold, rank=2, distance=0.2, score=0.8),
        ]
    )
    sparse = FakeSparse([SparseSearchResult(chunk=gold, rank=1, score=4.0)])
    hybrid_result = HybridSearchResult(
        chunk=gold,
        rrf_score=0.03,
        dense_rank=2,
        sparse_rank=1,
        retrieval_sources=("dense", "sparse"),
        final_rank=1,
        reranker_score=0.99,
    )
    report = RetrievalEvaluator(
        dense=dense,
        sparse=sparse,
        hybrid=FakeHybrid(hybrid_result),
    ).evaluate([_case(), _case("unanswerable", answerable=False)], benchmark_name="test", top_k=2)

    assert report.cases == 1
    assert {summary.method for summary in report.summaries} == {
        "dense",
        "sparse",
        "hybrid_rrf",
        "hybrid_reranked",
    }
    hybrid = next(item for item in report.summaries if item.method == "hybrid_reranked")
    assert hybrid.recall_at_k == 1.0
    path = save_report(report, tmp_path / "retrieval.json")
    assert load_retrieval_report(path) == report


class FakeAgent:
    def ask(self, question: str, *, top_k: int | None = None) -> AgentAnswer:
        del top_k
        unanswerable = "absent" in question
        return AgentAnswer(
            question=question,
            answer=(
                "The indexed documents do not provide enough verified evidence."
                if unanswerable
                else "- Demand and supply contributed. [S1]"
            ),
            sources=(
                ()
                if unanswerable
                else (
                    CitationSource(
                        label="S1",
                        chunk_id="g",
                        filename="gold.pdf",
                        title="Gold",
                        page_number=2,
                    ),
                )
            ),
            insufficient_evidence=unanswerable,
            citation_validation=CitationValidation(
                valid=True,
                accepted_claims=0 if unanswerable else 1,
                rejected_claims=0,
            ),
            retrieval_diagnostics=_diagnostics(),
            graph_trace=(
                GraphTraceEvent(
                    node="rewrite_query",
                    action="rewrote_query",
                    detail="test",
                    retrieval_attempt=1,
                ),
            ),
            usage=ModelUsage(api_calls=1, input_tokens=10, output_tokens=5),
            retrieval_attempts=2,
            final_query=question,
        )


class FakeStore:
    def get_chunks(self, chunk_ids: set[str]) -> dict[str, DocumentChunk]:
        return {"g": _chunk("g", "gold.pdf", 2)} if "g" in chunk_ids else {}


class FakeJudge:
    def judge(
        self,
        *,
        claim: str,
        citations: Sequence[str],
        evidence: Sequence[DocumentChunk],
    ) -> EntailmentResult:
        assert evidence
        return EntailmentResult(
            claim=claim,
            citations=tuple(citations),
            label="supported",
            explanation="Direct support.",
        )


def test_agent_evaluator_scores_answers_refusals_and_optional_entailment(
    tmp_path: Path,
) -> None:
    answerable = _case()
    unanswerable = BenchmarkCase(
        id="absent",
        question="Which fact is absent?",
        category="unanswerable",
        answerable=False,
    )
    report = AgentEvaluator(
        agent=FakeAgent(), judge=FakeJudge(), evidence_store=FakeStore()
    ).evaluate([answerable, unanswerable], benchmark_name="test")

    assert report.summary.citation_validity_rate == 1.0
    assert report.summary.refusal_accuracy == 1.0
    assert report.summary.concept_coverage == 1.0
    assert report.summary.entailment_support_rate == 1.0
    assert report.summary.rewrite_rate == 1.0
    path = save_report(report, tmp_path / "agent.json")
    assert load_agent_report(path) == report

    regression = regression_result(
        retrieval=None,
        agent=report,
        thresholds=RegressionThresholds(),
    )
    assert regression.passed is False  # Average attempts are 2.0, above the 1.5 gate.
    markdown = render_markdown(retrieval=None, agent=report, regression=regression)
    assert "# Evaluation Summary" in markdown
    assert "Overall: **FAIL**" in markdown


def test_regression_skips_refusal_gate_when_run_has_no_unanswerable_cases() -> None:
    report = AgentEvaluator(agent=FakeAgent()).evaluate([_case()], benchmark_name="smoke")

    regression = regression_result(
        retrieval=None,
        agent=report,
        thresholds=RegressionThresholds(),
    )

    assert regression.passed is False  # The fake agent still exceeds the attempts gate.
    assert "refusal_accuracy" not in regression.checks
    assert report.summary.refusal_accuracy is None

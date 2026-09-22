"""Evaluation-dashboard services independent of Streamlit rendering."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from advanced_rag.config import Settings
from advanced_rag.evaluation import AgentEvaluator, RetrievalEvaluator, load_benchmark
from advanced_rag.evaluation.models import (
    AgentEvaluationReport,
    BenchmarkCase,
    RegressionResult,
    RegressionThresholds,
    RetrievalEvaluationReport,
)
from advanced_rag.evaluation.reporting import (
    load_agent_report,
    load_retrieval_report,
    regression_result,
    save_report,
)
from advanced_rag.evaluation.retrieval import benchmark_name
from advanced_rag.graph import AgenticRAG
from advanced_rag.retrieval.dense import ChromaDenseIndex
from advanced_rag.retrieval.sparse import BM25SparseIndex
from advanced_rag.runtime import create_hybrid_retriever


@dataclass(frozen=True)
class BenchmarkOverview:
    """Compact composition summary for one portable benchmark."""

    path: Path
    cases: tuple[BenchmarkCase, ...]
    answerable: int
    unanswerable: int
    categories: dict[str, int]


@dataclass(frozen=True)
class SavedEvaluationReports:
    """Validated saved reports plus non-fatal loading errors."""

    retrieval: RetrievalEvaluationReport | None
    agent: AgentEvaluationReport | None
    errors: tuple[str, ...] = ()


def discover_benchmarks(directory: Path) -> tuple[Path, ...]:
    """Discover benchmark JSONL files in a stable order."""
    if not directory.is_dir():
        return ()
    return tuple(sorted(path for path in directory.glob("*.jsonl") if path.is_file()))


def inspect_benchmark(path: Path) -> BenchmarkOverview:
    """Load and summarize a benchmark without running retrieval or generation."""
    cases = load_benchmark(path)
    answerable = sum(case.answerable for case in cases)
    return BenchmarkOverview(
        path=path,
        cases=cases,
        answerable=answerable,
        unanswerable=len(cases) - answerable,
        categories=dict(sorted(Counter(case.category for case in cases).items())),
    )


def thresholds_from_settings(settings: Settings) -> RegressionThresholds:
    """Create the same quality gates used by the CLI."""
    return RegressionThresholds(
        hybrid_recall_at_k=settings.evaluation_hybrid_recall_threshold,
        citation_validity_rate=settings.evaluation_citation_validity_threshold,
        refusal_accuracy=settings.evaluation_refusal_accuracy_threshold,
        maximum_average_retrieval_attempts=settings.evaluation_max_average_attempts,
    )


def load_saved_reports(settings: Settings) -> SavedEvaluationReports:
    """Load standard report paths while isolating corrupt or incompatible files."""
    retrieval: RetrievalEvaluationReport | None = None
    agent: AgentEvaluationReport | None = None
    errors: list[str] = []
    retrieval_path = settings.evaluation_dir / "retrieval-results.json"
    agent_path = settings.evaluation_dir / "agent-results.json"
    if retrieval_path.is_file():
        try:
            retrieval = load_retrieval_report(retrieval_path)
        except Exception as exc:
            errors.append(f"Could not load {retrieval_path}: {exc}")
    if agent_path.is_file():
        try:
            agent = load_agent_report(agent_path)
        except Exception as exc:
            errors.append(f"Could not load {agent_path}: {exc}")
    return SavedEvaluationReports(
        retrieval=retrieval,
        agent=agent,
        errors=tuple(errors),
    )


def evaluate_retrieval(
    *,
    settings: Settings,
    dense: ChromaDenseIndex,
    sparse: BM25SparseIndex,
    benchmark_path: Path,
    limit: int,
    top_k: int,
) -> RetrievalEvaluationReport:
    """Run and persist a bounded comparative retrieval evaluation."""
    cases = _limited_cases(benchmark_path, limit)
    report = RetrievalEvaluator(
        dense=dense,
        sparse=sparse,
        hybrid=create_hybrid_retriever(settings, dense=dense, sparse=sparse),
        rrf_k=settings.hybrid_rrf_k,
    ).evaluate(cases, benchmark_name=benchmark_name(benchmark_path), top_k=top_k)
    save_report(report, settings.evaluation_dir / "retrieval-results.json")
    return report


def evaluate_agent(
    *,
    settings: Settings,
    agent: AgenticRAG,
    benchmark_path: Path,
    limit: int,
    top_k: int,
) -> AgentEvaluationReport:
    """Run and persist a bounded deterministic agent evaluation."""
    cases = _limited_cases(benchmark_path, limit)
    report = AgentEvaluator(agent=agent).evaluate(
        cases,
        benchmark_name=benchmark_name(benchmark_path),
        top_k=top_k,
    )
    save_report(report, settings.evaluation_dir / "agent-results.json")
    return report


def combined_regression(
    settings: Settings,
    reports: SavedEvaluationReports,
) -> RegressionResult:
    """Apply configured gates to whichever reports are currently available."""
    return regression_result(
        retrieval=reports.retrieval,
        agent=reports.agent,
        thresholds=thresholds_from_settings(settings),
    )


def agent_report_usage(report: AgentEvaluationReport) -> tuple[int, int]:
    """Return actual reported chat calls and tokens across all evaluated cases."""
    return (
        sum(result.chat_api_calls for result in report.results),
        sum(result.total_tokens for result in report.results),
    )


def _limited_cases(path: Path, limit: int) -> tuple[BenchmarkCase, ...]:
    cases = load_benchmark(path)
    if not 1 <= limit <= len(cases):
        raise ValueError(f"limit must be between 1 and {len(cases)}")
    return cases[:limit]

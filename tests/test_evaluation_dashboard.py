from pathlib import Path

from advanced_rag.config import Settings
from advanced_rag.evaluation.models import (
    AgentCaseResult,
    AgentEvaluationReport,
    AgentSummary,
)
from advanced_rag.interfaces.documents import synchronize_corpus
from advanced_rag.interfaces.evaluation_dashboard import (
    agent_report_usage,
    discover_benchmarks,
    evaluate_retrieval,
    inspect_benchmark,
    load_saved_reports,
)
from advanced_rag.runtime import create_dense_index, create_sparse_index


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "raw",
        index_dir=tmp_path / "indexes",
        chroma_dir=tmp_path / "indexes" / "chroma",
        bm25_dir=tmp_path / "indexes" / "bm25",
        chroma_collection="dashboard-test",
        embedding_provider="deterministic",
        embedding_dimensions=16,
        reranker_provider="none",
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_size=1,
        evaluation_benchmark_dir=tmp_path / "evaluations",
        evaluation_dir=tmp_path / "reports",
    )


def test_benchmark_discovery_and_overview() -> None:
    benchmarks = discover_benchmarks(Path("evaluations"))
    monetary = next(path for path in benchmarks if path.name == "monetary_policy.jsonl")

    overview = inspect_benchmark(monetary)

    assert len(overview.cases) == 33
    assert overview.answerable == 29
    assert overview.unanswerable == 4
    assert overview.categories["unanswerable"] == 4


def test_saved_report_loading_isolates_invalid_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.evaluation_dir.mkdir(parents=True)
    (settings.evaluation_dir / "retrieval-results.json").write_text(
        "not-json",
        encoding="utf-8",
    )

    reports = load_saved_reports(settings)

    assert reports.retrieval is None
    assert reports.agent is None
    assert len(reports.errors) == 1


def test_retrieval_dashboard_runner_persists_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()
    settings.evaluation_benchmark_dir.mkdir(parents=True)
    (settings.data_dir / "policy.txt").write_text(
        "Inflation expectations influence monetary policy decisions.",
        encoding="utf-8",
    )
    benchmark = settings.evaluation_benchmark_dir / "test.jsonl"
    benchmark.write_text(
        '{"id":"case-1","question":"What influences monetary policy decisions?",'
        '"category":"single_document","answerable":true,"gold_targets":['
        '{"filename":"policy.txt"}],"must_include":["inflation"]}\n',
        encoding="utf-8",
    )
    dense = create_dense_index(settings)
    sparse = create_sparse_index(settings)
    synchronize_corpus(settings, dense, sparse)

    report = evaluate_retrieval(
        settings=settings,
        dense=dense,
        sparse=sparse,
        benchmark_path=benchmark,
        limit=1,
        top_k=1,
    )

    assert report.cases == 1
    assert (settings.evaluation_dir / "retrieval-results.json").is_file()
    assert load_saved_reports(settings).retrieval == report


def test_agent_report_usage_sums_actual_case_usage() -> None:
    result = AgentCaseResult(
        case_id="case-1",
        category="exact_fact",
        answerable=True,
        refused=False,
        refusal_correct=True,
        citation_valid=True,
        expected_source_hit=True,
        concept_coverage=1,
        retrieval_attempts=1,
        rewrite_used=False,
        chat_api_calls=2,
        total_tokens=345,
        latency_ms=10,
        answer="answer",
    )
    summary = AgentSummary(
        cases=2,
        citation_validity_rate=1,
        expected_source_hit_rate=1,
        concept_coverage=1,
        refusal_accuracy=None,
        rewrite_rate=0,
        average_retrieval_attempts=1,
        average_chat_api_calls=2,
        average_total_tokens=345,
        average_latency_ms=10,
    )
    report = AgentEvaluationReport(
        benchmark="test.jsonl",
        cases=2,
        judged_entailment=False,
        results=(result, result.model_copy(update={"case_id": "case-2"})),
        summary=summary,
    )

    assert agent_report_usage(report) == (4, 690)

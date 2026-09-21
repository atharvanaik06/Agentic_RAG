"""Command-line interface for ingestion and retrieval-index operations."""

import argparse
import json
from pathlib import Path

from advanced_rag.config import Settings, get_settings
from advanced_rag.evaluation import AgentEvaluator, RetrievalEvaluator, load_benchmark
from advanced_rag.evaluation.judges import OpenAIEntailmentJudge
from advanced_rag.evaluation.models import RegressionThresholds
from advanced_rag.evaluation.reporting import (
    load_agent_report,
    load_retrieval_report,
    regression_result,
    render_markdown,
    save_markdown,
    save_report,
)
from advanced_rag.evaluation.retrieval import benchmark_name
from advanced_rag.generation.models import AgentAnswer
from advanced_rag.graph import AgenticRAG
from advanced_rag.ingestion import IngestionPipeline
from advanced_rag.interfaces.ui import launch_streamlit
from advanced_rag.retrieval.dense import ChromaDenseIndex
from advanced_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from advanced_rag.retrieval.models import DenseSearchFilters, RetrievalFilters
from advanced_rag.retrieval.sparse import BM25SparseIndex
from advanced_rag.runtime import (
    create_agent,
    create_dense_index,
    create_hybrid_retriever,
    create_sparse_index,
)


def _dense_index(settings: Settings) -> ChromaDenseIndex:
    return create_dense_index(settings)


def _sparse_index(settings: Settings) -> BM25SparseIndex:
    return create_sparse_index(settings)


def _hybrid_retriever(settings: Settings) -> HybridRetriever:
    return _configured_hybrid(settings, _dense_index(settings), _sparse_index(settings))


def _configured_hybrid(
    settings: Settings,
    dense: ChromaDenseIndex,
    sparse: BM25SparseIndex,
) -> HybridRetriever:
    return create_hybrid_retriever(settings, dense=dense, sparse=sparse)


def _agent(settings: Settings) -> AgenticRAG:
    return create_agent(settings)


def _thresholds(settings: Settings) -> RegressionThresholds:
    return RegressionThresholds(
        hybrid_recall_at_k=settings.evaluation_hybrid_recall_threshold,
        citation_validity_rate=settings.evaluation_citation_validity_threshold,
        refusal_accuracy=settings.evaluation_refusal_accuracy_threshold,
        maximum_average_retrieval_attempts=settings.evaluation_max_average_attempts,
    )


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _print_agent_answer(answer: AgentAnswer) -> None:
    print(answer.answer)
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            page = f", page {source.page_number}" if source.page_number else ""
            print(f"- [{source.label}] {source.filename}{page} (chunk {source.chunk_id})")
    diagnostics = answer.retrieval_diagnostics
    print(
        "\nRun summary: "
        f"{answer.retrieval_attempts} retrieval attempt(s), "
        f"{diagnostics.distinct_sources} source(s), "
        f"confidence={diagnostics.confidence}, "
        f"API calls={answer.usage.api_calls}, "
        f"tokens={answer.usage.input_tokens + answer.usage.output_tokens}"
    )


def _add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-id")
    parser.add_argument("--filename")
    parser.add_argument("--file-type", choices=["pdf", "markdown", "text"])
    parser.add_argument("--page", type=int)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser separately so it can be tested without exiting."""
    parser = argparse.ArgumentParser(prog="rag", description="Advanced Agentic RAG utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Inspect document ingestion")
    ingest.add_argument("source", nargs="?", type=Path)

    index = subparsers.add_parser("index", help="Synchronize documents into ChromaDB")
    index.add_argument("source", nargs="?", type=Path)
    index.add_argument(
        "--no-prune",
        action="store_true",
        help="Keep indexed sources that are absent from this directory scan",
    )

    subparsers.add_parser("index-info", help="Show dense collection information")

    search = subparsers.add_parser("search-dense", help="Run direct semantic search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=6)
    _add_filter_arguments(search)

    reset = subparsers.add_parser("reset-index", help="Permanently clear the dense collection")
    reset.add_argument("--yes", action="store_true", help="Confirm destructive deletion")

    sparse_index = subparsers.add_parser(
        "index-sparse", help="Build or refresh the local BM25 index"
    )
    sparse_index.add_argument("source", nargs="?", type=Path)

    subparsers.add_parser("sparse-info", help="Show sparse index information")

    sparse_search = subparsers.add_parser("search-sparse", help="Run BM25 keyword search")
    sparse_search.add_argument("query")
    sparse_search.add_argument("--top-k", type=int, default=6)
    _add_filter_arguments(sparse_search)

    sparse_reset = subparsers.add_parser(
        "reset-sparse", help="Permanently clear the local BM25 index"
    )
    sparse_reset.add_argument("--yes", action="store_true", help="Confirm destructive deletion")

    hybrid_search = subparsers.add_parser(
        "search-hybrid", help="Run fused dense/BM25 search with reranking"
    )
    hybrid_search.add_argument("query")
    hybrid_search.add_argument("--top-k", type=int, default=6)
    _add_filter_arguments(hybrid_search)

    compare = subparsers.add_parser(
        "compare-retrievers", help="Compare dense, sparse, and fused rankings"
    )
    compare.add_argument("query")
    compare.add_argument("--top-k", type=int, default=6)
    _add_filter_arguments(compare)

    ask = subparsers.add_parser("ask", help="Run the bounded LangGraph RAG workflow")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--json", action="store_true", help="Print the complete structured result")
    _add_filter_arguments(ask)

    retrieval_eval = subparsers.add_parser(
        "evaluate-retrieval", help="Benchmark dense, sparse, fusion, and reranking"
    )
    retrieval_eval.add_argument("benchmark", type=Path)
    retrieval_eval.add_argument("--top-k", type=int, default=5)
    retrieval_eval.add_argument("--limit", type=int)
    retrieval_eval.add_argument("--output-dir", type=Path)
    retrieval_eval.add_argument("--enforce", action="store_true")

    agent_eval = subparsers.add_parser(
        "evaluate-agent", help="Benchmark the complete LangGraph answer workflow"
    )
    agent_eval.add_argument("benchmark", type=Path)
    agent_eval.add_argument("--top-k", type=int, default=6)
    agent_eval.add_argument("--limit", type=int)
    agent_eval.add_argument("--output-dir", type=Path)
    agent_eval.add_argument("--judge-entailment", action="store_true")
    agent_eval.add_argument("--enforce", action="store_true")

    report = subparsers.add_parser(
        "evaluate-report", help="Render Markdown from saved evaluation JSON"
    )
    report.add_argument("--retrieval", type=Path)
    report.add_argument("--agent", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--enforce", action="store_true")

    ui = subparsers.add_parser("ui", help="Launch the local Streamlit interface")
    ui.add_argument("--address", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--headless", action="store_true", help="Do not open a browser window")
    return parser


def main() -> int:
    """Execute one CLI operation and return a process exit code."""
    args = build_parser().parse_args()
    settings = get_settings()

    if args.command == "ui":
        return launch_streamlit(address=args.address, port=args.port, headless=args.headless)

    if args.command == "ingest":
        ingestion_result = IngestionPipeline(settings).ingest(args.source or settings.data_dir)
        _print_json(
            {
                "discovered_files": ingestion_result.discovered_files,
                "processed_files": ingestion_result.processed_files,
                "skipped_files": ingestion_result.skipped_files,
                "failed_files": ingestion_result.failed_files,
                "chunks": len(ingestion_result.chunks),
                "issues": [issue.model_dump(mode="json") for issue in ingestion_result.issues],
            }
        )
        return 1 if ingestion_result.failed_files else 0

    if args.command == "index":
        dense_index = _dense_index(settings)
        index_result, ingestion = dense_index.index_path(
            args.source or settings.data_dir,
            pipeline=IngestionPipeline(settings),
            prune_missing=not args.no_prune,
        )
        _print_json(
            {
                **index_result.model_dump(mode="json"),
                "issues": [issue.model_dump(mode="json") for issue in ingestion.issues],
            }
        )
        return 1 if ingestion.failed_files else 0

    if args.command == "index-info":
        dense_index = _dense_index(settings)
        _print_json(dense_index.info().model_dump(mode="json"))
        return 0

    if args.command == "search-dense":
        dense_index = _dense_index(settings)
        filters = DenseSearchFilters(
            source_id=args.source_id,
            filename=args.filename,
            file_type=args.file_type,
            page_number=args.page,
        )
        results = dense_index.search(args.query, top_k=args.top_k, filters=filters)
        _print_json([result.model_dump(mode="json") for result in results])
        return 0

    if args.command == "reset-index":
        if not args.yes:
            print("Refusing to delete the index without --yes")
            return 2
        dense_index = _dense_index(settings)
        dense_index.reset()
        _print_json(dense_index.info().model_dump(mode="json"))
        return 0

    if args.command == "search-hybrid":
        hybrid_filters = RetrievalFilters(
            source_id=args.source_id,
            filename=args.filename,
            file_type=args.file_type,
            page_number=args.page,
        )
        hybrid_response = _hybrid_retriever(settings).search(
            args.query,
            top_k=args.top_k,
            filters=hybrid_filters,
        )
        _print_json(hybrid_response.model_dump(mode="json"))
        return 0

    if args.command == "compare-retrievers":
        comparison_filters = RetrievalFilters(
            source_id=args.source_id,
            filename=args.filename,
            file_type=args.file_type,
            page_number=args.page,
        )
        comparison_dense = _dense_index(settings).search(
            args.query,
            top_k=args.top_k,
            filters=DenseSearchFilters.model_validate(comparison_filters.model_dump()),
        )
        comparison_sparse = _sparse_index(settings).search(
            args.query,
            top_k=args.top_k,
            filters=comparison_filters,
        )
        comparison_fused = reciprocal_rank_fusion(
            comparison_dense,
            comparison_sparse,
            rrf_k=settings.hybrid_rrf_k,
        )
        _print_json(
            {
                "query": args.query,
                "dense": [item.model_dump(mode="json") for item in comparison_dense],
                "sparse": [item.model_dump(mode="json") for item in comparison_sparse],
                "fused": [item.model_dump(mode="json") for item in comparison_fused[: args.top_k]],
            }
        )
        return 0

    if args.command == "ask":
        ask_filters = RetrievalFilters(
            source_id=args.source_id,
            filename=args.filename,
            file_type=args.file_type,
            page_number=args.page,
        )
        agent_answer = _agent(settings).ask(
            args.question,
            filters=ask_filters,
            top_k=args.top_k,
        )
        if args.json:
            _print_json(agent_answer.model_dump(mode="json"))
        else:
            _print_agent_answer(agent_answer)
        return 0

    if args.command == "evaluate-retrieval":
        cases = load_benchmark(args.benchmark)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            cases = cases[: args.limit]
        dense = _dense_index(settings)
        sparse = _sparse_index(settings)
        evaluator = RetrievalEvaluator(
            dense=dense,
            sparse=sparse,
            hybrid=_configured_hybrid(settings, dense, sparse),
            rrf_k=settings.hybrid_rrf_k,
        )
        retrieval_evaluation = evaluator.evaluate(
            cases,
            benchmark_name=benchmark_name(args.benchmark),
            top_k=args.top_k,
        )
        output_dir = args.output_dir or settings.evaluation_dir
        output_path = save_report(retrieval_evaluation, output_dir / "retrieval-results.json")
        regression = regression_result(
            retrieval=retrieval_evaluation,
            agent=None,
            thresholds=_thresholds(settings),
        )
        _print_json(
            {
                "report": str(output_path),
                "summaries": [
                    item.model_dump(mode="json") for item in retrieval_evaluation.summaries
                ],
                "reranker": retrieval_evaluation.reranker.model_dump(mode="json"),
                "regression": regression.model_dump(mode="json"),
            }
        )
        return 1 if args.enforce and not regression.passed else 0

    if args.command == "evaluate-agent":
        cases = load_benchmark(args.benchmark)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            cases = cases[: args.limit]
        judge = None
        evidence_store = None
        if args.judge_entailment:
            api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
            judge = OpenAIEntailmentJudge(
                api_key=api_key,
                model=settings.evaluation_entailment_model,
            )
            evidence_store = _sparse_index(settings)
        agent_evaluation = AgentEvaluator(
            agent=_agent(settings),
            judge=judge,
            evidence_store=evidence_store,
        ).evaluate(
            cases,
            benchmark_name=benchmark_name(args.benchmark),
            top_k=args.top_k,
        )
        output_dir = args.output_dir or settings.evaluation_dir
        output_path = save_report(agent_evaluation, output_dir / "agent-results.json")
        regression = regression_result(
            retrieval=None,
            agent=agent_evaluation,
            thresholds=_thresholds(settings),
        )
        _print_json(
            {
                "report": str(output_path),
                "summary": agent_evaluation.summary.model_dump(mode="json"),
                "regression": regression.model_dump(mode="json"),
            }
        )
        return 1 if args.enforce and not regression.passed else 0

    if args.command == "evaluate-report":
        retrieval_path = args.retrieval or settings.evaluation_dir / "retrieval-results.json"
        agent_path = args.agent or settings.evaluation_dir / "agent-results.json"
        retrieval_report = (
            load_retrieval_report(retrieval_path) if retrieval_path.is_file() else None
        )
        agent_report = load_agent_report(agent_path) if agent_path.is_file() else None
        if retrieval_report is None and agent_report is None:
            raise ValueError("No retrieval or agent report was found")
        regression = regression_result(
            retrieval=retrieval_report,
            agent=agent_report,
            thresholds=_thresholds(settings),
        )
        output = args.output or settings.evaluation_dir / "evaluation-summary.md"
        save_markdown(
            render_markdown(
                retrieval=retrieval_report,
                agent=agent_report,
                regression=regression,
            ),
            output,
        )
        _print_json({"report": str(output), "regression": regression.model_dump(mode="json")})
        return 1 if args.enforce and not regression.passed else 0

    sparse_index = _sparse_index(settings)
    if args.command == "index-sparse":
        sparse_result, sparse_ingestion = sparse_index.index_path(
            args.source or settings.data_dir,
            pipeline=IngestionPipeline(settings),
        )
        _print_json(
            {
                **sparse_result.model_dump(mode="json"),
                "issues": [issue.model_dump(mode="json") for issue in sparse_ingestion.issues],
            }
        )
        return 1 if sparse_ingestion.failed_files else 0

    if args.command == "sparse-info":
        _print_json(sparse_index.info().model_dump(mode="json"))
        return 0

    if args.command == "search-sparse":
        sparse_filters = RetrievalFilters(
            source_id=args.source_id,
            filename=args.filename,
            file_type=args.file_type,
            page_number=args.page,
        )
        sparse_results = sparse_index.search(args.query, top_k=args.top_k, filters=sparse_filters)
        _print_json([result.model_dump(mode="json") for result in sparse_results])
        return 0

    if args.command == "reset-sparse":
        if not args.yes:
            print("Refusing to delete the sparse index without --yes")
            return 2
        sparse_index.reset()
        _print_json(sparse_index.info().model_dump(mode="json"))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

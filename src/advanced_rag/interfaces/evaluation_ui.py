"""Streamlit rendering for benchmark execution and saved evaluation reports."""

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from advanced_rag.config import Settings
from advanced_rag.evaluation.models import AgentEvaluationReport, RetrievalEvaluationReport
from advanced_rag.graph import AgenticRAG, bounded_retrieval_attempts
from advanced_rag.interfaces.evaluation_dashboard import (
    SavedEvaluationReports,
    agent_report_usage,
    combined_regression,
    discover_benchmarks,
    evaluate_agent,
    evaluate_retrieval,
    inspect_benchmark,
    load_saved_reports,
    thresholds_from_settings,
)
from advanced_rag.interfaces.spending import (
    SessionUsage,
    SpendingLimits,
    conservative_request_token_reserve,
    spending_decision,
)
from advanced_rag.retrieval.dense import ChromaDenseIndex
from advanced_rag.retrieval.sparse import BM25SparseIndex


def render_evaluation_dashboard(
    *,
    settings: Settings,
    dense: ChromaDenseIndex,
    sparse: BM25SparseIndex,
    agent_factory: Callable[[], AgenticRAG],
    limits: SpendingLimits,
    usage: SessionUsage,
    indexes_ready: bool,
) -> None:
    """Render saved metrics and guarded benchmark execution controls."""
    with st.expander("Evaluation dashboard", expanded=False):
        _render_notice()
        benchmarks = discover_benchmarks(settings.evaluation_benchmark_dir)
        if not benchmarks:
            st.warning(
                f"No `.jsonl` benchmarks were found in `{settings.evaluation_benchmark_dir}`."
            )
            return

        default_index = next(
            (
                position
                for position, path in enumerate(benchmarks)
                if path.name == "monetary_policy.jsonl"
            ),
            0,
        )
        selected = st.selectbox(
            "Benchmark",
            options=benchmarks,
            index=default_index,
            format_func=lambda path: path.name,
        )
        overview = inspect_benchmark(selected)
        reports = load_saved_reports(settings)
        for error in reports.errors:
            st.error(error)

        metric_columns = st.columns(4)
        metric_columns[0].metric("Benchmark cases", len(overview.cases))
        metric_columns[1].metric("Answerable", overview.answerable)
        metric_columns[2].metric("Should refuse", overview.unanswerable)
        metric_columns[3].metric("Categories", len(overview.categories))
        st.caption(
            "Categories: "
            + " · ".join(f"{category}: {count}" for category, count in overview.categories.items())
        )

        overview_tab, retrieval_tab, agent_tab, run_tab = st.tabs(
            ["Quality gates", "Retrieval report", "Agent report", "Run evaluations"]
        )
        with overview_tab:
            _render_quality_gates(settings, reports)
        with retrieval_tab:
            _render_retrieval_report(reports.retrieval)
        with agent_tab:
            _render_agent_report(reports.agent)
        with run_tab:
            _render_run_controls(
                settings=settings,
                dense=dense,
                sparse=sparse,
                agent_factory=agent_factory,
                benchmark_path=selected,
                case_count=len(overview.cases),
                reports=reports,
                limits=limits,
                usage=usage,
                indexes_ready=indexes_ready,
            )


def _render_quality_gates(settings: Settings, reports: SavedEvaluationReports) -> None:
    if reports.retrieval is None and reports.agent is None:
        st.info("Run an evaluation or use the CLI to create saved reports under `reports/`.")
        return
    if (
        reports.retrieval is not None
        and reports.agent is not None
        and reports.retrieval.benchmark != reports.agent.benchmark
    ):
        st.warning("The saved retrieval and agent reports use different benchmarks.")
    regression = combined_regression(settings, reports)
    if regression.passed:
        st.success("All available regression gates pass.")
    else:
        st.error("One or more regression gates fail.")
    thresholds = thresholds_from_settings(settings)
    labels = {
        "hybrid_recall_at_k": f"Hybrid Recall@k ≥ {thresholds.hybrid_recall_at_k:.2f}",
        "citation_validity_rate": (f"Citation validity ≥ {thresholds.citation_validity_rate:.2f}"),
        "refusal_accuracy": f"Refusal accuracy ≥ {thresholds.refusal_accuracy:.2f}",
        "maximum_average_retrieval_attempts": (
            f"Average retrieval attempts ≤ {thresholds.maximum_average_retrieval_attempts:.2f}"
        ),
    }
    for name, passed in regression.checks.items():
        st.markdown(f"- {'✅' if passed else '❌'} {labels.get(name, name)}")


def _render_retrieval_report(report: RetrievalEvaluationReport | None) -> None:
    if report is None:
        st.info("No saved retrieval report is available.")
        return
    st.caption(
        f"Benchmark: `{report.benchmark}` · {report.cases} answerable cases · top-k {report.top_k}"
    )
    st.dataframe(
        [
            {
                "Method": summary.method,
                "Recall@k": summary.recall_at_k,
                "Precision@k": summary.precision_at_k,
                "MRR": summary.mrr,
                "MAP": summary.map,
                "NDCG@k": summary.ndcg_at_k,
                "Source hit": summary.source_hit_rate,
                "Page hit": summary.page_hit_rate,
                "Latency ms": summary.average_latency_ms,
            }
            for summary in report.summaries
        ],
        use_container_width=True,
        hide_index=True,
    )
    reranker = report.reranker
    columns = st.columns(4)
    columns[0].metric("Reranker improved", reranker.improved)
    columns[1].metric("Unchanged", reranker.unchanged)
    columns[2].metric("Worsened", reranker.worsened)
    columns[3].metric("Missing", reranker.missing)
    if st.checkbox("Show per-case retrieval results", key="show_retrieval_cases"):
        st.dataframe(
            [
                {
                    "Case": result.case_id,
                    "Category": result.category,
                    "Method": result.method,
                    "Recall@k": result.recall_at_k,
                    "MRR": result.reciprocal_rank,
                    "Source hit": result.source_hit,
                    "Page hit": result.page_hit,
                    "Latency ms": result.latency_ms,
                    "Error": result.error or "",
                }
                for result in report.results
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_agent_report(report: AgentEvaluationReport | None) -> None:
    if report is None:
        st.info("No saved agent report is available.")
        return
    summary = report.summary
    st.caption(
        f"Benchmark: `{report.benchmark}` · {report.cases} cases · "
        f"entailment judge: {'enabled' if report.judged_entailment else 'disabled'}"
    )
    first = st.columns(4)
    first[0].metric("Citation validity", _percent(summary.citation_validity_rate))
    first[1].metric("Source hit", _percent(summary.expected_source_hit_rate))
    first[2].metric("Concept coverage", _percent(summary.concept_coverage))
    first[3].metric("Refusal accuracy", _optional_percent(summary.refusal_accuracy))
    second = st.columns(4)
    second[0].metric("Rewrite rate", _percent(summary.rewrite_rate))
    second[1].metric("Avg attempts", f"{summary.average_retrieval_attempts:.2f}")
    second[2].metric("Avg chat calls", f"{summary.average_chat_api_calls:.2f}")
    second[3].metric("Avg tokens", f"{summary.average_total_tokens:,.0f}")
    if st.checkbox("Show per-case agent results", key="show_agent_cases"):
        st.dataframe(
            [
                {
                    "Case": result.case_id,
                    "Category": result.category,
                    "Answerable": result.answerable,
                    "Refused": result.refused,
                    "Refusal correct": result.refusal_correct,
                    "Citation valid": result.citation_valid,
                    "Source hit": result.expected_source_hit,
                    "Coverage": result.concept_coverage,
                    "Attempts": result.retrieval_attempts,
                    "Chat calls": result.chat_api_calls,
                    "Tokens": result.total_tokens,
                    "Latency ms": result.latency_ms,
                    "Errors": " ".join(result.errors),
                }
                for result in report.results
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_run_controls(
    *,
    settings: Settings,
    dense: ChromaDenseIndex,
    sparse: BM25SparseIndex,
    agent_factory: Callable[[], AgenticRAG],
    benchmark_path: Path,
    case_count: int,
    reports: SavedEvaluationReports,
    limits: SpendingLimits,
    usage: SessionUsage,
    indexes_ready: bool,
) -> None:
    st.warning(
        "Evaluation can make paid API calls. Start with a small limit and inspect the report "
        "before running the complete benchmark."
    )
    top_k = st.slider("Evaluation top-k", min_value=1, max_value=10, value=5)
    limit = int(
        st.number_input(
            "Evaluation case limit",
            min_value=1,
            max_value=case_count,
            value=min(3, case_count),
            step=1,
        )
    )
    if not indexes_ready:
        st.error("Both matching indexes and an API key are required before evaluation.")

    st.markdown("#### Retrieval evaluation")
    st.caption(
        "Compares dense, BM25, RRF, and reranked retrieval. Each answerable case can make "
        "two query-embedding requests; BM25 and reranking are local."
    )
    retrieval_confirmed = st.checkbox(
        "I understand retrieval evaluation can use embedding API calls",
        key="evaluation_retrieval_confirmed",
    )
    replace_retrieval = reports.retrieval is None or st.checkbox(
        "Replace the saved retrieval report",
        key="evaluation_replace_retrieval",
    )
    if st.button(
        "Run retrieval evaluation",
        disabled=not indexes_ready or not retrieval_confirmed or not replace_retrieval,
    ):
        try:
            with st.spinner("Evaluating all retrieval stages..."):
                retrieval_report = evaluate_retrieval(
                    settings=settings,
                    dense=dense,
                    sparse=sparse,
                    benchmark_path=benchmark_path,
                    limit=limit,
                    top_k=top_k,
                )
        except Exception as exc:
            st.error("Retrieval evaluation failed.")
            st.exception(exc)
        else:
            errors = sum(result.error is not None for result in retrieval_report.results)
            _set_notice(
                "warning" if errors else "success",
                f"Saved retrieval report for {retrieval_report.cases} answerable case(s); "
                f"{errors} method error(s).",
            )
            st.rerun()

    st.divider()
    st.markdown("#### Agent evaluation")
    attempts = bounded_retrieval_attempts(
        settings.agent_max_retrieval_attempts,
        settings.agent_max_steps,
    )
    calls_per_case = attempts * (2 if settings.agent_semantic_evidence_grading else 1)
    tokens_per_case = conservative_request_token_reserve(
        settings,
        attempts,
        semantic_evidence_grading=settings.agent_semantic_evidence_grading,
    )
    decision = spending_decision(
        limits,
        usage,
        question_reserve=limit,
        api_call_reserve=limit * calls_per_case,
        token_reserve=limit * tokens_per_case,
    )
    st.caption(
        f"Conservative reserve for {limit} case(s): {limit * calls_per_case} chat calls and "
        f"{limit * tokens_per_case:,} tokens. Actual provider-reported usage is recorded."
    )
    if not decision.allowed:
        st.warning("Agent evaluation is paused. " + " ".join(decision.reasons))
    agent_confirmed = st.checkbox(
        "I understand agent evaluation uses chat-model calls",
        key="evaluation_agent_confirmed",
    )
    replace_agent = reports.agent is None or st.checkbox(
        "Replace the saved agent report",
        key="evaluation_replace_agent",
    )
    if st.button(
        "Run agent evaluation",
        disabled=(
            not indexes_ready or not decision.allowed or not agent_confirmed or not replace_agent
        ),
    ):
        try:
            with st.spinner("Running bounded agent cases sequentially..."):
                agent_report = evaluate_agent(
                    settings=settings,
                    agent=agent_factory(),
                    benchmark_path=benchmark_path,
                    limit=limit,
                    top_k=top_k,
                )
        except Exception as exc:
            st.error("Agent evaluation failed.")
            st.exception(exc)
        else:
            api_calls, tokens = agent_report_usage(agent_report)
            st.session_state["session_usage"] = usage.add(
                questions=agent_report.cases,
                api_calls=api_calls,
                tokens=tokens,
            )
            failed_cases = sum(bool(result.errors) for result in agent_report.results)
            _set_notice(
                "warning" if failed_cases else "success",
                f"Saved agent report for {agent_report.cases} case(s); used {api_calls} chat calls "
                f"and {tokens:,} tokens; {failed_cases} case(s) reported errors.",
            )
            st.rerun()

    st.caption(
        "Optional claim-level entailment judging remains available through "
        "`rag evaluate-agent --judge-entailment`; it is excluded here because its call count "
        "depends on generated claim count."
    )


def _render_notice() -> None:
    notice = st.session_state.pop("evaluation_notice", None)
    if isinstance(notice, tuple) and len(notice) == 2:
        level, message = notice
        getattr(st, level, st.info)(message)


def _set_notice(level: str, message: str) -> None:
    st.session_state["evaluation_notice"] = (level, message)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _optional_percent(value: float | None) -> str:
    return "N/A" if value is None else _percent(value)

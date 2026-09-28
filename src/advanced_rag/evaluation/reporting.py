"""Persist machine-readable results, render Markdown, and enforce quality gates."""

from dataclasses import dataclass
from pathlib import Path

from advanced_rag.evaluation.models import (
    AgentEvaluationReport,
    RegressionResult,
    RegressionThresholds,
    RetrievalEvaluationReport,
)


@dataclass(frozen=True)
class AnswerableRefusalDiagnostics:
    """Separate incorrect refusals with and without an expected-source retrieval hit."""

    incorrect_refusal_ids: tuple[str, ...]
    expected_source_retrieved_ids: tuple[str, ...]
    expected_source_missing_ids: tuple[str, ...]
    unavailable_ids: tuple[str, ...]


def save_report(
    report: RetrievalEvaluationReport | AgentEvaluationReport,
    path: Path | str,
) -> Path:
    """Write a validated report as formatted JSON."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report_path


def load_retrieval_report(path: Path | str) -> RetrievalEvaluationReport:
    return RetrievalEvaluationReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_agent_report(path: Path | str) -> AgentEvaluationReport:
    report = AgentEvaluationReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    success_rate, refusal_rate = _answerable_rates(report)
    if report.summary.answerable_success_rate is None and success_rate is not None:
        report = report.model_copy(
            update={
                "summary": report.summary.model_copy(
                    update={
                        "answerable_success_rate": success_rate,
                        "answerable_refusal_rate": refusal_rate,
                    }
                )
            }
        )
    return report


def answerable_refusal_diagnostics(
    *,
    retrieval: RetrievalEvaluationReport,
    agent: AgentEvaluationReport,
) -> AnswerableRefusalDiagnostics:
    """Correlate incorrect refusals with final hybrid expected-source retrieval hits."""
    if retrieval.benchmark != agent.benchmark:
        raise ValueError("Retrieval and agent reports must use the same benchmark")
    final_hybrid = {
        result.case_id: result for result in retrieval.results if result.method == "hybrid_reranked"
    }
    incorrect = tuple(
        result.case_id for result in agent.results if result.answerable and result.refused
    )
    retrieved = tuple(
        case_id
        for case_id in incorrect
        if case_id in final_hybrid and final_hybrid[case_id].source_hit
    )
    missing = tuple(
        case_id
        for case_id in incorrect
        if case_id in final_hybrid and not final_hybrid[case_id].source_hit
    )
    unavailable = tuple(case_id for case_id in incorrect if case_id not in final_hybrid)
    return AnswerableRefusalDiagnostics(
        incorrect_refusal_ids=incorrect,
        expected_source_retrieved_ids=retrieved,
        expected_source_missing_ids=missing,
        unavailable_ids=unavailable,
    )


def regression_result(
    *,
    retrieval: RetrievalEvaluationReport | None,
    agent: AgentEvaluationReport | None,
    thresholds: RegressionThresholds,
) -> RegressionResult:
    """Evaluate only gates whose corresponding report was supplied."""
    checks: dict[str, bool] = {}
    if retrieval is not None:
        hybrid = next(
            summary for summary in retrieval.summaries if summary.method == "hybrid_reranked"
        )
        checks["hybrid_recall_at_k"] = hybrid.recall_at_k >= thresholds.hybrid_recall_at_k
    if agent is not None:
        checks["citation_validity_rate"] = (
            agent.summary.citation_validity_rate >= thresholds.citation_validity_rate
        )
        if any(result.answerable for result in agent.results):
            answerable_success_rate, _ = _answerable_rates(agent)
            checks["answerable_success_rate"] = (
                answerable_success_rate is not None
                and answerable_success_rate >= thresholds.answerable_success_rate
            )
        if any(not result.answerable for result in agent.results):
            refusal_accuracy = agent.summary.refusal_accuracy
            checks["refusal_accuracy"] = (
                refusal_accuracy is not None and refusal_accuracy >= thresholds.refusal_accuracy
            )
        checks["maximum_average_retrieval_attempts"] = (
            agent.summary.average_retrieval_attempts
            <= thresholds.maximum_average_retrieval_attempts
        )
    return RegressionResult(passed=all(checks.values()), checks=checks)


def render_markdown(
    *,
    retrieval: RetrievalEvaluationReport | None,
    agent: AgentEvaluationReport | None,
    regression: RegressionResult,
) -> str:
    """Render a compact GitHub-friendly evaluation summary."""
    lines = ["# Evaluation Summary", ""]
    if retrieval is not None:
        lines.extend(
            [
                f"Benchmark: `{retrieval.benchmark}` ({retrieval.cases} answerable cases)",
                "",
                "## Retrieval",
                "",
                "| Method | Recall@k | Precision@k | MRR | MAP | NDCG@k | "
                "Source hit | Latency ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for retrieval_summary in retrieval.summaries:
            lines.append(
                f"| {retrieval_summary.method} | {retrieval_summary.recall_at_k:.3f} | "
                f"{retrieval_summary.precision_at_k:.3f} | {retrieval_summary.mrr:.3f} | "
                f"{retrieval_summary.map:.3f} | {retrieval_summary.ndcg_at_k:.3f} | "
                f"{retrieval_summary.source_hit_rate:.3f} | "
                f"{retrieval_summary.average_latency_ms:.1f} |"
            )
        reranker = retrieval.reranker
        lines.extend(
            [
                "",
                "### Reranker movement",
                "",
                f"Improved: {reranker.improved}; unchanged: {reranker.unchanged}; "
                f"worsened: {reranker.worsened}; missing: {reranker.missing}; "
                f"average rank change: {reranker.average_rank_change:.3f}.",
                "",
            ]
        )
    if agent is not None:
        agent_summary = agent.summary
        lines.extend(
            [
                "## Agent",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Citation validity | {agent_summary.citation_validity_rate:.3f} |",
                f"| Expected-source hit | {agent_summary.expected_source_hit_rate:.3f} |",
                f"| Concept coverage | {agent_summary.concept_coverage:.3f} |",
                "| Answerable success | "
                + (
                    f"{agent_summary.answerable_success_rate:.3f} |"
                    if agent_summary.answerable_success_rate is not None
                    else "N/A |"
                ),
                "| Incorrect refusal | "
                + (
                    f"{agent_summary.answerable_refusal_rate:.3f} |"
                    if agent_summary.answerable_refusal_rate is not None
                    else "N/A |"
                ),
                "| Refusal accuracy | "
                + (
                    f"{agent_summary.refusal_accuracy:.3f} |"
                    if agent_summary.refusal_accuracy is not None
                    else "N/A |"
                ),
                f"| Rewrite rate | {agent_summary.rewrite_rate:.3f} |",
                f"| Average retrieval attempts | {agent_summary.average_retrieval_attempts:.3f} |",
                f"| Average chat calls | {agent_summary.average_chat_api_calls:.3f} |",
                f"| Average tokens | {agent_summary.average_total_tokens:.1f} |",
                f"| Average latency ms | {agent_summary.average_latency_ms:.1f} |",
                "",
            ]
        )
        if agent_summary.entailment_support_rate is not None:
            lines.append(
                f"Optional entailment support rate: {agent_summary.entailment_support_rate:.3f}."
            )
            lines.append("")
    if retrieval is not None and agent is not None and retrieval.benchmark == agent.benchmark:
        diagnostics = answerable_refusal_diagnostics(retrieval=retrieval, agent=agent)
        lines.extend(
            [
                "## Incorrect-refusal diagnosis",
                "",
                f"Answerable cases refused: {len(diagnostics.incorrect_refusal_ids)}.",
                "",
                "- Expected source retrieved by final hybrid search: "
                f"{len(diagnostics.expected_source_retrieved_ids)} "
                f"({', '.join(diagnostics.expected_source_retrieved_ids) or 'none'})",
                "- Expected source missing from final hybrid search: "
                f"{len(diagnostics.expected_source_missing_ids)} "
                f"({', '.join(diagnostics.expected_source_missing_ids) or 'none'})",
                "- No comparable retrieval case: "
                f"{len(diagnostics.unavailable_ids)} "
                f"({', '.join(diagnostics.unavailable_ids) or 'none'})",
                "",
                "A source-level hit does not prove that the selected chunk directly supports "
                "the answer; inspect passage content before attributing the refusal to grading.",
                "",
            ]
        )
    status = "PASS" if regression.passed else "FAIL"
    lines.extend(["## Regression gates", "", f"Overall: **{status}**", ""])
    for name, passed in regression.checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    return "\n".join(lines) + "\n"


def save_markdown(content: str, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


def _answerable_rates(agent: AgentEvaluationReport) -> tuple[float | None, float | None]:
    answerable = [result for result in agent.results if result.answerable]
    if not answerable:
        return None, None
    success_rate = sum(not result.refused for result in answerable) / len(answerable)
    return success_rate, 1.0 - success_rate

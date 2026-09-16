"""Persist machine-readable results, render Markdown, and enforce quality gates."""

from pathlib import Path

from advanced_rag.evaluation.models import (
    AgentEvaluationReport,
    RegressionResult,
    RegressionThresholds,
    RetrievalEvaluationReport,
)


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
    return AgentEvaluationReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


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

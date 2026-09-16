"""End-to-end LangGraph answer, routing, citation, and refusal evaluation."""

import re
from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from advanced_rag.evaluation.judges import EntailmentJudge
from advanced_rag.evaluation.metrics import mean
from advanced_rag.evaluation.models import (
    AgentCaseResult,
    AgentEvaluationReport,
    AgentSummary,
    BenchmarkCase,
    EntailmentResult,
)
from advanced_rag.generation.models import AgentAnswer
from advanced_rag.ingestion.models import DocumentChunk

_CLAIM_PATTERN = re.compile(r"^-\s+(.*?)\s+((?:\[S\d+\]\s*)+)$")
_LABEL_PATTERN = re.compile(r"\[(S\d+)\]")


class AnsweringAgent(Protocol):
    def ask(self, question: str, *, top_k: int | None = None) -> AgentAnswer: ...


class EvidenceStore(Protocol):
    def get_chunks(self, chunk_ids: set[str]) -> dict[str, DocumentChunk]: ...


class AgentEvaluator:
    """Evaluate the complete graph with deterministic checks and an optional judge."""

    def __init__(
        self,
        *,
        agent: AnsweringAgent,
        judge: EntailmentJudge | None = None,
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        if judge is not None and evidence_store is None:
            raise ValueError("An evidence store is required when an entailment judge is enabled")
        self.agent = agent
        self.judge = judge
        self.evidence_store = evidence_store

    def evaluate(
        self,
        cases: Sequence[BenchmarkCase],
        *,
        benchmark_name: str,
        top_k: int = 6,
    ) -> AgentEvaluationReport:
        """Run all cases sequentially so token use and failures remain auditable."""
        results = [self._evaluate_case(case, top_k) for case in cases]
        answerable = [item for item in results if item.answerable]
        refusal_cases = [item for item in results if not item.answerable]
        entailment = [verdict for item in results for verdict in item.entailment]
        supported = sum(verdict.label == "supported" for verdict in entailment)
        summary = AgentSummary(
            cases=len(results),
            citation_validity_rate=mean([float(item.citation_valid) for item in results]),
            expected_source_hit_rate=mean([float(item.expected_source_hit) for item in answerable]),
            concept_coverage=mean([item.concept_coverage for item in answerable]),
            refusal_accuracy=(
                mean([float(item.refusal_correct) for item in refusal_cases])
                if refusal_cases
                else None
            ),
            rewrite_rate=mean([float(item.rewrite_used) for item in results]),
            average_retrieval_attempts=mean([float(item.retrieval_attempts) for item in results]),
            average_chat_api_calls=mean([float(item.chat_api_calls) for item in results]),
            average_total_tokens=mean([float(item.total_tokens) for item in results]),
            average_latency_ms=mean([item.latency_ms for item in results]),
            entailment_support_rate=(supported / len(entailment) if entailment else None),
        )
        return AgentEvaluationReport(
            benchmark=benchmark_name,
            cases=len(results),
            judged_entailment=self.judge is not None,
            results=tuple(results),
            summary=summary,
        )

    def _evaluate_case(self, case: BenchmarkCase, top_k: int) -> AgentCaseResult:
        started = perf_counter()
        try:
            answer = self.agent.ask(case.question, top_k=top_k)
        except Exception as exc:
            return AgentCaseResult(
                case_id=case.id,
                category=case.category,
                answerable=case.answerable,
                refused=True,
                refusal_correct=not case.answerable,
                citation_valid=False,
                expected_source_hit=not case.answerable,
                concept_coverage=0.0,
                retrieval_attempts=0,
                rewrite_used=False,
                chat_api_calls=0,
                total_tokens=0,
                latency_ms=(perf_counter() - started) * 1000,
                answer="",
                errors=(f"Agent evaluation failed: {exc}",),
            )
        latency_ms = (perf_counter() - started) * 1000
        answer_lower = answer.answer.lower()
        concepts_found = sum(term.lower() in answer_lower for term in case.must_include)
        concept_coverage = concepts_found / len(case.must_include) if case.must_include else 1.0
        expected_filenames = {target.filename for target in case.gold_targets}
        actual_filenames = {source.filename for source in answer.sources}
        source_hit = bool(expected_filenames & actual_filenames) if case.answerable else True
        refused = answer.insufficient_evidence
        refusal_correct = refused == (not case.answerable)
        rewrite_used = any(event.node == "rewrite_query" for event in answer.graph_trace)
        entailment = self._judge_answer(answer) if self.judge else ()
        return AgentCaseResult(
            case_id=case.id,
            category=case.category,
            answerable=case.answerable,
            refused=refused,
            refusal_correct=refusal_correct,
            citation_valid=answer.citation_validation.valid,
            expected_source_hit=source_hit,
            concept_coverage=concept_coverage,
            retrieval_attempts=answer.retrieval_attempts,
            rewrite_used=rewrite_used,
            chat_api_calls=answer.usage.api_calls,
            total_tokens=answer.usage.input_tokens + answer.usage.output_tokens,
            latency_ms=latency_ms,
            entailment=entailment,
            answer=answer.answer,
            errors=answer.errors,
        )

    def _judge_answer(self, answer: AgentAnswer) -> tuple[EntailmentResult, ...]:
        assert self.judge is not None
        assert self.evidence_store is not None
        source_by_label = {source.label: source for source in answer.sources}
        chunk_ids = {source.chunk_id for source in answer.sources}
        chunks = self.evidence_store.get_chunks(chunk_ids)
        verdicts: list[EntailmentResult] = []
        for line in answer.answer.splitlines():
            match = _CLAIM_PATTERN.match(line.strip())
            if not match:
                continue
            claim = match.group(1)
            labels = tuple(_LABEL_PATTERN.findall(match.group(2)))
            evidence = [
                chunks[source_by_label[label].chunk_id]
                for label in labels
                if label in source_by_label and source_by_label[label].chunk_id in chunks
            ]
            try:
                verdict = self.judge.judge(claim=claim, citations=labels, evidence=evidence)
            except Exception as exc:
                verdict = EntailmentResult(
                    claim=claim,
                    citations=labels,
                    label="unsupported",
                    explanation=f"Judge failed: {exc}",
                )
            verdicts.append(verdict)
        return tuple(verdicts)

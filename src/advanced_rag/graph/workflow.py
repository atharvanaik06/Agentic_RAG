"""Bounded LangGraph workflow for grounded, citation-validated RAG answers."""

from typing import Any, Literal, cast

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from advanced_rag.generation.chat import ChatModel
from advanced_rag.generation.models import (
    AgentAnswer,
    AnswerClaim,
    CitationSource,
    CitationValidation,
    GraphTraceEvent,
    GroundedDraft,
    ModelUsage,
)
from advanced_rag.graph.state import AgentState
from advanced_rag.retrieval.hybrid_models import (
    HybridSearchResponse,
    RetrievalDiagnostics,
)
from advanced_rag.retrieval.models import RetrievalFilters


class AgenticRAG:
    """Execute a controlled local graph around the hybrid retrieval tool."""

    def __init__(
        self,
        *,
        search_tool: BaseTool,
        chat_model: ChatModel,
        max_retrieval_attempts: int = 2,
        top_k: int = 6,
    ) -> None:
        if not 1 <= max_retrieval_attempts <= 5:
            raise ValueError("max_retrieval_attempts must be between 1 and 5")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        self.search_tool = search_tool
        self.chat_model = chat_model
        self.max_retrieval_attempts = max_retrieval_attempts
        self.top_k = top_k
        self.graph = self._build_graph()

    def ask(
        self,
        question: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> AgentAnswer:
        """Run one bounded graph invocation and return its validated answer."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        result_limit = top_k or self.top_k
        if not 1 <= result_limit <= 20:
            raise ValueError("top_k must be between 1 and 20")
        initial: AgentState = {
            "question": question.strip(),
            "filters": filters or RetrievalFilters(),
            "top_k": result_limit,
            "max_retrieval_attempts": self.max_retrieval_attempts,
            "retrieval_attempts": 0,
            "trace": [],
            "errors": [],
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        state = cast(
            AgentState,
            self.graph.invoke(initial, {"recursion_limit": 4 * self.max_retrieval_attempts + 8}),
        )
        answer = state.get("final_answer")
        if answer is None:
            raise RuntimeError("Agent graph completed without a final answer")
        return answer

    def _build_graph(self) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
        builder = StateGraph(AgentState)
        builder.add_node("analyze_question", self._analyze_question)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("grade_evidence", self._grade_evidence)
        builder.add_node("rewrite_query", self._rewrite_query)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("validate_answer", self._validate_answer)
        builder.add_edge(START, "analyze_question")
        builder.add_edge("analyze_question", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "grade_evidence")
        builder.add_conditional_edges(
            "grade_evidence",
            self._route_after_grading,
            {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
        )
        builder.add_edge("rewrite_query", "retrieve_evidence")
        builder.add_edge("generate_answer", "validate_answer")
        builder.add_edge("validate_answer", END)
        return builder.compile()

    def _analyze_question(self, state: AgentState) -> AgentState:
        query = " ".join(state["question"].split())
        return {
            "current_query": query,
            "trace": [self._event("analyze_question", "prepared_query", query, state)],
        }

    def _retrieve_evidence(self, state: AgentState) -> AgentState:
        attempt = state.get("retrieval_attempts", 0) + 1
        filters = state["filters"]
        arguments: dict[str, Any] = {
            "query": state["current_query"],
            "top_k": state["top_k"],
            **filters.model_dump(mode="json", exclude_none=True),
        }
        try:
            raw = self.search_tool.invoke(arguments)
            evidence = HybridSearchResponse.model_validate(raw)
            detail = (
                f"Retrieved {len(evidence.results)} chunks from "
                f"{evidence.diagnostics.distinct_sources} sources"
            )
            errors: list[str] = []
        except Exception as exc:
            message = f"Knowledge-base search failed: {exc}"
            evidence = _empty_evidence(state["current_query"], message)
            detail = message
            errors = [message]
        return {
            "evidence": evidence,
            "retrieval_attempts": attempt,
            "errors": errors,
            "trace": [
                GraphTraceEvent(
                    node="retrieve_evidence",
                    action="searched_local_indexes",
                    detail=detail,
                    retrieval_attempt=attempt,
                )
            ],
        }

    def _grade_evidence(self, state: AgentState) -> AgentState:
        evidence = state["evidence"]
        if not evidence.results:
            sufficient = False
            reason = "No relevant local evidence was retrieved"
        elif evidence.diagnostics.confidence == "low":
            sufficient = False
            reason = "Retrieval confidence was low"
        else:
            sufficient = True
            reason = (
                f"Evidence passed with {evidence.diagnostics.agreement_count} "
                "dense/sparse agreements"
            )
        return {
            "evidence_sufficient": sufficient,
            "grade_reason": reason,
            "trace": [self._event("grade_evidence", "graded", reason, state)],
        }

    def _route_after_grading(
        self, state: AgentState
    ) -> Literal["rewrite_query", "generate_answer"]:
        if state["evidence_sufficient"]:
            return "generate_answer"
        if state["retrieval_attempts"] < state["max_retrieval_attempts"]:
            return "rewrite_query"
        return "generate_answer"

    def _rewrite_query(self, state: AgentState) -> AgentState:
        try:
            query, usage = self.chat_model.rewrite_query(
                question=state["question"],
                previous_query=state["current_query"],
                evidence=state["evidence"].results,
            )
            errors: list[str] = []
        except Exception as exc:
            query = _fallback_rewrite(state["question"])
            usage = ModelUsage()
            errors = [f"Query rewrite failed; used local fallback: {exc}"]
        return {
            "current_query": query,
            "errors": errors,
            **_usage_update(usage),
            "trace": [self._event("rewrite_query", "rewrote_query", query, state)],
        }

    def _generate_answer(self, state: AgentState) -> AgentState:
        evidence = state["evidence"].results
        if not evidence:
            draft = GroundedDraft(
                insufficient_evidence=True,
                limitation="The local document index did not contain relevant evidence.",
            )
            usage = ModelUsage()
            errors: list[str] = []
            detail = "Skipped model generation because no evidence was available"
        else:
            try:
                draft, usage = self.chat_model.generate_answer(
                    question=state["question"], evidence=evidence
                )
                errors = []
                detail = f"Generated {len(draft.claims)} structured claims"
            except Exception as exc:
                draft = GroundedDraft(
                    insufficient_evidence=True,
                    limitation="Answer generation failed; inspect the retrieved evidence directly.",
                )
                usage = ModelUsage()
                errors = [f"Answer generation failed: {exc}"]
                detail = errors[0]
        return {
            "draft": draft,
            "errors": errors,
            **_usage_update(usage),
            "trace": [self._event("generate_answer", "generated_draft", detail, state)],
        }

    def _validate_answer(self, state: AgentState) -> AgentState:
        event = self._event(
            "validate_answer",
            "validated_citations",
            "Checked every claim against retrieved chunk labels",
            state,
        )
        answer = _render_answer(
            question=state["question"],
            query=state["current_query"],
            draft=state["draft"],
            evidence=state["evidence"],
            trace=(*state.get("trace", []), event),
            errors=tuple(state.get("errors", [])),
            usage=ModelUsage(
                api_calls=state.get("api_calls", 0),
                input_tokens=state.get("input_tokens", 0),
                output_tokens=state.get("output_tokens", 0),
            ),
            retrieval_attempts=state["retrieval_attempts"],
        )
        return {"final_answer": answer, "trace": [event]}

    @staticmethod
    def _event(node: str, action: str, detail: str, state: AgentState) -> GraphTraceEvent:
        return GraphTraceEvent(
            node=node,
            action=action,
            detail=detail,
            retrieval_attempt=state.get("retrieval_attempts", 0),
        )


def _render_answer(
    *,
    question: str,
    query: str,
    draft: GroundedDraft,
    evidence: HybridSearchResponse,
    trace: tuple[GraphTraceEvent, ...],
    errors: tuple[str, ...],
    usage: ModelUsage,
    retrieval_attempts: int,
) -> AgentAnswer:
    label_map = {f"S{position}": item for position, item in enumerate(evidence.results, start=1)}
    accepted: list[tuple[AnswerClaim, tuple[str, ...]]] = []
    issues: list[str] = []
    for position, claim in enumerate(draft.claims, start=1):
        labels = tuple(dict.fromkeys(_normalize_label(value) for value in claim.citations))
        unknown = [label for label in labels if label not in label_map]
        if not labels:
            issues.append(f"Claim {position} had no citation")
        elif unknown:
            issues.append(f"Claim {position} referenced unknown citations: {', '.join(unknown)}")
        else:
            accepted.append((claim, labels))

    cited_labels = tuple(dict.fromkeys(label for _claim, labels in accepted for label in labels))
    sources = tuple(
        CitationSource(
            label=label,
            chunk_id=label_map[label].chunk.chunk_id,
            filename=label_map[label].chunk.metadata.filename,
            title=label_map[label].chunk.metadata.title,
            page_number=label_map[label].chunk.metadata.page_number,
        )
        for label in sorted(cited_labels, key=lambda value: int(value[1:]))
    )
    insufficient = draft.insufficient_evidence or not accepted
    if accepted:
        lines = [
            f"- {claim.text.strip()} " + " ".join(f"[{label}]" for label in labels)
            for claim, labels in accepted
        ]
        if draft.limitation:
            lines.append(f"\nLimitation: {draft.limitation.strip()}")
        answer_text = "\n".join(lines)
    else:
        answer_text = (
            "The indexed documents do not provide enough verified evidence to answer this question."
        )
        if draft.limitation:
            answer_text += f" {draft.limitation.strip()}"

    validation = CitationValidation(
        valid=not issues and (bool(accepted) or draft.insufficient_evidence),
        accepted_claims=len(accepted),
        rejected_claims=len(draft.claims) - len(accepted),
        issues=tuple(issues),
    )
    return AgentAnswer(
        question=question,
        answer=answer_text,
        sources=sources,
        insufficient_evidence=insufficient,
        citation_validation=validation,
        retrieval_diagnostics=evidence.diagnostics,
        graph_trace=trace,
        usage=usage,
        retrieval_attempts=retrieval_attempts,
        final_query=query,
        errors=errors,
    )


def _empty_evidence(query: str, warning: str) -> HybridSearchResponse:
    return HybridSearchResponse(
        query=query,
        results=(),
        diagnostics=RetrievalDiagnostics(
            dense_candidates=0,
            sparse_candidates=0,
            fused_candidates=0,
            reranked_candidates=0,
            agreement_count=0,
            distinct_sources=0,
            evidence_tokens=0,
            confidence="low",
            reranker="unavailable",
            rerank_applied=False,
            warnings=(warning,),
        ),
    )


def _fallback_rewrite(question: str) -> str:
    return f"{question.strip()} key mechanisms evidence findings"


def _normalize_label(value: str) -> str:
    return value.strip().upper().removeprefix("[").removesuffix("]")


def _usage_update(usage: ModelUsage) -> AgentState:
    return {
        "api_calls": usage.api_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }

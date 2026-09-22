"""Bounded LangGraph workflow for grounded, citation-validated RAG answers."""

import re
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
        max_agent_steps: int = 8,
        top_k: int = 6,
        scope_description: str | None = None,
        scope_terms: tuple[str, ...] = (),
    ) -> None:
        if not 1 <= max_retrieval_attempts <= 5:
            raise ValueError("max_retrieval_attempts must be between 1 and 5")
        if not 5 <= max_agent_steps <= 100:
            raise ValueError("max_agent_steps must be between 5 and 100")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        self.search_tool = search_tool
        self.chat_model = chat_model
        self.max_retrieval_attempts = max_retrieval_attempts
        self.max_agent_steps = max_agent_steps
        self.top_k = top_k
        self.scope_description = scope_description
        self.scope_terms = tuple(term for term in scope_terms if term.strip())
        self.graph = self._build_graph()

    def ask(
        self,
        question: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        max_retrieval_attempts: int | None = None,
        max_agent_steps: int | None = None,
    ) -> AgentAnswer:
        """Run one bounded graph invocation and return its validated answer."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        result_limit = top_k or self.top_k
        if not 1 <= result_limit <= 20:
            raise ValueError("top_k must be between 1 and 20")
        requested_attempts = (
            self.max_retrieval_attempts
            if max_retrieval_attempts is None
            else max_retrieval_attempts
        )
        if not 1 <= requested_attempts <= 5:
            raise ValueError("max_retrieval_attempts must be between 1 and 5")
        step_limit = self.max_agent_steps if max_agent_steps is None else max_agent_steps
        if not 5 <= step_limit <= 100:
            raise ValueError("max_agent_steps must be between 5 and 100")
        attempt_limit = bounded_retrieval_attempts(requested_attempts, step_limit)
        initial: AgentState = {
            "question": question.strip(),
            "filters": filters or RetrievalFilters(),
            "top_k": result_limit,
            "max_retrieval_attempts": attempt_limit,
            "retrieval_attempts": 0,
            "trace": [],
            "errors": [],
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        state = cast(
            AgentState,
            self.graph.invoke(initial, {"recursion_limit": step_limit + 3}),
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
        builder.add_conditional_edges(
            "analyze_question",
            self._route_after_analysis,
            {"retrieve_evidence": "retrieve_evidence", "validate_answer": "validate_answer"},
        )
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
        in_scope = question_matches_scope(query, self.scope_terms)
        if not in_scope:
            description = self.scope_description or "the configured document domain"
            reason = f"Question is outside the configured scope: {description}."
            return {
                "current_query": query,
                "question_in_scope": False,
                "evidence": _empty_evidence(query, reason),
                "draft": GroundedDraft(insufficient_evidence=True, limitation=reason),
                "trace": [self._event("analyze_question", "rejected_out_of_scope", reason, state)],
            }
        return {
            "current_query": query,
            "question_in_scope": True,
            "trace": [self._event("analyze_question", "prepared_query", query, state)],
        }

    @staticmethod
    def _route_after_analysis(state: AgentState) -> Literal["retrieve_evidence", "validate_answer"]:
        return "retrieve_evidence" if state["question_in_scope"] else "validate_answer"

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
            text=label_map[label].chunk.text,
            token_count=label_map[label].chunk.token_count,
            final_rank=label_map[label].final_rank,
            dense_rank=label_map[label].dense_rank,
            dense_score=label_map[label].dense_score,
            sparse_rank=label_map[label].sparse_rank,
            sparse_score=label_map[label].sparse_score,
            rrf_score=label_map[label].rrf_score,
            reranker_score=label_map[label].reranker_score,
            retrieval_sources=label_map[label].retrieval_sources,
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


def bounded_retrieval_attempts(requested_attempts: int, max_agent_steps: int) -> int:
    """Fit retrieval attempts into a graph budget of five base and three retry steps."""
    if not 1 <= requested_attempts <= 5:
        raise ValueError("requested_attempts must be between 1 and 5")
    if not 5 <= max_agent_steps <= 100:
        raise ValueError("max_agent_steps must be between 5 and 100")
    attempts_allowed_by_steps = max(1, (max_agent_steps - 2) // 3)
    return min(requested_attempts, attempts_allowed_by_steps)


def question_matches_scope(question: str, scope_terms: tuple[str, ...]) -> bool:
    """Return whether a question contains a configured domain phrase.

    An empty term list disables the optional domain gate for reusable library callers.
    """
    if not scope_terms:
        return True
    question_tokens = re.findall(r"\w+", question.casefold())
    for term in scope_terms:
        term_tokens = re.findall(r"\w+", term.casefold())
        if term_tokens and _contains_scope_phrase(question_tokens, term_tokens):
            return True
    return False


def _contains_scope_phrase(question_tokens: list[str], term_tokens: list[str]) -> bool:
    """Match one contiguous phrase while tolerating a simple English plural suffix."""
    width = len(term_tokens)
    return any(
        all(
            _scope_words_match(question_word, term_word)
            for question_word, term_word in zip(
                question_tokens[start : start + width], term_tokens, strict=True
            )
        )
        for start in range(len(question_tokens) - width + 1)
    )


def _scope_words_match(question_word: str, term_word: str) -> bool:
    if question_word == term_word:
        return True
    if len(term_word) < 3:
        return False
    return question_word in {f"{term_word}s", f"{term_word}es"}

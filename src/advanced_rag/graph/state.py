"""Typed state passed between the local LangGraph nodes."""

import operator
from typing import Annotated, TypedDict

from advanced_rag.generation.models import AgentAnswer, GraphTraceEvent, GroundedDraft
from advanced_rag.retrieval.hybrid_models import HybridSearchResponse
from advanced_rag.retrieval.models import RetrievalFilters


class AgentState(TypedDict, total=False):
    """Complete state for one bounded question-answering run."""

    question: str
    current_query: str
    question_in_scope: bool
    filters: RetrievalFilters
    top_k: int
    max_retrieval_attempts: int
    semantic_evidence_grading: bool
    retrieval_attempts: int
    evidence: HybridSearchResponse
    evidence_sufficient: bool
    grade_reason: str
    draft: GroundedDraft
    final_answer: AgentAnswer
    trace: Annotated[list[GraphTraceEvent], operator.add]
    errors: Annotated[list[str], operator.add]
    api_calls: Annotated[int, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]

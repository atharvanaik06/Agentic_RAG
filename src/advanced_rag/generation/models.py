"""Typed inputs and outputs for grounded answer generation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from advanced_rag.retrieval.hybrid_models import RetrievalDiagnostics


class ModelUsage(BaseModel):
    """Token and call counts reported by a hosted chat provider."""

    model_config = ConfigDict(frozen=True)

    api_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class QueryRewrite(BaseModel):
    """A focused replacement query for a second local retrieval attempt."""

    query: str = Field(min_length=1, max_length=500)


class EvidenceGrade(BaseModel):
    """Structured judgment of whether retrieved passages can answer the question."""

    sufficient: bool
    supporting_labels: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=800)
    missing_information: str | None = Field(default=None, max_length=800)


class AnswerClaim(BaseModel):
    """One independently cited factual statement."""

    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class GroundedDraft(BaseModel):
    """Structured model output rendered and validated locally."""

    claims: list[AnswerClaim] = Field(default_factory=list)
    insufficient_evidence: bool = False
    limitation: str | None = None


class CitationSource(BaseModel):
    """Human-readable source corresponding to one local evidence label."""

    model_config = ConfigDict(frozen=True)

    label: str
    chunk_id: str
    filename: str
    title: str | None
    page_number: int | None
    text: str = ""
    token_count: int = Field(default=0, ge=0)
    final_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    sparse_rank: int | None = Field(default=None, ge=1)
    sparse_score: float | None = Field(default=None, ge=0)
    rrf_score: float | None = Field(default=None, ge=0)
    reranker_score: float | None = None
    retrieval_sources: tuple[Literal["dense", "sparse"], ...] = ()


class CitationValidation(BaseModel):
    """Deterministic validation result for a generated draft."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    accepted_claims: int = Field(ge=0)
    rejected_claims: int = Field(ge=0)
    issues: tuple[str, ...] = ()


class GraphTraceEvent(BaseModel):
    """One concise, secret-free graph execution event."""

    model_config = ConfigDict(frozen=True)

    node: str
    action: str
    detail: str
    retrieval_attempt: int = Field(default=0, ge=0)


class AgentAnswer(BaseModel):
    """Final local graph result returned by the CLI or a future UI."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    sources: tuple[CitationSource, ...]
    insufficient_evidence: bool
    citation_validation: CitationValidation
    retrieval_diagnostics: RetrievalDiagnostics
    graph_trace: tuple[GraphTraceEvent, ...]
    usage: ModelUsage
    retrieval_attempts: int = Field(ge=0)
    final_query: str
    errors: tuple[str, ...] = ()

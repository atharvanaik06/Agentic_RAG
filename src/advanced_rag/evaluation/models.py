"""Schemas for portable benchmarks, evaluation results, and regression gates."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvaluationCategory = Literal[
    "exact_fact",
    "numerical",
    "single_document",
    "cross_document",
    "multi_step",
    "rewrite",
    "unanswerable",
]
RetrievalMethod = Literal["dense", "sparse", "hybrid_rrf", "hybrid_reranked"]
RankChange = Literal["improved", "unchanged", "worsened", "missing"]
EntailmentLabel = Literal["supported", "partially_supported", "unsupported", "contradicted"]


class GoldTarget(BaseModel):
    """One acceptable source, optionally narrowed to pages or chunks."""

    model_config = ConfigDict(frozen=True)

    filename: str
    pages: tuple[int, ...] = ()
    chunk_ids: tuple[str, ...] = ()


class BenchmarkCase(BaseModel):
    """One human-curated evaluation item."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    question: str = Field(min_length=3)
    category: EvaluationCategory
    answerable: bool = True
    gold_targets: tuple[GoldTarget, ...] = ()
    reference_answer: str | None = None
    must_include: tuple[str, ...] = ()
    notes: str | None = None

    @model_validator(mode="after")
    def validate_gold_targets(self) -> "BenchmarkCase":
        if self.answerable and not self.gold_targets:
            raise ValueError("Answerable cases require at least one gold target")
        if not self.answerable and self.gold_targets:
            raise ValueError("Unanswerable cases must not define gold targets")
        return self


class RankedRecord(BaseModel):
    """Minimal auditable retrieval record saved in reports."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    chunk_id: str
    filename: str
    page_number: int | None
    relevant: bool
    score: float | None = None


class RetrievalCaseResult(BaseModel):
    """Metrics for one method on one benchmark case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    category: EvaluationCategory
    method: RetrievalMethod
    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    average_precision: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    source_hit: bool
    page_hit: bool | None
    latency_ms: float = Field(ge=0)
    records: tuple[RankedRecord, ...]
    error: str | None = None


class MethodSummary(BaseModel):
    """Macro-averaged retrieval metrics for one method."""

    model_config = ConfigDict(frozen=True)

    method: RetrievalMethod
    cases: int = Field(ge=0)
    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    map: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    source_hit_rate: float = Field(ge=0, le=1)
    page_hit_rate: float | None = Field(default=None, ge=0, le=1)
    average_latency_ms: float = Field(ge=0)


class RerankerSummary(BaseModel):
    """Best-gold-rank movement caused by final reranking and selection."""

    model_config = ConfigDict(frozen=True)

    improved: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    worsened: int = Field(ge=0)
    missing: int = Field(ge=0)
    average_rank_change: float


class RetrievalEvaluationReport(BaseModel):
    """Complete retrieval benchmark output."""

    model_config = ConfigDict(frozen=True)

    benchmark: str
    top_k: int = Field(gt=0)
    cases: int = Field(ge=0)
    results: tuple[RetrievalCaseResult, ...]
    summaries: tuple[MethodSummary, ...]
    reranker: RerankerSummary


class EntailmentResult(BaseModel):
    """Optional model-judge verdict for one answer claim."""

    model_config = ConfigDict(frozen=True)

    claim: str
    citations: tuple[str, ...]
    label: EntailmentLabel
    explanation: str


class AgentCaseResult(BaseModel):
    """Deterministic and optional judged metrics for one graph run."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    category: EvaluationCategory
    answerable: bool
    refused: bool
    refusal_correct: bool
    citation_valid: bool
    expected_source_hit: bool
    concept_coverage: float = Field(ge=0, le=1)
    retrieval_attempts: int = Field(ge=0)
    rewrite_used: bool
    chat_api_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    entailment: tuple[EntailmentResult, ...] = ()
    answer: str
    errors: tuple[str, ...] = ()


class AgentSummary(BaseModel):
    """Macro metrics for the full LangGraph workflow."""

    model_config = ConfigDict(frozen=True)

    cases: int = Field(ge=0)
    citation_validity_rate: float = Field(ge=0, le=1)
    expected_source_hit_rate: float = Field(ge=0, le=1)
    concept_coverage: float = Field(ge=0, le=1)
    refusal_accuracy: float | None = Field(default=None, ge=0, le=1)
    rewrite_rate: float = Field(ge=0, le=1)
    average_retrieval_attempts: float = Field(ge=0)
    average_chat_api_calls: float = Field(ge=0)
    average_total_tokens: float = Field(ge=0)
    average_latency_ms: float = Field(ge=0)
    entailment_support_rate: float | None = Field(default=None, ge=0, le=1)


class AgentEvaluationReport(BaseModel):
    """Complete agent benchmark output."""

    model_config = ConfigDict(frozen=True)

    benchmark: str
    cases: int = Field(ge=0)
    judged_entailment: bool
    results: tuple[AgentCaseResult, ...]
    summary: AgentSummary


class RegressionThresholds(BaseModel):
    """Configurable minimum quality gates for repeatable releases."""

    model_config = ConfigDict(frozen=True)

    hybrid_recall_at_k: float = Field(default=0.75, ge=0, le=1)
    citation_validity_rate: float = Field(default=1.0, ge=0, le=1)
    refusal_accuracy: float = Field(default=0.8, ge=0, le=1)
    maximum_average_retrieval_attempts: float = Field(default=1.5, ge=1)


class RegressionResult(BaseModel):
    """Pass/fail details for configured quality gates."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    checks: dict[str, bool]

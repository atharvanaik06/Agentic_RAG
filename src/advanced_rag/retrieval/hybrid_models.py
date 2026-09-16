"""Typed records for fusion, reranking, and agent-facing retrieval."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from advanced_rag.ingestion.models import DocumentChunk

RetrievalSource = Literal["dense", "sparse"]
ConfidenceLevel = Literal["low", "medium", "high"]


class FusedCandidate(BaseModel):
    """A unique chunk with provenance from every retriever that found it."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    rrf_score: float = Field(ge=0)
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    sparse_rank: int | None = Field(default=None, ge=1)
    sparse_score: float | None = Field(default=None, ge=0)
    retrieval_sources: tuple[RetrievalSource, ...]


class RerankScore(BaseModel):
    """Reranker score tied to a stable canonical chunk ID."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    score: float


class HybridSearchResult(FusedCandidate):
    """A selected evidence chunk in final relevance order."""

    final_rank: int = Field(ge=1)
    reranker_score: float | None = None


class RetrievalDiagnostics(BaseModel):
    """Signals used by later LangGraph routing and debugging."""

    model_config = ConfigDict(frozen=True)

    dense_candidates: int = Field(ge=0)
    sparse_candidates: int = Field(ge=0)
    fused_candidates: int = Field(ge=0)
    reranked_candidates: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    distinct_sources: int = Field(ge=0)
    evidence_tokens: int = Field(ge=0)
    confidence: ConfidenceLevel
    reranker: str
    rerank_applied: bool
    warnings: tuple[str, ...] = ()


class HybridSearchResponse(BaseModel):
    """Structured retrieval response suitable for an agent tool result."""

    model_config = ConfigDict(frozen=True)

    query: str
    results: tuple[HybridSearchResult, ...]
    diagnostics: RetrievalDiagnostics

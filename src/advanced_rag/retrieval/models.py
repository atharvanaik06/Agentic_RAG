"""Typed models shared by dense and sparse retrieval."""

from typing import cast

from chromadb.api.types import Where
from pydantic import BaseModel, ConfigDict, Field, model_validator

from advanced_rag.ingestion.models import DocumentChunk, FileType


class RetrievalFilters(BaseModel):
    """Supported exact-match filters across retrieval backends."""

    model_config = ConfigDict(frozen=True)

    source_id: str | None = None
    filename: str | None = None
    file_type: FileType | None = None
    page_number: int | None = Field(default=None, ge=1)

    def matches(self, chunk: DocumentChunk) -> bool:
        """Return whether a canonical chunk satisfies every populated filter."""
        metadata = chunk.metadata
        return all(
            value is None or getattr(metadata, field) == value
            for field, value in self.model_dump(exclude_none=True).items()
        )


class DenseSearchFilters(RetrievalFilters):
    """Exact-match filters with Chroma expression conversion."""

    def to_chroma(self) -> Where | None:
        """Convert populated fields into a Chroma metadata expression."""
        conditions: list[dict[str, object]] = []
        for key, value in self.model_dump(exclude_none=True, mode="json").items():
            conditions.append({key: {"$eq": value}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return cast(Where, conditions[0])
        return cast(Where, {"$and": conditions})


class DenseSearchResult(BaseModel):
    """One ranked dense-search match."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    rank: int = Field(ge=1)
    distance: float = Field(ge=0)
    score: float


class SparseSearchResult(BaseModel):
    """One ranked BM25 keyword-search match."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    rank: int = Field(ge=1)
    score: float = Field(ge=0)


class SparseIndexingResult(BaseModel):
    """Summary of one deterministic sparse-index synchronization."""

    model_config = ConfigDict(frozen=True)

    discovered_files: int = Field(ge=0)
    processed_files: int = Field(ge=0)
    failed_files: int = Field(ge=0)
    rebuilt: bool
    indexed_chunks: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    fingerprint: str


class SparseIndexInfo(BaseModel):
    """Inspectable state and compatibility metadata for a BM25 index."""

    model_config = ConfigDict(frozen=True)

    path: str
    count: int = Field(ge=0)
    method: str
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)
    fingerprint: str | None
    schema_version: int = Field(ge=1)


class IndexingResult(BaseModel):
    """Summary of one idempotent synchronization operation."""

    model_config = ConfigDict(frozen=True)

    discovered_files: int = Field(ge=0)
    processed_files: int = Field(ge=0)
    failed_files: int = Field(ge=0)
    embedded_chunks: int = Field(ge=0)
    upserted_chunks: int = Field(ge=0)
    unchanged_chunks: int = Field(ge=0)
    deleted_chunks: int = Field(ge=0)
    total_chunks: int = Field(ge=0)


class CollectionInfo(BaseModel):
    """Inspectable collection state and embedding compatibility metadata."""

    model_config = ConfigDict(frozen=True)

    name: str
    count: int = Field(ge=0)
    path: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int = Field(gt=0)


class StoredMetadata(BaseModel):
    """Flat metadata representation accepted by Chroma."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_path: str
    filename: str
    file_type: FileType
    content_hash: str
    title: str
    page_number: int
    heading_path: str
    chunk_index: int = Field(ge=0)
    token_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_sentinels(self) -> "StoredMetadata":
        if self.page_number != -1 and self.page_number < 1:
            raise ValueError("page_number must be -1 or a positive integer")
        return self

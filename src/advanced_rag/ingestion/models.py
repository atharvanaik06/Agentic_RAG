"""Typed records shared by ingestion and downstream indexes."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class FileType(StrEnum):
    """Source formats supported by the ingestion pipeline."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class IssueCode(StrEnum):
    """Stable error and skip categories for reporting."""

    UNSUPPORTED = "unsupported_file"
    EMPTY = "empty_document"
    DUPLICATE = "duplicate_document"
    ENCRYPTED = "encrypted_pdf"
    IMAGE_ONLY = "image_only_pdf"
    EXTRACTION_FAILED = "extraction_failed"


class TextSegment(BaseModel):
    """A citation-safe unit emitted by a format-specific loader."""

    model_config = ConfigDict(frozen=True)

    text: str
    page_number: int | None = Field(default=None, ge=1)
    heading_path: tuple[str, ...] = ()


class LoadedDocument(BaseModel):
    """Extracted document before normalization and chunking."""

    model_config = ConfigDict(frozen=True)

    path: Path
    file_type: FileType
    title: str | None = None
    segments: tuple[TextSegment, ...]


class ChunkMetadata(BaseModel):
    """Provenance required for filtering, updates, and citations."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_path: str
    filename: str
    file_type: FileType
    content_hash: str
    title: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    heading_path: tuple[str, ...] = ()
    chunk_index: int = Field(ge=0)


class DocumentChunk(BaseModel):
    """Canonical record consumed by dense and sparse indexes."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    metadata: ChunkMetadata


class IngestionIssue(BaseModel):
    """A non-fatal file-level ingestion problem."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    code: IssueCode
    message: str


class IngestionResult(BaseModel):
    """Chunks and an auditable summary for one ingestion run."""

    model_config = ConfigDict(frozen=True)

    discovered_files: int = Field(ge=0)
    processed_files: int = Field(ge=0)
    skipped_files: int = Field(ge=0)
    failed_files: int = Field(ge=0)
    chunks: tuple[DocumentChunk, ...] = ()
    issues: tuple[IngestionIssue, ...] = ()

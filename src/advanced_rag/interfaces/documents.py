"""Safe local-corpus management shared by the Streamlit interface and tests."""

import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from advanced_rag.config import Settings
from advanced_rag.ingestion.discovery import discover_files, relative_source_path
from advanced_rag.ingestion.models import DocumentChunk, IngestionIssue, IngestionResult
from advanced_rag.ingestion.pipeline import IngestionPipeline
from advanced_rag.retrieval.dense import ChromaDenseIndex
from advanced_rag.retrieval.models import IndexingResult, SparseIndexingResult
from advanced_rag.retrieval.sparse import BM25SparseIndex

SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".md", ".markdown", ".txt"})


@dataclass(frozen=True)
class UploadPayload:
    """A browser upload detached from Streamlit-specific types."""

    name: str
    content: bytes


@dataclass(frozen=True)
class UploadResult:
    """Outcome of an atomic upload batch."""

    saved: tuple[str, ...]
    replaced: tuple[str, ...]
    skipped: tuple[str, ...]
    rejected: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.saved or self.replaced)


@dataclass(frozen=True)
class CorpusFile:
    """One visible file currently present in the local corpus directory."""

    relative_path: str
    extension: str
    size_bytes: int
    supported: bool


@dataclass(frozen=True)
class DocumentInspection:
    """Per-document extraction and chunking status."""

    relative_path: str
    size_bytes: int
    status: str
    chunks: int
    pages: int
    tokens: int
    detail: str


@dataclass(frozen=True)
class CorpusInspection:
    """Auditable, API-free inspection of the complete local corpus."""

    ingestion: IngestionResult
    documents: tuple[DocumentInspection, ...]

    @property
    def chunks(self) -> int:
        return len(self.ingestion.chunks)

    @property
    def pages(self) -> int:
        return len(
            {
                (chunk.metadata.source_id, chunk.metadata.page_number)
                for chunk in self.ingestion.chunks
                if chunk.metadata.page_number is not None
            }
        )

    @property
    def tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.ingestion.chunks)


@dataclass(frozen=True)
class CorpusBuildResult:
    """Combined result from one canonical ingestion and both index updates."""

    inspection: CorpusInspection
    dense: IndexingResult
    sparse: SparseIndexingResult


class CorpusBuildError(RuntimeError):
    """Raised before index mutation when the corpus is unsafe to synchronize."""

    def __init__(self, message: str, inspection: CorpusInspection) -> None:
        super().__init__(message)
        self.inspection = inspection


def list_corpus_files(data_dir: Path) -> tuple[CorpusFile, ...]:
    """List corpus files without reading their contents."""
    data_dir.mkdir(parents=True, exist_ok=True)
    return tuple(
        CorpusFile(
            relative_path=relative_source_path(path, data_dir),
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size,
            supported=path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS,
        )
        for path in discover_files(data_dir)
    )


def save_uploads(
    data_dir: Path,
    uploads: Sequence[UploadPayload],
    *,
    overwrite: bool,
    maximum_bytes: int,
) -> UploadResult:
    """Validate and atomically save browser uploads to the corpus root."""
    data_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    replaced: list[str] = []
    skipped: list[str] = []
    rejected: list[str] = []

    for upload in uploads:
        reason = _upload_rejection(upload, maximum_bytes=maximum_bytes)
        if reason:
            rejected.append(f"{upload.name}: {reason}")
            continue

        target = data_dir / upload.name
        existed = target.exists()
        if existed and not overwrite:
            skipped.append(upload.name)
            continue

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}-",
            suffix=".upload",
            dir=data_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(upload.content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

        (replaced if existed else saved).append(upload.name)

    return UploadResult(
        saved=tuple(saved),
        replaced=tuple(replaced),
        skipped=tuple(skipped),
        rejected=tuple(rejected),
    )


def delete_corpus_files(data_dir: Path, relative_paths: Iterable[str]) -> tuple[str, ...]:
    """Delete explicitly selected files while preventing traversal outside the corpus."""
    root = data_dir.resolve()
    deleted: list[str] = []
    for relative_path in relative_paths:
        candidate = (root / relative_path).resolve()
        if candidate.parent != root and root not in candidate.parents:
            raise ValueError(f"Document path escapes the corpus directory: {relative_path}")
        if candidate.is_file():
            candidate.unlink()
            deleted.append(candidate.relative_to(root).as_posix())
    return tuple(deleted)


def inspect_corpus(settings: Settings) -> CorpusInspection:
    """Extract and chunk every document without calling an embedding or chat API."""
    files = list_corpus_files(settings.data_dir)
    ingestion = IngestionPipeline(settings).ingest(settings.data_dir)
    chunks_by_source: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in ingestion.chunks:
        chunks_by_source[chunk.metadata.source_path].append(chunk)
    issues_by_source: dict[str, list[IngestionIssue]] = defaultdict(list)
    for issue in ingestion.issues:
        issues_by_source[issue.source_path].append(issue)

    documents: list[DocumentInspection] = []
    for corpus_file in files:
        chunks = chunks_by_source[corpus_file.relative_path]
        issues = issues_by_source[corpus_file.relative_path]
        if chunks:
            status = "ready"
        elif issues:
            status = issues[0].code.value
        else:
            status = "skipped"
        page_numbers = {
            chunk.metadata.page_number for chunk in chunks if chunk.metadata.page_number is not None
        }
        documents.append(
            DocumentInspection(
                relative_path=corpus_file.relative_path,
                size_bytes=corpus_file.size_bytes,
                status=status,
                chunks=len(chunks),
                pages=len(page_numbers),
                tokens=sum(chunk.token_count for chunk in chunks),
                detail="; ".join(issue.message for issue in issues),
            )
        )
    return CorpusInspection(ingestion=ingestion, documents=tuple(documents))


def synchronize_corpus(
    settings: Settings,
    dense: ChromaDenseIndex,
    sparse: BM25SparseIndex,
    *,
    inspection: CorpusInspection | None = None,
) -> CorpusBuildResult:
    """Synchronize both retrievers from one validated canonical ingestion result."""
    inspected = inspection or inspect_corpus(settings)
    if inspected.ingestion.failed_files:
        raise CorpusBuildError(
            "Indexing stopped because one or more supported documents failed extraction.",
            inspected,
        )
    if not inspected.ingestion.chunks:
        raise CorpusBuildError(
            "Indexing stopped because the corpus produced no usable chunks.",
            inspected,
        )

    active_source_ids = {chunk.metadata.source_id for chunk in inspected.ingestion.chunks}
    dense_result = dense.sync(
        inspected.ingestion,
        active_source_ids=active_source_ids,
        prune_missing=True,
    )
    sparse_result = sparse.sync(inspected.ingestion)
    return CorpusBuildResult(
        inspection=inspected,
        dense=dense_result,
        sparse=sparse_result,
    )


def _upload_rejection(upload: UploadPayload, *, maximum_bytes: int) -> str | None:
    name = upload.name.strip()
    if not name or name != upload.name or name != Path(name).name or "/" in name or "\\" in name:
        return "filename must not contain a directory path"
    if Path(name).suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        return "unsupported file type"
    if not upload.content:
        return "file is empty"
    if len(upload.content) > maximum_bytes:
        return f"file exceeds the {maximum_bytes // (1024 * 1024)} MB upload limit"
    return None

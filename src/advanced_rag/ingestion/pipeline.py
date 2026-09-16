"""End-to-end document ingestion orchestration."""

from collections.abc import Mapping
from pathlib import Path

from advanced_rag.config import Settings
from advanced_rag.ingestion.chunking import TokenChunker
from advanced_rag.ingestion.discovery import discover_files, relative_source_path
from advanced_rag.ingestion.errors import DocumentLoadError, EmptyDocumentError
from advanced_rag.ingestion.hashing import hash_file, stable_id
from advanced_rag.ingestion.loaders import MarkdownLoader, PdfLoader, TextLoader
from advanced_rag.ingestion.loaders.base import DocumentLoader
from advanced_rag.ingestion.models import (
    ChunkMetadata,
    DocumentChunk,
    IngestionIssue,
    IngestionResult,
    IssueCode,
    LoadedDocument,
)
from advanced_rag.ingestion.normalization import normalize_text


class IngestionPipeline:
    """Load, normalize, chunk, identify, and report local documents."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        loaders: Mapping[str, DocumentLoader] | None = None,
        chunker: TokenChunker | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.loaders: Mapping[str, DocumentLoader] = loaders or {
            ".pdf": PdfLoader(),
            ".md": MarkdownLoader(),
            ".markdown": MarkdownLoader(),
            ".txt": TextLoader(),
        }
        self.chunker = chunker or TokenChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            min_chunk_size=self.settings.min_chunk_size,
            encoding_name=self.settings.token_encoding,
        )

    def ingest(self, source: Path | str) -> IngestionResult:
        """Ingest a file or directory while isolating file-level failures."""
        source_path = Path(source)
        files = discover_files(source_path)
        chunks: list[DocumentChunk] = []
        issues: list[IngestionIssue] = []
        seen_content: dict[str, str] = {}
        processed = skipped = failed = 0

        for path in files:
            relative_path = relative_source_path(path, source_path)
            loader = self.loaders.get(path.suffix.lower())
            if loader is None:
                skipped += 1
                issues.append(
                    IngestionIssue(
                        source_path=relative_path,
                        code=IssueCode.UNSUPPORTED,
                        message=f"Unsupported extension: {path.suffix or '<none>'}",
                    )
                )
                continue

            try:
                content_hash = hash_file(path)
                if original := seen_content.get(content_hash):
                    skipped += 1
                    issues.append(
                        IngestionIssue(
                            source_path=relative_path,
                            code=IssueCode.DUPLICATE,
                            message=f"Exact duplicate of {original}",
                        )
                    )
                    continue

                document = loader.load(path)
                document_chunks = self._chunk_document(
                    document,
                    source_path=relative_path,
                    content_hash=content_hash,
                )
                if not document_chunks:
                    raise EmptyDocumentError("Document produced no usable chunks")

                seen_content[content_hash] = relative_path
                chunks.extend(document_chunks)
                processed += 1
            except DocumentLoadError as error:
                failed += 1
                issues.append(
                    IngestionIssue(
                        source_path=relative_path,
                        code=error.code,
                        message=str(error),
                    )
                )
            except Exception as error:
                failed += 1
                issues.append(
                    IngestionIssue(
                        source_path=relative_path,
                        code=IssueCode.EXTRACTION_FAILED,
                        message=f"{type(error).__name__}: {error}",
                    )
                )

        return IngestionResult(
            discovered_files=len(files),
            processed_files=processed,
            skipped_files=skipped,
            failed_files=failed,
            chunks=tuple(chunks),
            issues=tuple(issues),
        )

    def _chunk_document(
        self,
        document: LoadedDocument,
        *,
        source_path: str,
        content_hash: str,
    ) -> list[DocumentChunk]:
        source_id = stable_id(source_path)
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for segment in document.segments:
            normalized = normalize_text(
                segment.text,
                repair_pdf_wraps=document.file_type.value == "pdf",
            )
            for text in self.chunker.split(normalized):
                metadata = ChunkMetadata(
                    source_id=source_id,
                    source_path=source_path,
                    filename=document.path.name,
                    file_type=document.file_type,
                    content_hash=content_hash,
                    title=document.title,
                    page_number=segment.page_number,
                    heading_path=segment.heading_path,
                    chunk_index=chunk_index,
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=stable_id(
                            source_id,
                            segment.page_number,
                            "/".join(segment.heading_path),
                            chunk_index,
                            text,
                        ),
                        text=text,
                        token_count=self.chunker.count_tokens(text),
                        metadata=metadata,
                    )
                )
                chunk_index += 1

        return chunks

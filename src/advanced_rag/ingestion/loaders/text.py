"""Plain-text document loader."""

from pathlib import Path

from advanced_rag.ingestion.errors import EmptyDocumentError
from advanced_rag.ingestion.models import FileType, LoadedDocument, TextSegment


class TextLoader:
    """Load UTF-8 plain text with a replacement fallback for bad bytes."""

    def load(self, path: Path) -> LoadedDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise EmptyDocumentError("Text file contains no usable content")
        return LoadedDocument(
            path=path,
            file_type=FileType.TEXT,
            title=path.stem,
            segments=(TextSegment(text=text),),
        )

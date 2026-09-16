"""Page-aware PDF document loader."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from advanced_rag.ingestion.errors import EncryptedPdfError, ImageOnlyPdfError
from advanced_rag.ingestion.models import FileType, LoadedDocument, TextSegment


class PdfLoader:
    """Extract PDF text page by page for dependable citations."""

    def load(self, path: Path) -> LoadedDocument:
        try:
            reader = PdfReader(path)
            if reader.is_encrypted and reader.decrypt("") == 0:
                raise EncryptedPdfError("PDF requires a password")

            segments = tuple(
                TextSegment(text=text, page_number=page_number)
                for page_number, page in enumerate(reader.pages, start=1)
                if (text := (page.extract_text() or "")).strip()
            )
        except EncryptedPdfError:
            raise
        except (PdfReadError, OSError, ValueError) as error:
            raise PdfReadError(f"Could not read PDF: {error}") from error

        if not segments:
            raise ImageOnlyPdfError("PDF contains no extractable text; OCR may be required")

        metadata = reader.metadata
        title = metadata.title if metadata and metadata.title else path.stem
        return LoadedDocument(
            path=path,
            file_type=FileType.PDF,
            title=title,
            segments=segments,
        )

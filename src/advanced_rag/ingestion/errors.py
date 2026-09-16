"""Expected document-loading failures."""

from advanced_rag.ingestion.models import IssueCode


class DocumentLoadError(Exception):
    """Base class for a file that cannot produce usable text."""

    code = IssueCode.EXTRACTION_FAILED


class EmptyDocumentError(DocumentLoadError):
    """The source contains no usable text."""

    code = IssueCode.EMPTY


class EncryptedPdfError(DocumentLoadError):
    """A PDF cannot be opened without a password."""

    code = IssueCode.ENCRYPTED


class ImageOnlyPdfError(DocumentLoadError):
    """A PDF has pages but no extractable text."""

    code = IssueCode.IMAGE_ONLY

"""Loader interface."""

from pathlib import Path
from typing import Protocol

from advanced_rag.ingestion.models import LoadedDocument


class DocumentLoader(Protocol):
    """Interface implemented by every source-format loader."""

    def load(self, path: Path) -> LoadedDocument:
        """Extract citation-safe segments from a file."""
        ...

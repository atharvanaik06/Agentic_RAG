"""Markdown loader with heading-path preservation."""

import re
from pathlib import Path

from advanced_rag.ingestion.errors import EmptyDocumentError
from advanced_rag.ingestion.models import FileType, LoadedDocument, TextSegment

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


class MarkdownLoader:
    """Split Markdown into sections without treating fenced code as headings."""

    def load(self, path: Path) -> LoadedDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise EmptyDocumentError("Markdown file contains no usable content")

        segments: list[TextSegment] = []
        headings: list[str] = []
        buffer: list[str] = []
        fence_marker: str | None = None

        def flush() -> None:
            content = "\n".join(buffer).strip()
            if content:
                segments.append(TextSegment(text=content, heading_path=tuple(headings)))
            buffer.clear()

        for line in text.splitlines():
            fence = _FENCE.match(line)
            if fence:
                marker = fence.group(1)[0]
                fence_marker = None if fence_marker == marker else marker
                buffer.append(line)
                continue

            heading = _HEADING.match(line) if fence_marker is None else None
            if heading:
                flush()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                headings[level - 1 :] = [title]
            else:
                buffer.append(line)

        flush()
        if not segments:
            raise EmptyDocumentError("Markdown file contains headings but no body content")

        title = next(
            (segment.heading_path[0] for segment in segments if segment.heading_path), path.stem
        )
        return LoadedDocument(
            path=path,
            file_type=FileType.MARKDOWN,
            title=title,
            segments=tuple(segments),
        )

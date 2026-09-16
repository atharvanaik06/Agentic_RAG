"""Conservative text cleanup that preserves document structure."""

import re
import unicodedata

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_PDF_HYPHENATED_LINE = re.compile(r"(?<=\w)-\n(?=\w)")


def normalize_text(text: str, *, repair_pdf_wraps: bool = False) -> str:
    """Normalize extracted text without destroying headings or paragraphs."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    if repair_pdf_wraps:
        normalized = _PDF_HYPHENATED_LINE.sub("", normalized)

    lines = [line.rstrip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    return _EXCESS_BLANK_LINES.sub("\n\n", normalized).strip()

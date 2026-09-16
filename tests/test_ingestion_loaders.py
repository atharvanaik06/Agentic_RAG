from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from advanced_rag.ingestion.errors import EmptyDocumentError, EncryptedPdfError, ImageOnlyPdfError
from advanced_rag.ingestion.loaders.markdown import MarkdownLoader
from advanced_rag.ingestion.loaders.pdf import PdfLoader
from advanced_rag.ingestion.loaders.text import TextLoader


def test_text_loader_reads_content(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Technical notes", encoding="utf-8")

    document = TextLoader().load(source)

    assert document.title == "notes"
    assert document.segments[0].text == "Technical notes"


def test_empty_text_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text(" \n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        TextLoader().load(source)


def test_markdown_loader_preserves_heading_hierarchy_and_code(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text(
        "# API Guide\n\nOverview.\n\n## Authentication\n\nUse a token.\n\n"
        "```markdown\n# This is code\n```\n",
        encoding="utf-8",
    )

    document = MarkdownLoader().load(source)

    assert document.title == "API Guide"
    assert document.segments[0].heading_path == ("API Guide",)
    assert document.segments[1].heading_path == ("API Guide", "Authentication")
    assert "# This is code" in document.segments[1].text


def test_markdown_with_only_headings_is_empty(tmp_path: Path) -> None:
    source = tmp_path / "empty.md"
    source.write_text("# Title\n## Section", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        MarkdownLoader().load(source)


class _FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


def test_pdf_loader_preserves_page_numbers_and_title(tmp_path: Path) -> None:
    source = tmp_path / "manual.pdf"
    source.touch()
    reader = SimpleNamespace(
        is_encrypted=False,
        pages=[_FakePage("First page"), _FakePage(None), _FakePage("Third page")],
        metadata=SimpleNamespace(title="Product Manual"),
    )

    with patch("advanced_rag.ingestion.loaders.pdf.PdfReader", return_value=reader):
        document = PdfLoader().load(source)

    assert document.title == "Product Manual"
    assert [segment.page_number for segment in document.segments] == [1, 3]


def test_encrypted_pdf_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "private.pdf"
    source.touch()
    reader = SimpleNamespace(is_encrypted=True, decrypt=lambda _: 0)

    with (
        patch("advanced_rag.ingestion.loaders.pdf.PdfReader", return_value=reader),
        pytest.raises(EncryptedPdfError),
    ):
        PdfLoader().load(source)


def test_image_only_pdf_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    source.touch()
    reader = SimpleNamespace(
        is_encrypted=False,
        pages=[_FakePage(None)],
        metadata=None,
    )

    with (
        patch("advanced_rag.ingestion.loaders.pdf.PdfReader", return_value=reader),
        pytest.raises(ImageOnlyPdfError),
    ):
        PdfLoader().load(source)

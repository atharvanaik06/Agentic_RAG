from pathlib import Path

import pytest

from advanced_rag.config import Settings
from advanced_rag.ingestion.discovery import discover_files
from advanced_rag.ingestion.models import IssueCode
from advanced_rag.ingestion.normalization import normalize_text
from advanced_rag.ingestion.pipeline import IngestionPipeline


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_size=5,
    )


def test_pipeline_ingests_supported_files_and_preserves_metadata(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("A plain text specification.", encoding="utf-8")
    (tmp_path / "guide.md").write_text(
        "# Guide\n\n## Authentication\n\nTokens expire after one hour.",
        encoding="utf-8",
    )

    result = IngestionPipeline(_settings()).ingest(tmp_path)

    assert result.discovered_files == 2
    assert result.processed_files == 2
    assert result.failed_files == 0
    assert len(result.chunks) == 2
    markdown = next(chunk for chunk in result.chunks if chunk.metadata.filename == "guide.md")
    assert markdown.metadata.heading_path == ("Guide", "Authentication")
    assert markdown.metadata.source_path == "guide.md"
    assert markdown.token_count > 0


def test_repeated_ingestion_produces_stable_ids(tmp_path: Path) -> None:
    source = tmp_path / "spec.txt"
    source.write_text("A stable document body.", encoding="utf-8")
    pipeline = IngestionPipeline(_settings())

    first = pipeline.ingest(tmp_path)
    second = pipeline.ingest(tmp_path)

    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert first.chunks[0].metadata.source_id == second.chunks[0].metadata.source_id
    assert first.chunks[0].metadata.content_hash == second.chunks[0].metadata.content_hash


def test_pipeline_reports_duplicates_unsupported_and_empty_files(tmp_path: Path) -> None:
    (tmp_path / "original.txt").write_text("identical", encoding="utf-8")
    (tmp_path / "duplicate.txt").write_text("identical", encoding="utf-8")
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "archive.csv").write_text("unsupported", encoding="utf-8")

    result = IngestionPipeline(_settings()).ingest(tmp_path)

    assert result.discovered_files == 4
    assert result.processed_files == 1
    assert result.skipped_files == 2
    assert result.failed_files == 1
    assert {issue.code for issue in result.issues} == {
        IssueCode.DUPLICATE,
        IssueCode.EMPTY,
        IssueCode.UNSUPPORTED,
    }


def test_hidden_files_are_not_discovered(tmp_path: Path) -> None:
    (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("hidden", encoding="utf-8")
    hidden_directory = tmp_path / ".cache"
    hidden_directory.mkdir()
    (hidden_directory / "nested.txt").write_text("hidden", encoding="utf-8")

    assert discover_files(tmp_path) == [tmp_path / "visible.txt"]


def test_missing_source_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        IngestionPipeline(_settings()).ingest(tmp_path / "missing")


def test_normalization_repairs_pdf_wraps_and_preserves_paragraphs() -> None:
    source = "inter-\r\nnational\x00  \n\n\nSecond paragraph"

    normalized = normalize_text(source, repair_pdf_wraps=True)

    assert normalized == "international\n\nSecond paragraph"

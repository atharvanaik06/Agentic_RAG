from pathlib import Path

import pytest

from advanced_rag.config import Settings
from advanced_rag.ingestion.models import IngestionResult
from advanced_rag.interfaces.documents import (
    CorpusBuildError,
    CorpusInspection,
    UploadPayload,
    delete_corpus_files,
    inspect_corpus,
    save_uploads,
    synchronize_corpus,
)
from advanced_rag.runtime import create_dense_index, create_sparse_index


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "raw",
        index_dir=tmp_path / "indexes",
        chroma_dir=tmp_path / "indexes" / "chroma",
        bm25_dir=tmp_path / "indexes" / "bm25",
        chroma_collection="document-management-test",
        embedding_provider="deterministic",
        embedding_dimensions=16,
        chunk_size=50,
        chunk_overlap=10,
        min_chunk_size=1,
    )


def test_uploads_are_validated_skipped_and_atomically_replaced(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    first = save_uploads(
        data_dir,
        [
            UploadPayload("notes.txt", b"first version"),
            UploadPayload("../escape.txt", b"unsafe"),
            UploadPayload("table.csv", b"unsupported"),
            UploadPayload("empty.md", b""),
        ],
        overwrite=False,
        maximum_bytes=1024,
    )

    assert first.saved == ("notes.txt",)
    assert len(first.rejected) == 3
    assert (data_dir / "notes.txt").read_bytes() == b"first version"
    assert not (tmp_path / "escape.txt").exists()

    skipped = save_uploads(
        data_dir,
        [UploadPayload("notes.txt", b"second version")],
        overwrite=False,
        maximum_bytes=1024,
    )
    replaced = save_uploads(
        data_dir,
        [UploadPayload("notes.txt", b"second version")],
        overwrite=True,
        maximum_bytes=1024,
    )

    assert skipped.skipped == ("notes.txt",)
    assert replaced.replaced == ("notes.txt",)
    assert (data_dir / "notes.txt").read_bytes() == b"second version"


def test_inspection_reports_documents_chunks_and_unsupported_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "policy.txt").write_text(
        "Inflation and monetary policy interact through aggregate demand.",
        encoding="utf-8",
    )
    (settings.data_dir / "brief.md").write_text(
        "# Policy brief\n\nCentral banks adjust policy rates.",
        encoding="utf-8",
    )
    (settings.data_dir / "ignored.csv").write_text("not,supported", encoding="utf-8")

    inspection = inspect_corpus(settings)

    assert inspection.ingestion.discovered_files == 3
    assert inspection.ingestion.processed_files == 2
    assert inspection.ingestion.skipped_files == 1
    assert inspection.ingestion.failed_files == 0
    assert inspection.chunks == 2
    assert inspection.tokens > 0
    assert {document.status for document in inspection.documents} == {
        "ready",
        "unsupported_file",
    }


def test_failed_inspection_blocks_both_indexes_before_mutation(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()
    dense = create_dense_index(settings)
    sparse = create_sparse_index(settings)
    failed = CorpusInspection(
        ingestion=IngestionResult(
            discovered_files=1,
            processed_files=0,
            skipped_files=0,
            failed_files=1,
        ),
        documents=(),
    )

    with pytest.raises(CorpusBuildError, match="failed extraction"):
        synchronize_corpus(settings, dense, sparse, inspection=failed)

    assert dense.info().count == 0
    assert sparse.info().count == 0


def test_dual_index_synchronization_is_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()
    (settings.data_dir / "policy.txt").write_text(
        "Inflation expectations affect wage setting and monetary policy decisions.",
        encoding="utf-8",
    )
    dense = create_dense_index(settings)
    sparse = create_sparse_index(settings)

    first = synchronize_corpus(settings, dense, sparse)
    second = synchronize_corpus(settings, dense, sparse)

    assert first.dense.embedded_chunks == first.inspection.chunks
    assert first.dense.total_chunks == first.sparse.total_chunks
    assert second.dense.embedded_chunks == 0
    assert second.dense.unchanged_chunks == first.inspection.chunks
    assert second.sparse.rebuilt is False


def test_delete_is_scoped_to_the_corpus_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "delete.txt").write_text("delete me", encoding="utf-8")
    outside = tmp_path / "keep.txt"
    outside.write_text("keep me", encoding="utf-8")

    assert delete_corpus_files(data_dir, ["delete.txt"]) == ("delete.txt",)
    with pytest.raises(ValueError, match="escapes"):
        delete_corpus_files(data_dir, ["../keep.txt"])
    assert outside.exists()

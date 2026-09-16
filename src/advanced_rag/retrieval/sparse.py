"""Persistent BM25 keyword indexing over canonical document chunks."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import bm25s
import Stemmer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from advanced_rag.ingestion.hashing import stable_id
from advanced_rag.ingestion.models import DocumentChunk, IngestionResult
from advanced_rag.ingestion.pipeline import IngestionPipeline
from advanced_rag.retrieval.errors import SparseIndexError
from advanced_rag.retrieval.models import (
    RetrievalFilters,
    SparseIndexInfo,
    SparseIndexingResult,
    SparseSearchResult,
)

_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_CORPUS_NAME = "chunks.jsonl"
_TOKENIZER = "bm25s-english-stemmed-decimals-v2"
_TOKEN_PATTERN = r"(?u)\b(?:\d+(?:\.\d+)+|\w+)\b"


class _Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    fingerprint: str
    count: int = Field(ge=0)
    method: str
    k1: float
    b: float
    tokenizer: str


class BM25SparseIndex:
    """Build, persist, memory-map, and query a local BM25S index."""

    def __init__(
        self,
        *,
        path: Path,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.path = path
        self.method = method
        self.k1 = k1
        self.b = b
        self._stemmer = Stemmer.Stemmer("english")
        self._retriever: Any | None = None
        self._chunks: tuple[DocumentChunk, ...] | None = None

    def index_path(
        self,
        source: Path | str,
        *,
        pipeline: IngestionPipeline | None = None,
    ) -> tuple[SparseIndexingResult, IngestionResult]:
        """Ingest a source path and synchronize its complete canonical corpus."""
        ingestion = (pipeline or IngestionPipeline()).ingest(source)
        result = self.sync(ingestion)
        return result, ingestion

    def sync(self, ingestion: IngestionResult) -> SparseIndexingResult:
        """Rebuild on corpus changes and skip byte-equivalent configurations."""
        chunks = tuple(sorted(ingestion.chunks, key=lambda chunk: chunk.chunk_id))
        fingerprint = _corpus_fingerprint(chunks)
        current = self._read_manifest(required=False)
        unchanged = current == self._expected_manifest(fingerprint, len(chunks))
        if unchanged and self._artifacts_exist(current):
            rebuilt = False
        else:
            self._rebuild(chunks, fingerprint)
            rebuilt = True

        return SparseIndexingResult(
            discovered_files=ingestion.discovered_files,
            processed_files=ingestion.processed_files,
            failed_files=ingestion.failed_files,
            rebuilt=rebuilt,
            indexed_chunks=len(chunks) if rebuilt else 0,
            total_chunks=len(chunks),
            fingerprint=fingerprint,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: RetrievalFilters | None = None,
    ) -> list[SparseSearchResult]:
        """Return positive-score BM25 matches with citation-ready metadata."""
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        retriever, chunks = self._load()
        if not chunks:
            return []

        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=self._stemmer,
            token_pattern=_TOKEN_PATTERN,
            show_progress=False,
        )
        candidate_count = len(chunks) if filters else min(top_k, len(chunks))
        response = retriever.retrieve(
            query_tokens,
            k=candidate_count,
            show_progress=False,
        )
        document_ids = cast(Any, response.documents)[0]
        scores = cast(Any, response.scores)[0]

        results: list[SparseSearchResult] = []
        for document_id, score in zip(document_ids, scores, strict=True):
            numeric_score = float(score)
            chunk = chunks[int(document_id)]
            if numeric_score <= 0 or (filters and not filters.matches(chunk)):
                continue
            results.append(
                SparseSearchResult(
                    chunk=chunk,
                    rank=len(results) + 1,
                    score=numeric_score,
                )
            )
            if len(results) == top_k:
                break
        return results

    def info(self) -> SparseIndexInfo:
        """Return persisted index metadata without loading the score matrix."""
        manifest = self._read_manifest(required=False)
        return SparseIndexInfo(
            path=str(self.path),
            count=manifest.count if manifest else 0,
            method=manifest.method if manifest else self.method,
            k1=manifest.k1 if manifest else self.k1,
            b=manifest.b if manifest else self.b,
            fingerprint=manifest.fingerprint if manifest else None,
            schema_version=manifest.schema_version if manifest else _SCHEMA_VERSION,
        )

    def get_chunks(self, chunk_ids: set[str]) -> dict[str, DocumentChunk]:
        """Return canonical chunks by ID for citation auditing."""
        _retriever, chunks = self._load()
        return {chunk.chunk_id: chunk for chunk in chunks if chunk.chunk_id in chunk_ids}

    def reset(self) -> None:
        """Delete only this generated sparse-index directory."""
        if self.path.exists():
            shutil.rmtree(self.path)
        self._retriever = None
        self._chunks = None

    def _rebuild(self, chunks: tuple[DocumentChunk, ...], fingerprint: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{self.path.name}-", dir=self.path.parent))
        backup = self.path.with_name(f".{self.path.name}-backup")
        try:
            if chunks:
                corpus_tokens = bm25s.tokenize(
                    [chunk.text for chunk in chunks],
                    stopwords="en",
                    stemmer=self._stemmer,
                    token_pattern=_TOKEN_PATTERN,
                    show_progress=False,
                )
                retriever = bm25s.BM25(k1=self.k1, b=self.b, method=self.method)
                retriever.index(corpus_tokens, show_progress=False)
                retriever.save(temporary, show_progress=False)

            corpus_path = temporary / _CORPUS_NAME
            with corpus_path.open("w", encoding="utf-8") as corpus_file:
                for chunk in chunks:
                    corpus_file.write(chunk.model_dump_json() + "\n")
            manifest = self._expected_manifest(fingerprint, len(chunks))
            (temporary / _MANIFEST_NAME).write_text(
                manifest.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )

            if backup.exists():
                shutil.rmtree(backup)
            if self.path.exists():
                os.replace(self.path, backup)
            os.replace(temporary, self.path)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if backup.exists() and not self.path.exists():
                os.replace(backup, self.path)
            raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self._retriever = None
        self._chunks = None

    def _load(self) -> tuple[Any, tuple[DocumentChunk, ...]]:
        if self._retriever is not None and self._chunks is not None:
            return self._retriever, self._chunks
        manifest = self._read_manifest(required=True)
        assert manifest is not None
        if not self._artifacts_exist(manifest):
            raise SparseIndexError(
                f"Sparse index at '{self.path}' is incomplete; run `rag index-sparse`."
            )
        try:
            chunks = tuple(
                DocumentChunk.model_validate_json(line)
                for line in (self.path / _CORPUS_NAME).read_text(encoding="utf-8").splitlines()
                if line
            )
            if len(chunks) != manifest.count:
                raise ValueError("corpus count differs from manifest")
            retriever = (
                bm25s.BM25.load(self.path, mmap=True, show_progress=False) if chunks else None
            )
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            raise SparseIndexError(
                f"Sparse index at '{self.path}' is corrupt; run `rag index-sparse`."
            ) from exc
        self._retriever = retriever
        self._chunks = chunks
        return retriever, chunks

    def _read_manifest(self, *, required: bool) -> _Manifest | None:
        manifest_path = self.path / _MANIFEST_NAME
        if not manifest_path.exists():
            if required:
                raise SparseIndexError(
                    f"Sparse index not found at '{self.path}'; run `rag index-sparse`."
                )
            return None
        try:
            manifest = _Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            if required:
                raise SparseIndexError(
                    f"Sparse index manifest at '{manifest_path}' is invalid; "
                    "run `rag index-sparse`."
                ) from exc
            return None
        if manifest.schema_version != _SCHEMA_VERSION:
            if required:
                raise SparseIndexError(
                    "Sparse index schema is incompatible; run `rag index-sparse`."
                )
            return None
        return manifest

    def _expected_manifest(self, fingerprint: str, count: int) -> _Manifest:
        return _Manifest(
            schema_version=_SCHEMA_VERSION,
            fingerprint=fingerprint,
            count=count,
            method=self.method,
            k1=self.k1,
            b=self.b,
            tokenizer=_TOKENIZER,
        )

    def _artifacts_exist(self, manifest: _Manifest | None) -> bool:
        if manifest is None or not (self.path / _CORPUS_NAME).is_file():
            return False
        if manifest.count == 0:
            return True
        return all(
            (self.path / filename).is_file()
            for filename in (
                "data.csc.index.npy",
                "indices.csc.index.npy",
                "indptr.csc.index.npy",
                "vocab.index.json",
                "params.index.json",
            )
        )


def _corpus_fingerprint(chunks: tuple[DocumentChunk, ...]) -> str:
    return stable_id(
        _SCHEMA_VERSION,
        _TOKENIZER,
        *(f"{chunk.chunk_id}:{chunk.metadata.content_hash}" for chunk in chunks),
    )

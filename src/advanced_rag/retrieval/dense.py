"""Persistent ChromaDB indexing and dense similarity search."""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Embeddings, Metadata

from advanced_rag.ingestion.discovery import discover_files, relative_source_path
from advanced_rag.ingestion.hashing import stable_id
from advanced_rag.ingestion.models import ChunkMetadata, DocumentChunk, IngestionResult
from advanced_rag.ingestion.pipeline import IngestionPipeline
from advanced_rag.retrieval.embeddings import EmbeddingProvider, _validate_vectors
from advanced_rag.retrieval.errors import DenseIndexError, EmbeddingConfigurationError
from advanced_rag.retrieval.models import (
    CollectionInfo,
    DenseSearchFilters,
    DenseSearchResult,
    IndexingResult,
    StoredMetadata,
)

_SCHEMA_VERSION = 1


class ChromaDenseIndex:
    """Synchronize canonical chunks and query a persistent Chroma collection."""

    def __init__(
        self,
        *,
        path: Path,
        collection_name: str,
        embedder: EmbeddingProvider,
        write_batch_size: int = 100,
    ) -> None:
        if write_batch_size < 1:
            raise ValueError("write_batch_size must be positive")
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder
        self.write_batch_size = write_batch_size
        self.client: ClientAPI = chromadb.PersistentClient(path=str(path))
        self.collection: Collection = self._get_or_create_collection()
        self._validate_collection_compatibility()

    def _get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "schema_version": _SCHEMA_VERSION,
                "embedding_provider": self.embedder.provider_name,
                "embedding_model": self.embedder.model_name,
                "embedding_dimensions": self.embedder.dimensions,
            },
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def _validate_collection_compatibility(self) -> None:
        metadata = self.collection.metadata or {}
        expected: dict[str, str | int] = {
            "schema_version": _SCHEMA_VERSION,
            "embedding_provider": self.embedder.provider_name,
            "embedding_model": self.embedder.model_name,
            "embedding_dimensions": self.embedder.dimensions,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (expected {expected_value!r})"
                for key, (actual, expected_value) in mismatches.items()
            )
            raise EmbeddingConfigurationError(
                f"Collection '{self.collection_name}' is incompatible: {details}. "
                "Use the original embedding configuration or reset the collection."
            )

    def index_path(
        self,
        source: Path | str,
        *,
        pipeline: IngestionPipeline | None = None,
        prune_missing: bool = True,
    ) -> tuple[IndexingResult, IngestionResult]:
        """Ingest a path and synchronize its supported sources into Chroma."""
        source_path = Path(source)
        ingestion_pipeline = pipeline or IngestionPipeline()
        ingestion = ingestion_pipeline.ingest(source_path)
        active_source_ids = {
            stable_id(relative_source_path(path, source_path))
            for path in discover_files(source_path)
            if path.suffix.lower() in ingestion_pipeline.loaders
        }
        result = self.sync(
            ingestion,
            active_source_ids=active_source_ids,
            prune_missing=prune_missing and source_path.is_dir(),
        )
        return result, ingestion

    def sync(
        self,
        ingestion: IngestionResult,
        *,
        active_source_ids: set[str] | None = None,
        prune_missing: bool = False,
    ) -> IndexingResult:
        """Upsert changed sources, skip unchanged chunks, and remove stale records."""
        existing = self._existing_records()
        existing_by_source: dict[str, set[str]] = defaultdict(set)
        for chunk_id, metadata in existing.items():
            source_id = metadata.get("source_id")
            if isinstance(source_id, str):
                existing_by_source[source_id].add(chunk_id)

        current_by_source: dict[str, list[DocumentChunk]] = defaultdict(list)
        for chunk in ingestion.chunks:
            current_by_source[chunk.metadata.source_id].append(chunk)

        changed_chunks: list[DocumentChunk] = []
        stale_ids: set[str] = set()
        unchanged_count = 0

        for source_id, chunks in current_by_source.items():
            current_ids = {chunk.chunk_id for chunk in chunks}
            existing_ids = existing_by_source.get(source_id, set())
            hashes = {
                metadata.get("content_hash")
                for chunk_id, metadata in existing.items()
                if chunk_id in existing_ids
            }
            unchanged = (
                current_ids == existing_ids
                and len(hashes) == 1
                and next(iter(hashes), None) == chunks[0].metadata.content_hash
            )
            if unchanged:
                unchanged_count += len(chunks)
                continue
            changed_chunks.extend(chunks)
            stale_ids.update(existing_ids - current_ids)

        if prune_missing and active_source_ids is not None:
            removed_sources = set(existing_by_source) - active_source_ids
            for source_id in removed_sources:
                stale_ids.update(existing_by_source[source_id])

        vectors = self.embedder.embed_documents([chunk.text for chunk in changed_chunks])
        _validate_vectors(
            vectors,
            expected_count=len(changed_chunks),
            dimensions=self.embedder.dimensions,
        )
        for start in range(0, len(changed_chunks), self.write_batch_size):
            batch_chunks = changed_chunks[start : start + self.write_batch_size]
            batch_vectors = vectors[start : start + self.write_batch_size]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                embeddings=cast(Embeddings, batch_vectors),
                metadatas=[_to_chroma_metadata(chunk) for chunk in batch_chunks],
            )

        if stale_ids:
            for ids in _batches(sorted(stale_ids), self.write_batch_size):
                self.collection.delete(ids=list(ids))

        return IndexingResult(
            discovered_files=ingestion.discovered_files,
            processed_files=ingestion.processed_files,
            failed_files=ingestion.failed_files,
            embedded_chunks=len(changed_chunks),
            upserted_chunks=len(changed_chunks),
            unchanged_chunks=unchanged_count,
            deleted_chunks=len(stale_ids),
            total_chunks=self.collection.count(),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: DenseSearchFilters | None = None,
    ) -> list[DenseSearchResult]:
        """Embed a query and return ranked, citation-ready chunks."""
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        count = self.collection.count()
        if count == 0:
            return []

        response = self.collection.query(
            query_embeddings=cast(Embeddings, [self.embedder.embed_query(query)]),
            n_results=min(top_k, count),
            where=filters.to_chroma() if filters else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        documents = _first_result_list(response.get("documents"))
        metadatas = _first_result_list(response.get("metadatas"))
        distances = _first_result_list(response.get("distances"))
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            raise DenseIndexError("Chroma returned misaligned dense-search fields")

        results: list[DenseSearchResult] = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True),
            start=1,
        ):
            if (
                not isinstance(document, str)
                or metadata is None
                or not isinstance(distance, (int, float))
            ):
                raise DenseIndexError("Chroma returned an incomplete dense-search record")
            stored = StoredMetadata.model_validate(metadata)
            numeric_distance = float(distance)
            chunk = _from_chroma_record(chunk_id, document, stored)
            results.append(
                DenseSearchResult(
                    chunk=chunk,
                    rank=rank,
                    distance=numeric_distance,
                    score=1.0 - numeric_distance,
                )
            )
        return results

    def info(self) -> CollectionInfo:
        """Return collection state without loading documents or vectors."""
        return CollectionInfo(
            name=self.collection.name,
            count=self.collection.count(),
            path=str(self.path),
            embedding_provider=self.embedder.provider_name,
            embedding_model=self.embedder.model_name,
            embedding_dimensions=self.embedder.dimensions,
        )

    def reset(self) -> None:
        """Permanently delete and recreate this collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()

    def _existing_records(self) -> dict[str, Mapping[str, object]]:
        response = self.collection.get(include=["metadatas"])
        metadatas = response.get("metadatas") or []
        return {
            chunk_id: cast(Mapping[str, object], metadata or {})
            for chunk_id, metadata in zip(response["ids"], metadatas, strict=True)
        }


def _to_chroma_metadata(chunk: DocumentChunk) -> Metadata:
    metadata = chunk.metadata
    stored = StoredMetadata(
        source_id=metadata.source_id,
        source_path=metadata.source_path,
        filename=metadata.filename,
        file_type=metadata.file_type,
        content_hash=metadata.content_hash,
        title=metadata.title or "",
        page_number=metadata.page_number or -1,
        heading_path=json.dumps(metadata.heading_path),
        chunk_index=metadata.chunk_index,
        token_count=chunk.token_count,
    )
    return cast(Metadata, stored.model_dump(mode="json"))


def _from_chroma_record(chunk_id: str, text: str, stored: StoredMetadata) -> DocumentChunk:
    try:
        heading_path = tuple(json.loads(stored.heading_path))
    except (json.JSONDecodeError, TypeError) as error:
        raise DenseIndexError(f"Invalid heading metadata for chunk {chunk_id}") from error
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        token_count=stored.token_count,
        metadata=ChunkMetadata(
            source_id=stored.source_id,
            source_path=stored.source_path,
            filename=stored.filename,
            file_type=stored.file_type,
            content_hash=stored.content_hash,
            title=stored.title or None,
            page_number=None if stored.page_number == -1 else stored.page_number,
            heading_path=heading_path,
            chunk_index=stored.chunk_index,
        ),
    )


def _batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _first_result_list(value: object) -> list[object]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    return list(first) if isinstance(first, list) else []

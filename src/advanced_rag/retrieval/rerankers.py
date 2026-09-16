"""Pluggable local rerankers for fused retrieval candidates."""

from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import Any, Protocol, cast

from advanced_rag.config import Settings
from advanced_rag.retrieval.errors import RerankerError
from advanced_rag.retrieval.hybrid_models import FusedCandidate, RerankScore


class Reranker(Protocol):
    """Minimal interface implemented by local or hosted rerankers."""

    @property
    def name(self) -> str: ...

    def rerank(self, query: str, candidates: Sequence[FusedCandidate]) -> list[RerankScore]: ...


class NoOpReranker:
    """Preserve fused order when reranking is explicitly disabled."""

    @property
    def name(self) -> str:
        return "none"

    def rerank(self, query: str, candidates: Sequence[FusedCandidate]) -> list[RerankScore]:
        del query
        return [
            RerankScore(chunk_id=candidate.chunk.chunk_id, score=candidate.rrf_score)
            for candidate in candidates
        ]


class FlashRankReranker:
    """Lazy local cross-encoder backed by FlashRank."""

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: Path,
        max_length: int = 512,
    ) -> None:
        if max_length < 64:
            raise ValueError("max_length must be at least 64")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self._ranker: Any | None = None

    @property
    def name(self) -> str:
        return f"flashrank:{self.model_name}"

    def rerank(self, query: str, candidates: Sequence[FusedCandidate]) -> list[RerankScore]:
        if not candidates:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._ranker is None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._ranker = Ranker(
                    model_name=self.model_name,
                    cache_dir=str(self.cache_dir),
                    max_length=self.max_length,
                    log_level="WARNING",
                )
            passages = [
                {"id": candidate.chunk.chunk_id, "text": candidate.chunk.text}
                for candidate in candidates
            ]
            raw = self._ranker.rerank(RerankRequest(query=query, passages=passages))
            scores = [self._parse_result(item) for item in raw]
        except Exception as exc:
            raise RerankerError(f"FlashRank reranking failed: {exc}") from exc

        expected = {candidate.chunk.chunk_id for candidate in candidates}
        returned = {result.chunk_id for result in scores}
        if returned != expected or len(scores) != len(candidates):
            raise RerankerError("FlashRank returned missing, duplicate, or unknown chunk IDs")
        return scores

    @staticmethod
    def _parse_result(item: object) -> RerankScore:
        if not isinstance(item, dict):
            raise RerankerError("FlashRank returned a non-object result")
        chunk_id = item.get("id")
        score = item.get("score")
        if not isinstance(chunk_id, str) or isinstance(score, bool):
            raise RerankerError("FlashRank returned an invalid ID or score")
        try:
            numeric_score = float(cast(Any, score))
        except (TypeError, ValueError) as exc:
            raise RerankerError("FlashRank returned an invalid ID or score") from exc
        if not isfinite(numeric_score):
            raise RerankerError("FlashRank returned a non-finite score")
        return RerankScore(chunk_id=chunk_id, score=numeric_score)


def create_reranker(settings: Settings) -> Reranker:
    """Create the configured reranker without loading its model eagerly."""
    if settings.reranker_provider == "none":
        return NoOpReranker()
    return FlashRankReranker(
        model_name=settings.reranker_model,
        cache_dir=settings.reranker_cache_dir,
        max_length=settings.reranker_max_length,
    )

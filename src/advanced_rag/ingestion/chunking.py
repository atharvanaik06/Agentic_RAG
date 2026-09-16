"""Token-aware, boundary-sensitive text chunking."""

import re
from collections.abc import Sequence

import tiktoken
from tiktoken import Encoding

_BOUNDARY = re.compile(r"\n\n|\n|(?<=[.!?])\s+")


class TokenChunker:
    """Split text by token budget while preferring natural boundaries."""

    def __init__(
        self,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        min_chunk_size: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if not 1 <= min_chunk_size <= chunk_size:
            raise ValueError("min_chunk_size must be between 1 and chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.encoding: Encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using the configured embedding-compatible encoding."""
        return len(self.encoding.encode(text))

    def split(self, text: str) -> list[str]:
        """Split non-empty text deterministically into overlapping chunks."""
        tokens = self.encoding.encode(text)
        if not tokens:
            return []
        if len(tokens) <= self.chunk_size:
            return [text.strip()]

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            remaining = len(tokens) - start
            if remaining <= self.chunk_size + self.min_chunk_size:
                end = len(tokens)
            else:
                end = self._natural_end(tokens, start)

            chunk = self.encoding.decode(tokens[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(tokens):
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _natural_end(self, tokens: Sequence[int], start: int) -> int:
        proposed_end = min(start + self.chunk_size, len(tokens))
        candidate = self.encoding.decode(tokens[start:proposed_end])
        minimum_character = int(len(candidate) * 0.6)
        boundaries = [match.end() for match in _BOUNDARY.finditer(candidate)]
        usable = [position for position in boundaries if position >= minimum_character]
        if not usable:
            return proposed_end

        prefix_tokens = self.encoding.encode(candidate[: usable[-1]])
        return min(start + len(prefix_tokens), proposed_end)

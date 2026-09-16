from advanced_rag.ingestion.chunking import TokenChunker


def test_short_text_remains_one_chunk() -> None:
    chunker = TokenChunker(chunk_size=20, chunk_overlap=4, min_chunk_size=3)

    assert chunker.split("A short technical sentence.") == ["A short technical sentence."]


def test_long_text_is_split_within_configured_budget() -> None:
    chunker = TokenChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    text = "\n\n".join(
        f"Paragraph {number} explains authentication behavior." for number in range(30)
    )

    chunks = chunker.split(text)

    assert len(chunks) > 1
    # The final window may absorb the configured minimum remainder.
    assert all(chunker.count_tokens(chunk) <= 35 for chunk in chunks)
    assert "Paragraph 0" in chunks[0]
    assert "Paragraph 29" in chunks[-1]


def test_invalid_chunk_configuration_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="positive"):
        TokenChunker(chunk_size=0)
    with pytest.raises(ValueError, match="smaller"):
        TokenChunker(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError, match="between"):
        TokenChunker(chunk_size=10, chunk_overlap=2, min_chunk_size=11)

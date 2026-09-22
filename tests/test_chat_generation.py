from types import SimpleNamespace
from typing import Any, cast

import pytest

from advanced_rag.generation.chat import OpenAIChatModel, create_chat_model
from advanced_rag.generation.models import (
    AnswerClaim,
    EvidenceGrade,
    GroundedDraft,
    QueryRewrite,
)
from advanced_rag.ingestion.models import ChunkMetadata, DocumentChunk, FileType
from advanced_rag.retrieval.errors import ChatModelError, EmbeddingConfigurationError
from advanced_rag.retrieval.hybrid_models import HybridSearchResult


def _result() -> HybridSearchResult:
    return HybridSearchResult(
        chunk=DocumentChunk(
            chunk_id="chunk-1",
            text="Inflation evidence.",
            token_count=3,
            metadata=ChunkMetadata(
                source_id="paper",
                source_path="paper.pdf",
                filename="paper.pdf",
                file_type=FileType.PDF,
                content_hash="hash",
                page_number=1,
                chunk_index=0,
            ),
        ),
        rrf_score=0.03,
        retrieval_sources=("dense",),
        final_rank=1,
    )


class FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.arguments: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.arguments = kwargs
        return SimpleNamespace(
            output_parsed=self.parsed,
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
        )


def test_openai_chat_uses_structured_responses_without_storage() -> None:
    model = OpenAIChatModel(api_key="test", model="gpt-4o-mini")
    responses = FakeResponses(
        GroundedDraft(claims=[AnswerClaim(text="Supported.", citations=["S1"])])
    )
    model.client = cast(Any, SimpleNamespace(responses=responses))

    draft, usage = model.generate_answer(question="Question", evidence=[_result()])

    assert draft.claims[0].citations == ["S1"]
    assert usage.api_calls == 1
    assert responses.arguments["store"] is False
    assert responses.arguments["temperature"] == 0
    assert responses.arguments["text_format"] is GroundedDraft
    assert "[S1]" in responses.arguments["input"]


def test_openai_chat_rewrites_and_rejects_empty_output() -> None:
    model = OpenAIChatModel(api_key="test", model="gpt-4o-mini")
    responses = FakeResponses(QueryRewrite(query="focused monetary query"))
    model.client = cast(Any, SimpleNamespace(responses=responses))

    query, usage = model.rewrite_query(question="Question", previous_query="Question", evidence=[])
    assert query == "focused monetary query"
    assert usage.input_tokens == 12

    model.client = cast(Any, SimpleNamespace(responses=FakeResponses(None)))
    with pytest.raises(ChatModelError, match="no structured output"):
        model.generate_answer(question="Question", evidence=[_result()])


def test_openai_chat_grades_direct_evidence_with_structured_output() -> None:
    model = OpenAIChatModel(api_key="test", model="gpt-4o-mini")
    responses = FakeResponses(
        EvidenceGrade(
            sufficient=True,
            supporting_labels=["S1"],
            reason="S1 directly answers the question.",
        )
    )
    model.client = cast(Any, SimpleNamespace(responses=responses))

    grade, usage = model.grade_evidence(question="Question", evidence=[_result()])

    assert grade.sufficient is True
    assert grade.supporting_labels == ["S1"]
    assert usage.api_calls == 1
    assert responses.arguments["text_format"] is EvidenceGrade
    assert "directly contains enough information" in responses.arguments["instructions"]


def test_chat_model_requires_user_api_key() -> None:
    from advanced_rag.config import Settings

    with pytest.raises(EmbeddingConfigurationError, match="RAG_OPENAI_API_KEY"):
        create_chat_model(Settings(_env_file=None, openai_api_key=None))

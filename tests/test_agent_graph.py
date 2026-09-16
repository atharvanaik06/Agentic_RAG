from collections.abc import Sequence

import pytest
from langchain_core.tools import StructuredTool

from advanced_rag.generation.models import AnswerClaim, GroundedDraft, ModelUsage
from advanced_rag.graph import AgenticRAG
from advanced_rag.ingestion.models import ChunkMetadata, DocumentChunk, FileType
from advanced_rag.retrieval.hybrid_models import (
    HybridSearchResponse,
    HybridSearchResult,
    RetrievalDiagnostics,
)


def _result(chunk_id: str = "chunk-1") -> HybridSearchResult:
    return HybridSearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            text="Expansionary demand and constrained supply increased inflation.",
            token_count=10,
            metadata=ChunkMetadata(
                source_id="paper",
                source_path="paper.pdf",
                filename="paper.pdf",
                file_type=FileType.PDF,
                content_hash="hash",
                title="Inflation Study",
                page_number=12,
                chunk_index=1,
            ),
        ),
        rrf_score=0.03,
        dense_rank=1,
        dense_score=0.9,
        sparse_rank=1,
        sparse_score=5.0,
        retrieval_sources=("dense", "sparse"),
        final_rank=1,
        reranker_score=0.95,
    )


def _response(query: str, *, confidence: str = "high", results: bool = True) -> dict[str, object]:
    matches = (_result(),) if results else ()
    response = HybridSearchResponse(
        query=query,
        results=matches,
        diagnostics=RetrievalDiagnostics(
            dense_candidates=1 if results else 0,
            sparse_candidates=1 if results else 0,
            fused_candidates=1 if results else 0,
            reranked_candidates=1 if results else 0,
            agreement_count=1 if results else 0,
            distinct_sources=1 if results else 0,
            evidence_tokens=10 if results else 0,
            confidence=confidence,
            reranker="test",
            rerank_applied=results,
        ),
    )
    return response.model_dump(mode="json")


def _search_tool(responses: list[dict[str, object]]) -> StructuredTool:
    def search_knowledge_base(query: str, top_k: int = 6) -> dict[str, object]:
        """Return the next prepared test response."""
        del top_k
        response = responses.pop(0)
        response["query"] = query
        return response

    return StructuredTool.from_function(search_knowledge_base)


class FakeChatModel:
    def __init__(self, draft: GroundedDraft) -> None:
        self.draft = draft
        self.rewrite_calls = 0
        self.generate_calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def rewrite_query(
        self,
        *,
        question: str,
        previous_query: str,
        evidence: Sequence[HybridSearchResult],
    ) -> tuple[str, ModelUsage]:
        del question, previous_query, evidence
        self.rewrite_calls += 1
        return "inflation demand supply mechanisms", ModelUsage(
            api_calls=1, input_tokens=20, output_tokens=5
        )

    def generate_answer(
        self,
        *,
        question: str,
        evidence: Sequence[HybridSearchResult],
    ) -> tuple[GroundedDraft, ModelUsage]:
        del question, evidence
        self.generate_calls += 1
        return self.draft, ModelUsage(api_calls=1, input_tokens=100, output_tokens=20)


def test_graph_answers_with_validated_citations_in_one_attempt() -> None:
    chat = FakeChatModel(
        GroundedDraft(
            claims=[AnswerClaim(text="Demand and supply both contributed.", citations=["S1"])]
        )
    )
    agent = AgenticRAG(search_tool=_search_tool([_response("query")]), chat_model=chat)

    answer = agent.ask("What drove inflation?")

    assert answer.answer == "- Demand and supply both contributed. [S1]"
    assert answer.sources[0].filename == "paper.pdf"
    assert answer.sources[0].page_number == 12
    assert answer.citation_validation.valid is True
    assert answer.retrieval_attempts == 1
    assert answer.usage == ModelUsage(api_calls=1, input_tokens=100, output_tokens=20)
    assert [event.node for event in answer.graph_trace] == [
        "analyze_question",
        "retrieve_evidence",
        "grade_evidence",
        "generate_answer",
        "validate_answer",
    ]
    assert chat.rewrite_calls == 0
    assert chat.generate_calls == 1


def test_graph_rewrites_once_when_first_evidence_is_weak() -> None:
    chat = FakeChatModel(
        GroundedDraft(claims=[AnswerClaim(text="Supported finding.", citations=["[S1]"])])
    )
    agent = AgenticRAG(
        search_tool=_search_tool(
            [
                _response("first", confidence="low", results=False),
                _response("second", confidence="high"),
            ]
        ),
        chat_model=chat,
        max_retrieval_attempts=2,
    )

    answer = agent.ask("Explain the mechanism")

    assert answer.retrieval_attempts == 2
    assert answer.final_query == "inflation demand supply mechanisms"
    assert answer.usage.api_calls == 2
    assert answer.usage.input_tokens == 120
    assert chat.rewrite_calls == 1
    assert [event.node for event in answer.graph_trace].count("retrieve_evidence") == 2


def test_graph_rejects_hallucinated_citations() -> None:
    chat = FakeChatModel(
        GroundedDraft(
            claims=[
                AnswerClaim(text="Valid claim.", citations=["S1"]),
                AnswerClaim(text="Unsupported claim.", citations=["S99"]),
            ]
        )
    )
    agent = AgenticRAG(search_tool=_search_tool([_response("query")]), chat_model=chat)

    answer = agent.ask("Question")

    assert "Valid claim" in answer.answer
    assert "Unsupported claim" not in answer.answer
    assert answer.citation_validation.valid is False
    assert answer.citation_validation.rejected_claims == 1
    assert "unknown citations" in answer.citation_validation.issues[0]


def test_graph_stops_after_limit_and_skips_generation_without_evidence() -> None:
    chat = FakeChatModel(GroundedDraft())
    agent = AgenticRAG(
        search_tool=_search_tool(
            [
                _response("first", confidence="low", results=False),
                _response("second", confidence="low", results=False),
            ]
        ),
        chat_model=chat,
        max_retrieval_attempts=2,
    )

    answer = agent.ask("Unanswerable question")

    assert answer.insufficient_evidence is True
    assert answer.sources == ()
    assert answer.retrieval_attempts == 2
    assert answer.usage.api_calls == 1
    assert chat.generate_calls == 0


def test_graph_validates_public_inputs() -> None:
    agent = AgenticRAG(
        search_tool=_search_tool([_response("query")]),
        chat_model=FakeChatModel(GroundedDraft()),
    )
    with pytest.raises(ValueError, match="must not be empty"):
        agent.ask(" ")
    with pytest.raises(ValueError, match="top_k"):
        agent.ask("question", top_k=21)
    with pytest.raises(ValueError, match="max_retrieval_attempts"):
        AgenticRAG(
            search_tool=_search_tool([_response("query")]),
            chat_model=FakeChatModel(GroundedDraft()),
            max_retrieval_attempts=6,
        )

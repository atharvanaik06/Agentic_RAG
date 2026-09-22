from collections.abc import Sequence

import pytest
from langchain_core.tools import StructuredTool

from advanced_rag.generation.models import (
    AnswerClaim,
    EvidenceGrade,
    GroundedDraft,
    ModelUsage,
)
from advanced_rag.graph import AgenticRAG
from advanced_rag.graph.workflow import bounded_retrieval_attempts, question_matches_scope
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


class FakeEvidenceGrader:
    def __init__(self, grades: list[EvidenceGrade]) -> None:
        self.grades = grades
        self.calls = 0

    def grade_evidence(
        self,
        *,
        question: str,
        evidence: Sequence[HybridSearchResult],
    ) -> tuple[EvidenceGrade, ModelUsage]:
        del question, evidence
        self.calls += 1
        return self.grades.pop(0), ModelUsage(
            api_calls=1,
            input_tokens=30,
            output_tokens=5,
        )


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
    assert answer.sources[0].text.startswith("Expansionary demand")
    assert answer.sources[0].dense_rank == 1
    assert answer.sources[0].sparse_rank == 1
    assert answer.sources[0].reranker_score == 0.95
    assert answer.sources[0].retrieval_sources == ("dense", "sparse")
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


def test_semantic_grader_rewrites_related_evidence_then_allows_direct_evidence() -> None:
    chat = FakeChatModel(
        GroundedDraft(claims=[AnswerClaim(text="Supported finding.", citations=["S1"])])
    )
    grader = FakeEvidenceGrader(
        [
            EvidenceGrade(
                sufficient=False,
                reason="The passage is only topically related.",
                missing_information="The requested policy decision.",
            ),
            EvidenceGrade(
                sufficient=True,
                supporting_labels=["S1"],
                reason="S1 directly states the requested decision.",
            ),
        ]
    )
    agent = AgenticRAG(
        search_tool=_search_tool([_response("first"), _response("second")]),
        chat_model=chat,
        evidence_grader=grader,
        semantic_evidence_grading=True,
        max_retrieval_attempts=2,
    )

    answer = agent.ask("What policy decision was made?")

    assert answer.insufficient_evidence is False
    assert answer.retrieval_attempts == 2
    assert grader.calls == 2
    assert chat.rewrite_calls == 1
    assert chat.generate_calls == 1
    assert answer.usage == ModelUsage(api_calls=4, input_tokens=180, output_tokens=35)
    assert [event.action for event in answer.graph_trace].count("graded_semantically") == 2


def test_semantic_grader_refuses_after_retry_without_generating() -> None:
    chat = FakeChatModel(GroundedDraft())
    grader = FakeEvidenceGrader(
        [
            EvidenceGrade(
                sufficient=False,
                reason="The value is absent.",
                missing_information="The requested interest rate.",
            ),
            EvidenceGrade(
                sufficient=False,
                reason="The rewritten search is still indirect.",
                missing_information="The requested interest rate.",
            ),
        ]
    )
    agent = AgenticRAG(
        search_tool=_search_tool([_response("first"), _response("second")]),
        chat_model=chat,
        evidence_grader=grader,
        semantic_evidence_grading=True,
        max_retrieval_attempts=2,
    )

    answer = agent.ask("What was the exact interest rate?")

    assert answer.insufficient_evidence is True
    assert "still indirect" in answer.answer
    assert answer.sources == ()
    assert answer.usage.api_calls == 3
    assert chat.generate_calls == 0


def test_semantic_grader_fails_closed_on_unknown_supporting_label() -> None:
    chat = FakeChatModel(
        GroundedDraft(claims=[AnswerClaim(text="Must not run.", citations=["S1"])])
    )
    grader = FakeEvidenceGrader(
        [
            EvidenceGrade(
                sufficient=True,
                supporting_labels=["S99"],
                reason="Incorrect label.",
            )
        ]
    )
    agent = AgenticRAG(
        search_tool=_search_tool([_response("query")]),
        chat_model=chat,
        evidence_grader=grader,
        semantic_evidence_grading=True,
        max_retrieval_attempts=1,
    )

    answer = agent.ask("What policy decision was made?")

    assert answer.insufficient_evidence is True
    assert "unknown supporting labels" in answer.answer
    assert chat.generate_calls == 0


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


def test_per_question_step_budget_bounds_selected_retries() -> None:
    responses = [
        _response("first", confidence="low", results=False),
        _response("unused", confidence="low", results=False),
    ]
    chat = FakeChatModel(GroundedDraft())
    agent = AgenticRAG(
        search_tool=_search_tool(responses),
        chat_model=chat,
        max_retrieval_attempts=5,
        max_agent_steps=17,
    )

    answer = agent.ask(
        "Unanswerable question",
        max_retrieval_attempts=5,
        max_agent_steps=5,
    )

    assert answer.retrieval_attempts == 1
    assert len(answer.graph_trace) == 5
    assert chat.rewrite_calls == 0
    assert len(responses) == 1


def test_retrieval_attempt_capacity_uses_three_steps_per_retry() -> None:
    assert bounded_retrieval_attempts(5, 5) == 1
    assert bounded_retrieval_attempts(5, 8) == 2
    assert bounded_retrieval_attempts(5, 17) == 5


def test_graph_rejects_out_of_scope_question_before_retrieval_or_generation() -> None:
    responses = [_response("unused")]
    chat = FakeChatModel(
        GroundedDraft(claims=[AnswerClaim(text="Should not be generated.", citations=["S1"])])
    )
    agent = AgenticRAG(
        search_tool=_search_tool(responses),
        chat_model=chat,
        scope_description="monetary-policy research",
        scope_terms=("monetary policy", "inflation", "central bank"),
    )

    answer = agent.ask("What day was Donald Trump elected?")

    assert answer.insufficient_evidence is True
    assert "outside the configured scope" in answer.answer
    assert answer.sources == ()
    assert answer.retrieval_attempts == 0
    assert answer.usage.api_calls == 0
    assert responses  # The search tool was never invoked.
    assert chat.generate_calls == 0
    assert [event.node for event in answer.graph_trace] == [
        "analyze_question",
        "validate_answer",
    ]


def test_scope_matching_is_phrase_aware_and_optional() -> None:
    terms = ("monetary policy", "inflation", "fed", "interest rate")

    assert question_matches_scope("How does monetary-policy affect inflation?", terms) is True
    assert question_matches_scope("What did the Fed decide?", terms) is True
    assert question_matches_scope("What is the Fed's current view of the market?", terms) is True
    assert question_matches_scope("What are the current interest rates?", terms) is True
    assert question_matches_scope("Who won the football match?", terms) is False
    assert question_matches_scope("Any reusable corpus question", ()) is True


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
    with pytest.raises(ValueError, match="max_agent_steps"):
        agent.ask("question", max_agent_steps=4)
    with pytest.raises(ValueError, match="requires an evidence grader"):
        agent.ask("question", semantic_evidence_grading=True)

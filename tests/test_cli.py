import json
import sys
from pathlib import Path

import pytest

from advanced_rag.config import get_settings
from advanced_rag.generation.models import (
    AgentAnswer,
    CitationValidation,
    ModelUsage,
)
from advanced_rag.interfaces.cli import main
from advanced_rag.retrieval.hybrid_models import RetrievalDiagnostics
from advanced_rag.retrieval.models import RetrievalFilters


def _configure_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("RAG_DATA_DIR", str(raw))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "indexes"))
    monkeypatch.setenv("RAG_CHROMA_DIR", str(tmp_path / "indexes" / "chroma"))
    monkeypatch.setenv("RAG_CHROMA_COLLECTION", "cli-test")
    monkeypatch.setenv("RAG_BM25_DIR", str(tmp_path / "indexes" / "bm25"))
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "deterministic")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "32")
    monkeypatch.setenv("RAG_RERANKER_PROVIDER", "none")
    get_settings.cache_clear()
    return raw


def test_cli_indexes_inspects_searches_and_resets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = _configure_cli(monkeypatch, tmp_path)
    (raw / "api.txt").write_text("API authentication uses access tokens.", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["rag", "index", str(raw)])
    assert main() == 0
    indexed = json.loads(capsys.readouterr().out)
    assert indexed["upserted_chunks"] == 1

    monkeypatch.setattr(sys, "argv", ["rag", "index-info"])
    assert main() == 0
    info = json.loads(capsys.readouterr().out)
    assert info["count"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["rag", "search-dense", "access token", "--filename", "api.txt"],
    )
    assert main() == 0
    results = json.loads(capsys.readouterr().out)
    assert results[0]["chunk"]["metadata"]["filename"] == "api.txt"

    monkeypatch.setattr(sys, "argv", ["rag", "reset-index"])
    assert main() == 2
    assert "without --yes" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["rag", "reset-index", "--yes"])
    assert main() == 0
    reset = json.loads(capsys.readouterr().out)
    assert reset["count"] == 0
    get_settings.cache_clear()


def test_cli_ingestion_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = _configure_cli(monkeypatch, tmp_path)
    (raw / "notes.md").write_text("# Notes\n\nUseful content.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["rag", "ingest", str(raw)])

    assert main() == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["processed_files"] == 1
    assert summary["chunks"] == 1
    get_settings.cache_clear()


def test_cli_sparse_index_search_info_and_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = _configure_cli(monkeypatch, tmp_path)
    (raw / "paper.txt").write_text(
        "BIS Working Paper 1047 covers monetary transmission.", encoding="utf-8"
    )

    monkeypatch.setattr(sys, "argv", ["rag", "index-sparse", str(raw)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["rebuilt"] is True

    monkeypatch.setattr(sys, "argv", ["rag", "sparse-info"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["rag", "search-sparse", "Working Paper 1047", "--filename", "paper.txt"],
    )
    assert main() == 0
    matches = json.loads(capsys.readouterr().out)
    assert matches[0]["chunk"]["metadata"]["filename"] == "paper.txt"

    monkeypatch.setattr(sys, "argv", ["rag", "reset-sparse"])
    assert main() == 2
    assert "without --yes" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["rag", "reset-sparse", "--yes"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["count"] == 0
    get_settings.cache_clear()


def test_cli_hybrid_search_and_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = _configure_cli(monkeypatch, tmp_path)
    (raw / "inflation.txt").write_text(
        "Demand and supply factors both drove post-pandemic inflation.", encoding="utf-8"
    )

    monkeypatch.setattr(sys, "argv", ["rag", "index", str(raw)])
    assert main() == 0
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["rag", "index-sparse", str(raw)])
    assert main() == 0
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["rag", "search-hybrid", "inflation", "--top-k", "1"])
    assert main() == 0
    response = json.loads(capsys.readouterr().out)
    assert response["results"][0]["chunk"]["metadata"]["filename"] == "inflation.txt"
    assert response["diagnostics"]["reranker"] == "none"

    monkeypatch.setattr(sys, "argv", ["rag", "compare-retrievers", "inflation", "--top-k", "1"])
    assert main() == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["dense"] and comparison["sparse"] and comparison["fused"]
    get_settings.cache_clear()


def test_cli_ask_prints_structured_agent_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure_cli(monkeypatch, tmp_path)
    expected = AgentAnswer(
        question="What drove inflation?",
        answer="Demand contributed. [S1]",
        sources=(),
        insufficient_evidence=False,
        citation_validation=CitationValidation(valid=True, accepted_claims=1, rejected_claims=0),
        retrieval_diagnostics=RetrievalDiagnostics(
            dense_candidates=1,
            sparse_candidates=1,
            fused_candidates=1,
            reranked_candidates=1,
            agreement_count=1,
            distinct_sources=1,
            evidence_tokens=10,
            confidence="medium",
            reranker="test",
            rerank_applied=True,
        ),
        graph_trace=(),
        usage=ModelUsage(api_calls=1, input_tokens=10, output_tokens=5),
        retrieval_attempts=1,
        final_query="What drove inflation?",
    )

    class FakeAgent:
        def ask(
            self,
            question: str,
            *,
            filters: RetrievalFilters | None = None,
            top_k: int | None = None,
        ) -> AgentAnswer:
            del question, filters, top_k
            return expected

    monkeypatch.setattr("advanced_rag.interfaces.cli._agent", lambda settings: FakeAgent())
    monkeypatch.setattr(
        sys,
        "argv",
        ["rag", "ask", "What drove inflation?", "--json"],
    )

    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["answer"] == "Demand contributed. [S1]"
    assert output["usage"]["api_calls"] == 1
    get_settings.cache_clear()

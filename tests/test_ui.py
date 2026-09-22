import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr
from streamlit.testing.v1 import AppTest

from advanced_rag.config import Settings, get_settings
from advanced_rag.interfaces.streamlit_app import _readiness, _request_token_reserve
from advanced_rag.interfaces.ui import launch_streamlit, streamlit_script_path
from advanced_rag.retrieval.models import CollectionInfo, SparseIndexInfo


def test_ui_launcher_uses_active_python_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        captured.extend(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("advanced_rag.interfaces.ui.subprocess.run", fake_run)

    assert launch_streamlit(address="127.0.0.1", port=9000, headless=True) == 0
    assert captured[1:4] == ["-m", "streamlit", "run"]
    assert "--server.port" in captured
    assert "9000" in captured
    assert captured[-1] == "false"


def test_ui_launcher_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        del command, check
        raise KeyboardInterrupt

    monkeypatch.setattr("advanced_rag.interfaces.ui.subprocess.run", interrupted_run)

    assert launch_streamlit(address="127.0.0.1", port=8501, headless=False) == 130


def test_streamlit_shell_reports_empty_indexes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "indexes"))
    monkeypatch.setenv("RAG_CHROMA_DIR", str(tmp_path / "indexes" / "chroma"))
    monkeypatch.setenv("RAG_BM25_DIR", str(tmp_path / "indexes" / "bm25"))
    monkeypatch.setenv("RAG_CHROMA_COLLECTION", "ui-test")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "deterministic")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "32")
    monkeypatch.setenv("RAG_OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()

    app = AppTest.from_file(streamlit_script_path()).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Advanced Agentic RAG"
    assert [metric.value for metric in app.metric[:3]] == ["Configured", "0", "0"]
    assert "Build both indexes" in app.warning[0].value
    assert app.chat_input[0].disabled is True
    assert {slider.label for slider in app.slider} >= {
        "Evidence chunks",
        "Maximum retrieval retries",
        "Maximum agent steps",
    }
    assert "Require semantic evidence grading" in {checkbox.label for checkbox in app.checkbox}
    get_settings.cache_clear()


def test_chat_readiness_requires_key_nonempty_matching_indexes() -> None:
    dense = CollectionInfo(
        name="test",
        count=4,
        path="dense",
        embedding_provider="openai",
        embedding_model="test",
        embedding_dimensions=8,
    )
    sparse = SparseIndexInfo(
        path="sparse",
        count=4,
        method="lucene",
        k1=1.5,
        b=0.75,
        fingerprint="abc",
        schema_version=1,
    )

    ready = _readiness(
        Settings(_env_file=None, openai_api_key=SecretStr("test-key")),
        dense,
        sparse,
    )
    missing_key = _readiness(Settings(_env_file=None), dense, sparse)
    mismatched = _readiness(
        Settings(_env_file=None, openai_api_key=SecretStr("test-key")),
        dense,
        sparse.model_copy(update={"count": 3}),
    )

    assert ready.ready is True
    assert missing_key.ready is False
    assert mismatched.ready is False


def test_semantic_grading_increases_conservative_token_reserve() -> None:
    settings = Settings(_env_file=None)

    local_only = _request_token_reserve(
        settings,
        2,
        semantic_evidence_grading=False,
    )
    semantic = _request_token_reserve(
        settings,
        2,
        semantic_evidence_grading=True,
    )

    assert semantic > local_only

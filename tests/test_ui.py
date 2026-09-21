import subprocess
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from advanced_rag.config import get_settings
from advanced_rag.interfaces.ui import launch_streamlit, streamlit_script_path


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
    get_settings.cache_clear()

    app = AppTest.from_file(streamlit_script_path()).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Advanced Agentic RAG"
    assert [metric.value for metric in app.metric] == ["Configured", "0", "0"]
    assert "Build both indexes" in app.warning[0].value
    get_settings.cache_clear()

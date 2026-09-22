from pathlib import Path

import pytest

from advanced_rag.config import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "advanced-agentic-rag"
    assert settings.environment == "development"
    assert settings.openai_api_key is None
    assert settings.enable_web_search is False
    assert "monetary" in settings.agent_scope_term_list
    assert settings.ui_session_question_limit == 10
    assert settings.ui_session_api_call_limit == 20
    assert settings.ui_session_token_budget == 50000


def test_settings_read_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENVIRONMENT", "test")
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"


def test_ensure_directories_creates_storage_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    index_dir = tmp_path / "indexes"
    settings = Settings(_env_file=None, data_dir=data_dir, index_dir=index_dir)

    settings.ensure_directories()

    assert data_dir.is_dir()
    assert index_dir.is_dir()

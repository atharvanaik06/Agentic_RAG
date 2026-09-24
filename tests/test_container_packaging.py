from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docker_image_is_locked_non_root_and_health_checked() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "USER rag" in dockerfile
    assert "_stcore/health" in dockerfile
    assert 'CMD ["rag", "ui", "--address", "0.0.0.0"' in dockerfile
    assert "COPY . ." not in dockerfile
    assert ".env" not in dockerfile


def test_compose_limits_network_exposure_and_persists_mutable_data() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert '"127.0.0.1:8501:8501"' in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:" in compose and "- ALL" in compose
    for path in ("raw", "indexes", "models"):
        assert f"./data/{path}:/app/data/{path}" in compose
    assert "./reports:/app/reports" in compose


def test_docker_context_excludes_secrets_and_mutable_data() -> None:
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".env", ".git", ".venv", "data", "reports"} <= ignored

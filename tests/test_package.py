import pytest

from advanced_rag import __version__
from advanced_rag.__main__ import main


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_smoke_entry_point(capfd: pytest.CaptureFixture[str]) -> None:
    main()

    captured = capfd.readouterr()
    assert "application_ready" in captured.err

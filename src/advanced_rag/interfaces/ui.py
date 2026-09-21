"""Launcher for the local Streamlit interface."""

import subprocess
import sys
from pathlib import Path


def streamlit_script_path() -> Path:
    """Return the installed Streamlit application entry point."""
    return Path(__file__).with_name("streamlit_app.py")


def launch_streamlit(*, address: str, port: int, headless: bool) -> int:
    """Run Streamlit in the active Python environment without invoking a shell."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_script_path()),
        "--server.address",
        address,
        "--server.port",
        str(port),
        "--server.headless",
        str(headless).lower(),
        "--browser.gatherUsageStats",
        "false",
    ]
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130

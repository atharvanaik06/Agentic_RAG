"""Deterministic source-file discovery."""

from pathlib import Path


def discover_files(source: Path) -> list[Path]:
    """Return a stable list of visible files beneath a path."""
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"Source path must be a file or directory: {source}")

    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(source).parts)
    )


def relative_source_path(path: Path, source: Path) -> str:
    """Return a portable path used as stable source identity."""
    base = source if source.is_dir() else source.parent
    return path.resolve().relative_to(base.resolve()).as_posix()

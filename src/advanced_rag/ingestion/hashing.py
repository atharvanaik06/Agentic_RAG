"""Stable hashing helpers for idempotent ingestion."""

from hashlib import sha256
from pathlib import Path


def hash_file(path: Path) -> str:
    """Hash a source file without loading the whole file into memory."""
    digest = sha256()
    with path.open("rb") as source_file:
        while block := source_file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: object) -> str:
    """Build a stable SHA-256 ID from unambiguous string components."""
    payload = "\x1f".join(str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()

"""Small command-line entry point for inspecting ingestion results."""

import argparse
import json
from pathlib import Path

from advanced_rag.config import get_settings
from advanced_rag.ingestion.models import IngestionResult
from advanced_rag.ingestion.pipeline import IngestionPipeline


def _summary(result: IngestionResult) -> dict[str, object]:
    return {
        "discovered_files": result.discovered_files,
        "processed_files": result.processed_files,
        "skipped_files": result.skipped_files,
        "failed_files": result.failed_files,
        "chunks": len(result.chunks),
        "issues": [issue.model_dump(mode="json") for issue in result.issues],
    }


def main() -> int:
    """Ingest a path and print a JSON summary without dumping document text."""
    parser = argparse.ArgumentParser(description="Inspect Advanced RAG document ingestion")
    parser.add_argument("source", nargs="?", type=Path, help="File or directory to ingest")
    args = parser.parse_args()

    settings = get_settings()
    source = args.source or settings.data_dir
    result = IngestionPipeline(settings).ingest(source)
    print(json.dumps(_summary(result), indent=2))
    return 1 if result.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())

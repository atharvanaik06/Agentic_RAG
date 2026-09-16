"""JSONL loading and validation for portable evaluation benchmarks."""

import json
from pathlib import Path

from pydantic import ValidationError

from advanced_rag.evaluation.models import BenchmarkCase


class BenchmarkError(ValueError):
    """A benchmark file is missing or contains invalid records."""


def load_benchmark(path: Path | str) -> tuple[BenchmarkCase, ...]:
    """Load a non-empty JSONL benchmark and reject duplicate case IDs."""
    benchmark_path = Path(path)
    if not benchmark_path.is_file():
        raise BenchmarkError(f"Benchmark not found: {benchmark_path}")
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        benchmark_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = BenchmarkCase.model_validate_json(line)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise BenchmarkError(
                f"Invalid benchmark record at {benchmark_path}:{line_number}: {exc}"
            ) from exc
        if case.id in seen:
            raise BenchmarkError(f"Duplicate benchmark ID '{case.id}' at line {line_number}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise BenchmarkError(f"Benchmark is empty: {benchmark_path}")
    return tuple(cases)

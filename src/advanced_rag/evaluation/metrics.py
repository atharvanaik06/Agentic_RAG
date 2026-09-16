"""Dependency-free information-retrieval metrics and relevance matching."""

from math import log2
from statistics import fmean

from advanced_rag.evaluation.models import BenchmarkCase, GoldTarget, RankedRecord
from advanced_rag.ingestion.models import DocumentChunk


def matching_target(chunk: DocumentChunk, case: BenchmarkCase) -> GoldTarget | None:
    """Return the first gold target matched by a retrieved canonical chunk."""
    for target in case.gold_targets:
        if chunk.metadata.filename != target.filename:
            continue
        if target.chunk_ids and chunk.chunk_id not in target.chunk_ids:
            continue
        if target.pages and chunk.metadata.page_number not in target.pages:
            continue
        return target
    return None


def retrieval_metrics(
    chunks: list[tuple[DocumentChunk, float | None]],
    case: BenchmarkCase,
    *,
    top_k: int,
) -> tuple[float, float, float, float, float, bool, bool | None, tuple[RankedRecord, ...]]:
    """Calculate binary relevance metrics at k for one ranked list."""
    selected = chunks[:top_k]
    records: list[RankedRecord] = []
    matched_targets: set[int] = set()
    relevant_flags: list[bool] = []
    first_relevant: int | None = None
    precision_sum = 0.0
    relevant_seen = 0
    for rank, (chunk, score) in enumerate(selected, start=1):
        target = matching_target(chunk, case)
        relevant = target is not None
        if relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
            first_relevant = first_relevant or rank
            matched_targets.add(case.gold_targets.index(target))
        relevant_flags.append(relevant)
        records.append(
            RankedRecord(
                rank=rank,
                chunk_id=chunk.chunk_id,
                filename=chunk.metadata.filename,
                page_number=chunk.metadata.page_number,
                relevant=relevant,
                score=score,
            )
        )
    gold_count = len(case.gold_targets)
    recall = len(matched_targets) / gold_count if gold_count else 0.0
    precision = relevant_seen / top_k
    reciprocal_rank = 1.0 / first_relevant if first_relevant else 0.0
    average_precision = precision_sum / gold_count if gold_count else 0.0
    dcg = sum(1.0 / log2(rank + 1) for rank, flag in enumerate(relevant_flags, 1) if flag)
    ideal_count = min(gold_count, top_k)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = min(dcg / ideal_dcg, 1.0) if ideal_dcg else 0.0
    source_hit = any(
        record.filename in {target.filename for target in case.gold_targets} for record in records
    )
    expected_pages = {
        (target.filename, page) for target in case.gold_targets for page in target.pages
    }
    page_hit = (
        any((record.filename, record.page_number) in expected_pages for record in records)
        if expected_pages
        else None
    )
    return (
        recall,
        precision,
        reciprocal_rank,
        min(average_precision, 1.0),
        ndcg,
        source_hit,
        page_hit,
        tuple(records),
    )


def mean(values: list[float]) -> float:
    """Return a safe arithmetic mean for report aggregation."""
    return fmean(values) if values else 0.0

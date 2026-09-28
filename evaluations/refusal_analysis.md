# Incorrect-refusal analysis

This analysis records the diagnosis of the 33-case `monetary_policy.jsonl`
evaluation run performed on 28 September 2026. It compares the full agent report
with the final reranked hybrid results at five chunks. Generated reports remain
local under `reports/`; this document preserves the actionable conclusions.

## Outcome

- Answerable cases: 29
- Answerable cases that received an answer: 11 (37.9%)
- Incorrect refusals: 18 (62.1%)
- Incorrect refusals where the expected source was in the final top five: 16
- Incorrect refusals where the expected source was absent: 2 (`mp-024`, `mp-033`)

The expected-source comparison is filename-level when a benchmark target does
not specify pages or chunks. It proves that the correct document was selected,
not that the selected passage necessarily contained the requested answer.

## Failure categories

| Cause | Cases | Interpretation |
| --- | --- | --- |
| Invalid semantic-grader labels | `mp-002`, `mp-006`, `mp-008`, `mp-009`, `mp-010`, `mp-012`, `mp-013`, `mp-015`, `mp-016`, `mp-021`, `mp-022`, `mp-023`, `mp-026` | The expected source was retrieved, but the grader returned concepts, headings, or filenames instead of labels such as `S1`. Local validation correctly failed closed. |
| Evidence judged insufficient after retrieval | `mp-011`, `mp-014`, `mp-027` | The expected document was present, but the selected passages did not directly establish the benchmark answer. These require passage-level retrieval and benchmark review. |
| Expected source not retrieved | `mp-024` | This is a retrieval failure: the January 2026 FOMC source was absent from the final top five. |
| Ambiguous success/refusal state and incomplete gold target | `mp-033` | The agent produced the correct federal-funds range from another Federal Reserve document but also marked the response insufficient; the benchmark only names the July 2026 minutes as acceptable evidence. |

## Initial conclusion

The dominant problem was **not general evidence-grader strictness and not broad
document retrieval failure**. Thirteen of eighteen incorrect refusals came from
the semantic grader violating the supporting-label contract even though the
expected source was retrieved.

The remaining cases are mixed:

1. Improve passage selection for broad, dated, and cross-document questions.
2. Review whether `mp-011` is directly supported by Working Paper 1167.
3. Expand acceptable gold targets for `mp-033` or narrow its wording.
4. Decide whether a draft may contain claims when `insufficient_evidence=true`.

## Contract repair and validation

The grader contract now returns positive integer evidence positions such as
`[1, 3]` instead of free-form strings such as `S1`. The OpenAI structured-output
schema enforces integer values, and the graph verifies that each position exists
before permitting generation. A two-case smoke test confirmed that `mp-002`,
which previously failed twice with topical labels, answered on its first attempt.

A complete 33-case rerun produced the following changes:

| Metric | Before | After |
| --- | ---: | ---: |
| Answerable success | 37.9% | 89.7% |
| Incorrect refusal | 62.1% | 10.3% |
| Expected-source hit | 31.0% | 79.3% |
| Rewrite rate | 66.7% | 21.2% |
| Average retrieval attempts | 1.636 | 1.182 |
| Average chat calls | 2.667 | 2.182 |

Citation validity and unanswerable-case refusal accuracy remained 100%. All
configured regression gates passed after the repair.

Three answerable cases remain refused:

- `mp-014`: the expected BIS document was retrieved, but the selected passages
  did not directly explain cross-country inflation-expectation connections.
- `mp-024`: the January 2026 FOMC source was absent from the final evidence.
- `mp-027`: both ECB issues were represented, but the passages did not support a
  direct comparison between their assessments.

These remaining failures should be addressed through passage selection,
date/document filtering, and benchmark specificity rather than by weakening the
evidence sufficiency standard.

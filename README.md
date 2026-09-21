# Advanced Agentic RAG

A controlled agentic retrieval-augmented generation project using LangGraph,
ChromaDB, hybrid dense/BM25 retrieval, and reranking.

The repository has completed **Phase 7: evaluation and regression testing**, and
the Phase 8 Streamlit interface is in progress. It can
discover, load, normalize, and chunk PDF, Markdown, and TXT sources; retrieve
from persistent ChromaDB and BM25 indexes in parallel; fuse rankings; rerank
locally; conditionally rewrite weak searches; generate grounded answers; and
validate every emitted citation. It also includes retrieval, reranker, and
end-to-end agent benchmarks with machine-readable results and quality gates.

## Local web interface

Launch the Phase 8 Streamlit application shell with:

```bash
uv run --no-editable rag ui
```

The interface verifies that the API key is configured, opens the local Chroma
and BM25 indexes, compares their chunk counts, and displays the active models
without exposing secrets. When the system is ready, the chat view runs the
existing six-node LangGraph workflow, preserves conversation results for the
current browser session, and displays validated answers with compact source and
usage summaries.

Use a different port or suppress automatic browser opening when needed:

```bash
uv run --no-editable rag ui --port 8502 --headless
```

Detailed source inspection, graph traces, document management, and the evaluation
dashboard are added in the remaining Phase 8 milestones.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended)

## Setup

```bash
uv sync --extra dev
cp .env.example .env
uv run python -m advanced_rag
```

If `uv` is not available, use a standard virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
.venv/bin/python -m advanced_rag
```

The smoke command prints the application name, version, and active
environment without exposing secrets.

For a first local run, add your own `RAG_OPENAI_API_KEY` to `.env`, copy
documents into `data/raw`, and build both indexes:

```bash
uv run rag index data/raw
uv run rag index-sparse data/raw
uv run rag ask "What does the corpus say about inflation?"
```

## Inspect document ingestion

Place source documents under `data/raw`, then run:

```bash
uv run rag ingest data/raw
```

The command prints a JSON summary with file and chunk counts plus structured
issues. It deliberately does not print document contents. A non-zero exit code
means at least one supported document failed; unsupported and duplicate files
are reported as skips.

The programmatic interface returns immutable, typed `DocumentChunk` records:

```python
from advanced_rag.ingestion import IngestionPipeline

result = IngestionPipeline().ingest("data/raw")
for chunk in result.chunks:
    print(chunk.chunk_id, chunk.metadata.source_path)
```

PDF chunks preserve 1-based page numbers, Markdown chunks preserve heading
paths, and every source and chunk receives a deterministic identifier. PDFs
without extractable text are reported as image-only and require a future OCR
extension.

## Dense indexing and search

Set `RAG_OPENAI_API_KEY` in `.env`, place source documents under `data/raw`, and
synchronize the collection:

```bash
uv run rag index data/raw
uv run rag index-info
uv run rag search-dense "How does token authentication work?" --top-k 6
```

Dense indexing is idempotent. Unchanged sources are not embedded again, edited
sources replace their stale chunks, and files deleted from a directory scan are
removed from the collection. Use `--no-prune` when intentionally indexing only
a subset of a collection.

Exact-match search filters are available for source ID, filename, file type,
and page number:

```bash
uv run rag search-dense "access token" --filename api-guide.pdf --page 17
```

For offline development and tests, set `RAG_EMBEDDING_PROVIDER=deterministic`
and choose a small `RAG_EMBEDDING_DIMENSIONS` value. Deterministic embeddings
verify plumbing only and must not be used to judge semantic retrieval quality.

Resetting a collection is intentionally guarded:

```bash
uv run rag reset-index --yes
```

## Sparse indexing and search

BM25 uses the same canonical chunks as dense retrieval, but runs entirely
locally: it does not call OpenAI and does not create additional embedding cost.
Build it after adding or removing documents:

```bash
uv run rag index-sparse data/raw
uv run rag sparse-info
uv run rag search-sparse "BIS Working Paper 1047" --top-k 6
```

The persisted index is stored under `RAG_BM25_DIR`. Documents are lowercased,
English stop words are removed, and tokens are English-stemmed; identifiers and
numbers such as `BIS`, `1047`, and policy-rate values remain searchable. A stable
corpus fingerprint makes an unchanged rebuild a no-op. Any changed, added, or
removed source triggers a complete local rebuild so the BM25 document statistics
remain correct.

Sparse search supports the same exact filters as dense search:

```bash
uv run rag search-sparse "forward guidance" --file-type pdf --page 17
```

Resetting the sparse index is also guarded:

```bash
uv run rag reset-sparse --yes
```

## Hybrid retrieval and reranking

Hybrid search collects candidates from Chroma and BM25 concurrently, then uses
Reciprocal Rank Fusion (RRF) to combine their rank positions without comparing
incompatible raw scores. The best fused candidates are reranked by FlashRank's
local cross-encoder and selected under token, per-source, and per-page limits.

```bash
uv run rag search-hybrid \
  "What demand and supply factors drove post-pandemic inflation?" \
  --top-k 6
```

The first FlashRank search downloads the configured local model into
`data/models/flashrank`. The default TinyBERT model is approximately 4 MB.
Reranking does not call OpenAI; hybrid search makes one embedding API call for
the dense query when the OpenAI embedding provider is active.

Inspect how the individual retrieval methods contributed to a ranking:

```bash
uv run rag compare-retrievers "BIS Working Paper 1047" --top-k 6
```

Hybrid results retain dense rank and score, sparse rank and score, fused RRF
score, reranker score, complete chunk metadata, and diagnostics. Diagnostics
include retrieval agreement, distinct source count, context tokens, degraded
backend warnings, and a routing-oriented confidence level.

The agent-facing tool can be created without coupling it to a particular graph:

```python
from advanced_rag.tools import create_search_knowledge_base_tool

search_tool = create_search_knowledge_base_tool(hybrid_retriever)
```

It is a LangChain-compatible structured tool named `search_knowledge_base` and
is invoked by the Phase 6 LangGraph workflow.

## Agentic question answering

The graph has six explicit nodes:

```text
analyze_question -> retrieve_evidence -> grade_evidence
                                           | sufficient
                                           v
rewrite_query <- weak evidence       generate_answer -> validate_answer
      |                                      ^
      +---------- retrieve again ------------+
```

Run a human-readable answer:

```bash
uv run rag ask "What drove the post-pandemic inflation surge?"
```

Return the complete graph trace, retrieval diagnostics, citation validation,
and token usage as JSON:

```bash
uv run rag ask "What drove inflation?" --json
```

Question analysis, retrieval grading, routing, citation validation, and final
rendering run locally. The graph makes one chat-model call for answer generation.
It makes one additional call only when weak evidence triggers query rewriting.
`RAG_AGENT_MAX_RETRIEVAL_ATTEMPTS` is bounded from 1 to 5 and defaults to 2.
Web search is not part of this graph and remains disabled by default.

Answers are generated as structured claims whose `S1`, `S2`, etc. references
are validated against the retrieved chunk set. Claims with missing or unknown
citations are removed before the answer is returned.

## Evaluation and regression testing

The included monetary-policy benchmark contains 31 questions: 27 answerable
questions with source/page targets and 4 deliberately unanswerable or
out-of-domain questions.
Run the retrieval evaluation after building both indexes:

```bash
uv run rag evaluate-retrieval evaluations/monetary_policy.jsonl --top-k 5
```

This compares dense search, BM25, fused rankings before reranking, and the final
reranked hybrid retriever. It records Recall@k, Precision@k, MRR, MAP, NDCG,
source/page hit rates, latency, individual failures, and rank movement caused by
the reranker.

Evaluate the complete six-node agent separately because every answerable case
uses the configured chat API:

```bash
# Start with a small, inexpensive smoke run.
uv run rag evaluate-agent evaluations/monetary_policy.jsonl --limit 3

# Run the complete benchmark when ready.
uv run rag evaluate-agent evaluations/monetary_policy.jsonl
```

Agent evaluation checks expected concepts and sources, refusal behavior,
citation validity against stored chunk IDs, rewrites, retrieval attempts, chat
calls, tokens, and latency. These deterministic checks are the default. An
optional model-based entailment check can inspect whether each claim is
supported by its cited evidence, but adds one API call per evaluated claim:

```bash
uv run rag evaluate-agent evaluations/monetary_policy.jsonl --judge-entailment
```

Generate a GitHub-friendly summary from whichever result files are present:

```bash
uv run rag evaluate-report
```

Reports are written under `reports/` and ignored by Git because generated
answers may reflect private corpora. Add `--enforce` to any evaluation command
to return a failing exit code when its configured regression gate fails. The
default gates cover hybrid recall, citation validity, refusal accuracy, and
average retrieval attempts and can be adjusted with the `RAG_EVALUATION_*`
settings in `.env`.

The first 27-case retrieval baseline on the development corpus achieved 0.870
Recall@5 and a 0.926 source-hit rate for the final reranked hybrid method. The
reranker improved one target rank, left 19 unchanged, worsened three, and did
not recover four. That mixed result is retained intentionally: Phase 7 makes
retrieval changes measurable instead of assuming that reranking always helps.

The checked-in questions are a seed benchmark for the named public documents,
not universal ground truth. If the corpus changes, copy
`evaluations/template.jsonl`, then review and version targets with the corpus.

## Local data and hosted APIs

The program, LangGraph execution, Chroma database, BM25 index, FlashRank model,
documents, traces, and rendered answers live on the user's machine. With the
default OpenAI configuration, document text is sent when creating embeddings,
queries are sent for dense query embeddings, and selected evidence is sent for
answer generation. Each downloader supplies their own API key. Generation uses
the Responses API with response storage disabled for the request.

`.env`, `data/raw`, generated indexes, and downloaded models are ignored by Git.
Do not commit personal documents, generated indexes, or API credentials.

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

Replace `uv run` with `.venv/bin/` when using the fallback setup.

## Repository layout

```text
data/
├── raw/                    # Source documents (not committed)
├── indexes/                # Generated indexes (not committed)
└── models/                 # Downloaded local reranker models (not committed)
evaluations/                # Versioned benchmark questions and gold targets
reports/                    # Generated evaluation outputs (not committed)
src/advanced_rag/
├── ingestion/              # Document loading and chunking
├── retrieval/              # Dense, sparse, fusion, and reranking
├── tools/                  # Typed tools exposed to the agent
├── graph/                  # LangGraph state, nodes, and routing
├── generation/             # Answer and citation generation
├── evaluation/             # Retrieval and answer evaluations
└── interfaces/             # CLI and Streamlit entry points
tests/                      # Unit and integration tests
```

## Configuration

Runtime settings use the `RAG_` prefix and can be supplied through environment
variables or a local `.env` file. Nested deployments can instantiate the
settings object directly instead of relying on process-global configuration.
See `.env.example` for the supported foundation settings.

Chunk size, overlap, minimum remainder size, and token encoding are controlled
with `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_MIN_CHUNK_SIZE`, and
`RAG_TOKEN_ENCODING`.

Dense-index configuration uses `RAG_CHROMA_DIR`, `RAG_CHROMA_COLLECTION`,
`RAG_EMBEDDING_PROVIDER`, `RAG_EMBEDDING_MODEL`,
`RAG_EMBEDDING_DIMENSIONS`, and `RAG_EMBEDDING_BATCH_SIZE`.

Sparse-index configuration uses `RAG_BM25_DIR`, `RAG_BM25_METHOD`,
`RAG_BM25_K1`, and `RAG_BM25_B`.

Hybrid configuration uses `RAG_HYBRID_CANDIDATE_K`, `RAG_HYBRID_RERANK_K`,
`RAG_HYBRID_RRF_K`, `RAG_HYBRID_CONTEXT_TOKEN_BUDGET`,
`RAG_HYBRID_MAX_CHUNKS_PER_SOURCE`, and `RAG_HYBRID_MAX_CHUNKS_PER_PAGE`.
The local reranker is configured with `RAG_RERANKER_PROVIDER`,
`RAG_RERANKER_MODEL`, `RAG_RERANKER_CACHE_DIR`, and
`RAG_RERANKER_MAX_LENGTH`. Set `RAG_RERANKER_PROVIDER=none` for deterministic
plumbing tests or to disable reranking explicitly.

Agent configuration uses `RAG_CHAT_PROVIDER`, `RAG_CHAT_MODEL`,
`RAG_CHAT_MAX_OUTPUT_TOKENS`, `RAG_AGENT_MAX_RETRIEVAL_ATTEMPTS`, and
`RAG_AGENT_TOP_K`. `RAG_AGENT_SCOPE_DESCRIPTION` and
`RAG_AGENT_SCOPE_TERMS` configure the inexpensive local domain gate; customize
them when using a different corpus. The current supported chat provider is
OpenAI; the graph depends on a small provider protocol so additional providers
can be added later.

Evaluation configuration uses `RAG_EVALUATION_DIR`,
`RAG_EVALUATION_ENTAILMENT_MODEL`,
`RAG_EVALUATION_HYBRID_RECALL_THRESHOLD`,
`RAG_EVALUATION_CITATION_VALIDITY_THRESHOLD`,
`RAG_EVALUATION_REFUSAL_ACCURACY_THRESHOLD`, and
`RAG_EVALUATION_MAX_AVERAGE_ATTEMPTS`.

Secrets are represented as Pydantic `SecretStr` values and are excluded from
the public settings summary.

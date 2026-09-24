# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS builder

ARG UV_VERSION=0.12.13

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-bookworm AS runtime

ARG APP_UID=1000
ARG APP_GID=1000

ENV HOME=/home/rag \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_ENVIRONMENT=production \
    RAG_DATA_DIR=/app/data/raw \
    RAG_INDEX_DIR=/app/data/indexes \
    RAG_CHROMA_DIR=/app/data/indexes/chroma \
    RAG_BM25_DIR=/app/data/indexes/bm25 \
    RAG_RERANKER_CACHE_DIR=/app/data/models/flashrank \
    RAG_EVALUATION_BENCHMARK_DIR=/app/evaluations \
    RAG_EVALUATION_DIR=/app/reports

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" rag \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home rag \
    && mkdir -p /app/data/raw /app/data/indexes /app/data/models /app/reports \
    && chown -R rag:rag /app /home/rag

WORKDIR /app

COPY --from=builder --chown=rag:rag /app/.venv /app/.venv
COPY --chown=rag:rag evaluations ./evaluations

USER rag

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read()"]

CMD ["rag", "ui", "--address", "0.0.0.0", "--port", "8501", "--headless"]

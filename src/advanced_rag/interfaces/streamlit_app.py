"""Local Streamlit application shell for Advanced Agentic RAG."""

from dataclasses import dataclass

import streamlit as st

from advanced_rag.config import Settings, get_settings
from advanced_rag.retrieval.dense import ChromaDenseIndex
from advanced_rag.retrieval.models import CollectionInfo, SparseIndexInfo
from advanced_rag.retrieval.sparse import BM25SparseIndex
from advanced_rag.runtime import create_dense_index, create_sparse_index


@dataclass(frozen=True)
class AppRuntime:
    """Long-lived local resources reused across Streamlit reruns."""

    settings: Settings
    dense: ChromaDenseIndex
    sparse: BM25SparseIndex


@st.cache_resource(show_spinner="Opening local indexes...")
def load_runtime() -> AppRuntime:
    """Initialize local indexes once for the Streamlit server process."""
    settings = get_settings()
    settings.ensure_directories()
    return AppRuntime(
        settings=settings,
        dense=create_dense_index(settings),
        sparse=create_sparse_index(settings),
    )


def _render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.header("Configuration")
        st.caption("Secrets are loaded from `.env` and are never displayed.")
        st.write(f"**Collection:** `{settings.chroma_collection}`")
        st.write(f"**Embedding:** `{settings.embedding_model}`")
        st.write(f"**Chat model:** `{settings.chat_model}`")
        st.write(f"**Reranker:** `{settings.reranker_model}`")
        st.write(f"**Maximum attempts:** `{settings.agent_max_retrieval_attempts}`")


def _render_readiness(
    settings: Settings,
    dense: CollectionInfo,
    sparse: SparseIndexInfo,
) -> None:
    st.subheader("System readiness")
    api_ready = settings.embedding_provider != "openai" or settings.openai_api_key is not None
    counts_match = dense.count == sparse.count
    corpus_ready = dense.count > 0 and sparse.count > 0

    api_column, dense_column, sparse_column = st.columns(3)
    api_column.metric("OpenAI API key", "Configured" if api_ready else "Missing")
    dense_column.metric("Dense chunks", dense.count)
    sparse_column.metric("BM25 chunks", sparse.count)

    if not api_ready:
        st.error("Add `RAG_OPENAI_API_KEY` to `.env`, then restart the app.")
    elif not corpus_ready:
        st.warning("Build both indexes before asking questions.")
        st.code("uv run rag index data/raw\nuv run rag index-sparse data/raw", language="bash")
    elif not counts_match:
        st.warning("Dense and BM25 chunk counts differ. Rebuild both indexes from the same corpus.")
    else:
        st.success(f"Ready: both retrievers contain {dense.count:,} chunks.")


def main() -> None:
    """Render the Phase 8 application shell."""
    st.set_page_config(
        page_title="Advanced Agentic RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Advanced Agentic RAG")
    st.caption("Local-first, citation-grounded research across your document collection.")

    try:
        runtime = load_runtime()
        dense_info = runtime.dense.info()
        sparse_info = runtime.sparse.info()
    except Exception as exc:
        st.error("The local RAG runtime could not be initialized.")
        st.exception(exc)
        return

    _render_sidebar(runtime.settings)
    _render_readiness(runtime.settings, dense_info, sparse_info)

    st.divider()
    st.subheader("Ask your corpus")
    st.info(
        "The application foundation is ready. Interactive questions, grounded answers, "
        "source cards, and the LangGraph trace arrive in the next Phase 8 milestone."
    )


main()

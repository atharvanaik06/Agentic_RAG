"""Local Streamlit application shell for Advanced Agentic RAG."""

from dataclasses import dataclass
from typing import cast

import streamlit as st

from advanced_rag.config import Settings, get_settings
from advanced_rag.generation.models import AgentAnswer
from advanced_rag.graph import AgenticRAG
from advanced_rag.retrieval.dense import ChromaDenseIndex
from advanced_rag.retrieval.models import CollectionInfo, SparseIndexInfo
from advanced_rag.retrieval.sparse import BM25SparseIndex
from advanced_rag.runtime import create_agent, create_dense_index, create_sparse_index


@dataclass(frozen=True)
class AppRuntime:
    """Long-lived local resources reused across Streamlit reruns."""

    settings: Settings
    dense: ChromaDenseIndex
    sparse: BM25SparseIndex


@dataclass(frozen=True)
class AppReadiness:
    """Conditions required before the interactive agent can run."""

    api_ready: bool
    corpus_ready: bool
    counts_match: bool

    @property
    def ready(self) -> bool:
        return self.api_ready and self.corpus_ready and self.counts_match


@dataclass(frozen=True)
class ChatTurn:
    """One question and its complete, inspectable agent result."""

    question: str
    result: AgentAnswer


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


@st.cache_resource(show_spinner="Loading the hybrid retriever and LangGraph agent...")
def load_agent() -> AgenticRAG:
    """Initialize the agent lazily and reuse it across Streamlit reruns."""
    runtime = load_runtime()
    return create_agent(
        runtime.settings,
        dense=runtime.dense,
        sparse=runtime.sparse,
    )


def _chat_turns() -> list[ChatTurn]:
    turns = st.session_state.get("chat_turns")
    if turns is None:
        turns = []
        st.session_state["chat_turns"] = turns
    return cast(list[ChatTurn], turns)


def _api_key_configured(settings: Settings) -> bool:
    return bool(settings.openai_api_key and settings.openai_api_key.get_secret_value().strip())


def _readiness(
    settings: Settings,
    dense: CollectionInfo,
    sparse: SparseIndexInfo,
) -> AppReadiness:
    return AppReadiness(
        api_ready=_api_key_configured(settings),
        corpus_ready=dense.count > 0 and sparse.count > 0,
        counts_match=dense.count == sparse.count,
    )


def _render_sidebar(settings: Settings) -> int:
    with st.sidebar:
        st.header("Configuration")
        st.caption("Secrets are loaded from `.env` and are never displayed.")
        st.write(f"**Collection:** `{settings.chroma_collection}`")
        st.write(f"**Embedding:** `{settings.embedding_model}`")
        st.write(f"**Chat model:** `{settings.chat_model}`")
        st.write(f"**Reranker:** `{settings.reranker_model}`")
        st.write(f"**Maximum attempts:** `{settings.agent_max_retrieval_attempts}`")
        top_k = st.slider(
            "Evidence chunks",
            min_value=1,
            max_value=10,
            value=settings.agent_top_k,
            help="Maximum number of reranked chunks supplied to answer generation.",
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_turns"] = []
            st.rerun()
        return top_k


def _render_readiness(
    settings: Settings,
    dense: CollectionInfo,
    sparse: SparseIndexInfo,
) -> AppReadiness:
    st.subheader("System readiness")
    readiness = _readiness(settings, dense, sparse)

    api_column, dense_column, sparse_column = st.columns(3)
    api_column.metric("OpenAI API key", "Configured" if readiness.api_ready else "Missing")
    dense_column.metric("Dense chunks", dense.count)
    sparse_column.metric("BM25 chunks", sparse.count)

    if not readiness.api_ready:
        st.error("Add `RAG_OPENAI_API_KEY` to `.env`, then restart the app.")
    elif not readiness.corpus_ready:
        st.warning("Build both indexes before asking questions.")
        st.code(
            "uv run --no-editable rag index data/raw\n"
            "uv run --no-editable rag index-sparse data/raw",
            language="bash",
        )
    elif not readiness.counts_match:
        st.warning("Dense and BM25 chunk counts differ. Rebuild both indexes from the same corpus.")
    else:
        st.success(f"Ready: both retrievers contain {dense.count:,} chunks.")
    return readiness


def _render_answer(result: AgentAnswer) -> None:
    if result.insufficient_evidence:
        st.warning(result.answer)
    else:
        st.markdown(result.answer)

    if result.sources:
        with st.expander(f"Sources ({len(result.sources)})"):
            for source in result.sources:
                page = f", page {source.page_number}" if source.page_number else ""
                title = f" — {source.title}" if source.title else ""
                st.markdown(f"**[{source.label}] {source.filename}**{title}{page}")

    total_tokens = result.usage.input_tokens + result.usage.output_tokens
    st.caption(
        f"{result.retrieval_attempts} retrieval attempt(s) · "
        f"{result.retrieval_diagnostics.distinct_sources} source(s) · "
        f"confidence: {result.retrieval_diagnostics.confidence} · "
        f"{result.usage.api_calls} API call(s) · {total_tokens:,} tokens"
    )
    if result.errors:
        st.warning("Run warnings: " + " ".join(result.errors))


def _render_history(turns: list[ChatTurn]) -> None:
    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            _render_answer(turn.result)


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

    top_k = _render_sidebar(runtime.settings)
    readiness = _render_readiness(runtime.settings, dense_info, sparse_info)

    st.divider()
    st.subheader("Ask your corpus")
    turns = _chat_turns()
    if not turns:
        st.info(
            "Ask about the indexed papers. Answers are generated from reranked local evidence "
            "and citations are validated before display."
        )
    _render_history(turns)

    question = st.chat_input(
        "Ask a question about your document collection",
        disabled=not readiness.ready,
    )
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Running the six-node LangGraph workflow..."):
                result = load_agent().ask(question, top_k=top_k)
        except Exception as exc:
            st.error("The agent could not complete this question.")
            st.exception(exc)
            return
        _render_answer(result)
    turns.append(ChatTurn(question=question, result=result))


if __name__ == "__main__":
    main()

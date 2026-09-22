"""Local Streamlit application shell for Advanced Agentic RAG."""

from dataclasses import dataclass
from typing import cast

import streamlit as st

from advanced_rag.config import Settings, get_settings
from advanced_rag.generation.models import AgentAnswer
from advanced_rag.graph import AgenticRAG, bounded_retrieval_attempts
from advanced_rag.interfaces.documents import (
    CorpusBuildError,
    CorpusBuildResult,
    CorpusInspection,
    UploadPayload,
    delete_corpus_files,
    inspect_corpus,
    list_corpus_files,
    save_uploads,
    synchronize_corpus,
)
from advanced_rag.interfaces.spending import (
    SessionUsage,
    SpendingDecision,
    SpendingLimits,
    spending_decision,
)
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
    corpus_current: bool = True

    @property
    def ready(self) -> bool:
        return self.api_ready and self.corpus_ready and self.counts_match and self.corpus_current


@dataclass(frozen=True)
class ChatTurn:
    """One question and its complete, inspectable agent result."""

    question: str
    result: AgentAnswer


@dataclass(frozen=True)
class SidebarOptions:
    """Per-session controls selected in the sidebar."""

    top_k: int
    max_retrieval_attempts: int
    max_agent_steps: int
    limits: SpendingLimits


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


def _session_usage() -> SessionUsage:
    usage = st.session_state.get("session_usage")
    if not isinstance(usage, SessionUsage):
        usage = SessionUsage()
        st.session_state["session_usage"] = usage
    return usage


def _api_key_configured(settings: Settings) -> bool:
    return bool(settings.openai_api_key and settings.openai_api_key.get_secret_value().strip())


def _readiness(
    settings: Settings,
    dense: CollectionInfo,
    sparse: SparseIndexInfo,
    *,
    corpus_current: bool = True,
) -> AppReadiness:
    return AppReadiness(
        api_ready=_api_key_configured(settings),
        corpus_ready=dense.count > 0 and sparse.count > 0,
        counts_match=dense.count == sparse.count,
        corpus_current=corpus_current,
    )


def _render_sidebar(settings: Settings, usage: SessionUsage) -> SidebarOptions:
    with st.sidebar:
        st.header("Configuration")
        st.caption("Secrets are loaded from `.env` and are never displayed.")
        st.write(f"**Collection:** `{settings.chroma_collection}`")
        st.write(f"**Embedding:** `{settings.embedding_model}`")
        st.write(f"**Chat model:** `{settings.chat_model}`")
        st.write(f"**Reranker:** `{settings.reranker_model}`")
        st.write(f"**Question scope:** {settings.agent_scope_description}")
        top_k = st.slider(
            "Evidence chunks",
            min_value=1,
            max_value=10,
            value=settings.agent_top_k,
            help="Maximum number of reranked chunks supplied to answer generation.",
        )

        with st.expander("Agent execution controls", expanded=True):
            retry_count = st.slider(
                "Maximum retrieval retries",
                min_value=0,
                max_value=4,
                value=settings.agent_max_retrieval_attempts - 1,
                help="A retry rewrites a weak query and searches the local indexes again.",
            )
            max_agent_steps = st.slider(
                "Maximum agent steps",
                min_value=5,
                max_value=17,
                value=min(17, settings.agent_max_steps),
                help=(
                    "The shortest full answer path uses 5 nodes. Each permitted retry needs "
                    "3 additional nodes."
                ),
            )
            requested_attempts = retry_count + 1
            effective_attempts = bounded_retrieval_attempts(
                requested_attempts,
                max_agent_steps,
            )
            effective_retries = effective_attempts - 1
            st.caption(
                f"Effective limit: {effective_attempts} retrieval attempt(s), "
                f"including {effective_retries} retr{'y' if effective_retries == 1 else 'ies'}."
            )
            if effective_attempts < requested_attempts:
                st.warning("The agent-step budget is too small to permit every selected retry.")

        with st.expander("Spending controls", expanded=True):
            st.caption(
                "Session-local safeguards. Set a value to 0 to disable that limit. "
                "These do not change provider account limits."
            )
            question_limit = int(
                st.number_input(
                    "Maximum questions",
                    min_value=0,
                    max_value=1000,
                    value=settings.ui_session_question_limit,
                    step=1,
                )
            )
            api_call_limit = int(
                st.number_input(
                    "Maximum chat API calls",
                    min_value=0,
                    max_value=10000,
                    value=settings.ui_session_api_call_limit,
                    step=1,
                )
            )
            token_budget = int(
                st.number_input(
                    "Token budget (0 = unlimited)",
                    min_value=0,
                    max_value=10000000,
                    value=settings.ui_session_token_budget,
                    step=1000,
                )
            )
            st.markdown(
                f"Used: **{usage.questions}** questions · **{usage.api_calls}** chat API calls · "
                f"**{usage.tokens:,}** chat tokens"
            )
            if st.button("Reset spending counters", use_container_width=True):
                st.session_state["session_usage"] = SessionUsage()
                st.rerun()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_turns"] = []
            st.rerun()
        return SidebarOptions(
            top_k=top_k,
            max_retrieval_attempts=effective_attempts,
            max_agent_steps=max_agent_steps,
            limits=SpendingLimits(
                questions=question_limit,
                api_calls=api_call_limit,
                tokens=token_budget,
            ),
        )


def _render_readiness(
    settings: Settings,
    dense: CollectionInfo,
    sparse: SparseIndexInfo,
    *,
    corpus_current: bool,
) -> AppReadiness:
    st.subheader("System readiness")
    readiness = _readiness(settings, dense, sparse, corpus_current=corpus_current)

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
    elif not readiness.corpus_current:
        st.warning("Local documents changed in this session. Rebuild both indexes before chatting.")
    else:
        st.success(f"Ready: both retrievers contain {dense.count:,} chunks.")
    return readiness


def _render_document_management(
    runtime: AppRuntime,
    dense: CollectionInfo,
    sparse: SparseIndexInfo,
) -> None:
    """Render upload, inspection, deletion, and synchronized-index controls."""
    settings = runtime.settings
    corpus_files = list_corpus_files(settings.data_dir)
    with st.expander("Documents & indexes", expanded=not corpus_files):
        notice = st.session_state.pop("document_notice", None)
        if isinstance(notice, tuple) and len(notice) == 2:
            level, message = notice
            getattr(st, level, st.info)(message)

        st.caption(
            f"Corpus directory: `{settings.data_dir}`. Uploading only saves files locally; "
            "indexing is a separate action. PDF, Markdown, and TXT are supported."
        )
        document_column, dense_column, sparse_column = st.columns(3)
        document_column.metric("Local files", len(corpus_files))
        dense_column.metric("Dense chunks", dense.count)
        sparse_column.metric("BM25 chunks", sparse.count)

        uploaded = st.file_uploader(
            "Add documents",
            type=["pdf", "md", "markdown", "txt"],
            accept_multiple_files=True,
            help=f"Each file may be up to {settings.ui_max_upload_mb} MB.",
        )
        overwrite = st.checkbox(
            "Replace local files that have the same name",
            help="This changes the source file only. Rebuild the indexes afterward.",
        )
        if st.button("Save uploaded files", disabled=not uploaded):
            result = save_uploads(
                settings.data_dir,
                [UploadPayload(name=file.name, content=file.getvalue()) for file in uploaded],
                overwrite=overwrite,
                maximum_bytes=settings.ui_max_upload_mb * 1024 * 1024,
            )
            if result.changed:
                st.session_state["corpus_dirty"] = True
                st.session_state.pop("corpus_inspection", None)
                st.session_state.pop("corpus_build", None)
            messages: list[str] = []
            if result.saved:
                messages.append(f"Saved {len(result.saved)} new file(s).")
            if result.replaced:
                messages.append(f"Replaced {len(result.replaced)} file(s).")
            if result.skipped:
                messages.append(f"Skipped {len(result.skipped)} existing file(s).")
            if result.rejected:
                messages.append("Rejected: " + " ".join(result.rejected))
            st.session_state["document_notice"] = (
                "success" if result.changed and not result.rejected else "warning",
                " ".join(messages) or "No files were saved.",
            )
            st.rerun()

        if corpus_files:
            st.dataframe(
                [
                    {
                        "File": item.relative_path,
                        "Type": item.extension.removeprefix(".") or "unknown",
                        "Size": _format_bytes(item.size_bytes),
                        "Supported": item.supported,
                    }
                    for item in corpus_files
                ],
                use_container_width=True,
                hide_index=True,
            )

            selected = st.multiselect(
                "Select local files to delete",
                options=[item.relative_path for item in corpus_files],
            )
            confirm_delete = st.checkbox(
                "I understand that selected source files will be permanently deleted",
                disabled=not selected,
            )
            if st.button(
                "Delete selected files",
                disabled=not selected or not confirm_delete,
                type="secondary",
            ):
                deleted = delete_corpus_files(settings.data_dir, selected)
                if deleted:
                    st.session_state["corpus_dirty"] = True
                    st.session_state.pop("corpus_inspection", None)
                    st.session_state.pop("corpus_build", None)
                st.session_state["document_notice"] = (
                    "warning",
                    f"Deleted {len(deleted)} local file(s). Rebuild both indexes to apply it.",
                )
                st.rerun()
        else:
            st.info("No local documents have been added yet.")

        if st.button("Inspect and chunk corpus", disabled=not corpus_files):
            with st.spinner("Extracting text and calculating chunks without API calls..."):
                st.session_state["corpus_inspection"] = inspect_corpus(settings)

        inspection = st.session_state.get("corpus_inspection")
        if isinstance(inspection, CorpusInspection):
            _render_corpus_inspection(inspection)

        st.divider()
        has_existing_index = dense.count > 0 or sparse.count > 0
        confirmed = not has_existing_index or st.checkbox(
            "I understand that this will synchronize and may replace existing index records",
            help="Source files are not deleted. Chroma embeds only changed chunks; BM25 is local.",
        )
        can_embed = settings.embedding_provider != "openai" or _api_key_configured(settings)
        if not can_embed:
            st.warning("An OpenAI API key is required to embed new or changed chunks.")
        st.caption(
            "The chat token budget does not cover embedding requests. Chroma skips unchanged "
            "chunks, and BM25 indexing does not call an external API."
        )
        if st.button(
            "Build or synchronize both indexes",
            type="primary",
            disabled=not corpus_files or not confirmed or not can_embed,
        ):
            st.session_state.pop("corpus_build", None)
            try:
                with st.spinner("Ingesting once, synchronizing Chroma, and rebuilding BM25..."):
                    current_inspection = inspect_corpus(settings)
                    build = synchronize_corpus(
                        settings,
                        runtime.dense,
                        runtime.sparse,
                        inspection=current_inspection,
                    )
            except CorpusBuildError as exc:
                st.session_state["corpus_inspection"] = exc.inspection
                st.error(str(exc))
            except Exception as exc:
                st.error("The indexes could not be synchronized; existing source files were kept.")
                st.exception(exc)
            else:
                st.session_state["corpus_build"] = build
                st.session_state["corpus_inspection"] = build.inspection
                st.session_state["corpus_dirty"] = False
                st.session_state["chat_turns"] = []
                load_agent.clear()
                st.session_state["document_notice"] = (
                    "success",
                    "Both indexes were synchronized from the same inspected corpus.",
                )
                st.rerun()

        build_result = st.session_state.get("corpus_build")
        if isinstance(build_result, CorpusBuildResult):
            _render_build_result(build_result)


def _render_corpus_inspection(inspection: CorpusInspection) -> None:
    summary = inspection.ingestion
    columns = st.columns(4)
    columns[0].metric("Processed", summary.processed_files)
    columns[1].metric("Skipped", summary.skipped_files)
    columns[2].metric("Failed", summary.failed_files)
    columns[3].metric("Chunks", f"{inspection.chunks:,}")
    st.caption(
        f"{inspection.pages:,} extracted PDF pages · {inspection.tokens:,} chunk tokens · "
        "inspection uses no model API"
    )
    st.dataframe(
        [
            {
                "Document": document.relative_path,
                "Status": document.status,
                "Size": _format_bytes(document.size_bytes),
                "Pages": document.pages or "—",
                "Chunks": document.chunks,
                "Tokens": document.tokens,
                "Detail": document.detail,
            }
            for document in inspection.documents
        ],
        use_container_width=True,
        hide_index=True,
    )
    if summary.failed_files:
        st.error(
            "Fix or remove failed documents before indexing. Existing indexes were not changed."
        )
    elif summary.issues:
        st.warning("Some files were skipped. Review the status and detail columns before indexing.")


def _render_build_result(result: CorpusBuildResult) -> None:
    with st.container(border=True):
        st.markdown("**Last index synchronization**")
        st.write(
            f"Dense: {result.dense.embedded_chunks:,} embedded, "
            f"{result.dense.unchanged_chunks:,} unchanged, "
            f"{result.dense.deleted_chunks:,} deleted, "
            f"{result.dense.total_chunks:,} total."
        )
        sparse_action = "rebuilt" if result.sparse.rebuilt else "unchanged"
        st.write(f"BM25: {sparse_action}, {result.sparse.total_chunks:,} total chunks.")


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _render_answer(result: AgentAnswer) -> None:
    answer_tab, evidence_tab, trace_tab, diagnostics_tab = st.tabs(
        ["Answer", "Evidence", "Agent trace", "Diagnostics"]
    )
    with answer_tab:
        if result.insufficient_evidence:
            st.warning(result.answer)
        else:
            st.markdown(result.answer)
        if result.errors:
            st.warning("Run warnings: " + " ".join(result.errors))

    with evidence_tab:
        if not result.sources:
            st.info("No evidence was cited for this response.")
        for source in result.sources:
            page = f"page {source.page_number}" if source.page_number else "page not available"
            title = source.title or source.filename
            with st.container(border=True):
                st.markdown(f"### [{source.label}] {title}")
                st.caption(
                    f"{source.filename} · {page} · chunk `{source.chunk_id}` · "
                    f"{source.token_count} tokens"
                )
                rank_columns = st.columns(4)
                rank_columns[0].metric("Final rank", source.final_rank or "—")
                rank_columns[1].metric("Dense rank", source.dense_rank or "—")
                rank_columns[2].metric("BM25 rank", source.sparse_rank or "—")
                rank_columns[3].metric("Reranker", _format_score(source.reranker_score))
                st.caption(
                    f"Dense score: {_format_score(source.dense_score)} · "
                    f"BM25 score: {_format_score(source.sparse_score)} · "
                    f"RRF score: {_format_score(source.rrf_score)} · "
                    f"Found by: {', '.join(source.retrieval_sources) or 'not recorded'}"
                )
                st.write(source.text or "Evidence text was not recorded for this result.")

    with trace_tab:
        if not result.graph_trace:
            st.info("No graph trace was recorded.")
        else:
            st.markdown(" → ".join(f"`{event.node}`" for event in result.graph_trace))
            for position, event in enumerate(result.graph_trace, start=1):
                st.markdown(f"**{position}. {event.node} — {event.action}**")
                st.caption(f"Retrieval attempt {event.retrieval_attempt} · {event.detail}")

    with diagnostics_tab:
        diagnostics = result.retrieval_diagnostics
        total_tokens = result.usage.input_tokens + result.usage.output_tokens
        metric_columns = st.columns(5)
        metric_columns[0].metric("Confidence", diagnostics.confidence)
        metric_columns[1].metric("Retrieval attempts", result.retrieval_attempts)
        metric_columns[2].metric("Agent steps", len(result.graph_trace))
        metric_columns[3].metric("Chat API calls", result.usage.api_calls)
        metric_columns[4].metric("Total tokens", f"{total_tokens:,}")
        st.json(
            {
                "query": result.final_query,
                "candidates": {
                    "dense": diagnostics.dense_candidates,
                    "sparse": diagnostics.sparse_candidates,
                    "fused": diagnostics.fused_candidates,
                    "reranked": diagnostics.reranked_candidates,
                },
                "retriever_agreements": diagnostics.agreement_count,
                "retrieved_sources": diagnostics.distinct_sources,
                "cited_sources": len(result.sources),
                "evidence_tokens": diagnostics.evidence_tokens,
                "reranker": diagnostics.reranker,
                "rerank_applied": diagnostics.rerank_applied,
                "citation_validation": result.citation_validation.model_dump(mode="json"),
                "usage": result.usage.model_dump(mode="json"),
                "warnings": diagnostics.warnings,
            },
            expanded=False,
        )


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _request_token_reserve(settings: Settings, max_retrieval_attempts: int) -> int:
    """Return a conservative preflight estimate, not a provider billing guarantee."""
    return (
        settings.hybrid_context_token_budget
        + settings.chat_max_output_tokens
        + 1000 * max_retrieval_attempts
    )


def _render_spending_status(
    decision: SpendingDecision,
    usage: SessionUsage,
) -> None:
    question_text = _remaining_text(decision.remaining_questions)
    call_text = _remaining_text(decision.remaining_api_calls)
    token_text = _remaining_text(decision.remaining_tokens, grouped=True)
    st.caption(
        f"Session spending: {usage.questions} questions, {usage.api_calls} chat API calls, "
        f"{usage.tokens:,} chat tokens used · Remaining: {question_text} questions, "
        f"{call_text} chat API calls, {token_text} chat tokens"
    )
    if not decision.allowed:
        st.warning("New questions are paused. " + " ".join(decision.reasons))


def _remaining_text(value: int | None, *, grouped: bool = False) -> str:
    if value is None:
        return "unlimited"
    return f"{value:,}" if grouped else str(value)


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

    usage = _session_usage()
    options = _render_sidebar(runtime.settings, usage)
    corpus_current = not bool(st.session_state.get("corpus_dirty", False))
    readiness = _render_readiness(
        runtime.settings,
        dense_info,
        sparse_info,
        corpus_current=corpus_current,
    )
    _render_document_management(runtime, dense_info, sparse_info)
    budget = spending_decision(
        options.limits,
        usage,
        api_call_reserve=options.max_retrieval_attempts,
        token_reserve=_request_token_reserve(
            runtime.settings,
            options.max_retrieval_attempts,
        ),
    )

    st.divider()
    st.subheader("Ask your corpus")
    _render_spending_status(budget, usage)
    turns = _chat_turns()
    if not turns:
        st.info(
            "Ask about the indexed papers. Answers are generated from reranked local evidence "
            "and citations are validated before display."
        )
    _render_history(turns)

    question = st.chat_input(
        "Ask a question about your document collection",
        disabled=not readiness.ready or not budget.allowed,
    )
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Running the six-node LangGraph workflow..."):
                result = load_agent().ask(
                    question,
                    top_k=options.top_k,
                    max_retrieval_attempts=options.max_retrieval_attempts,
                    max_agent_steps=options.max_agent_steps,
                )
        except Exception as exc:
            st.error("The agent could not complete this question.")
            st.exception(exc)
            return
        _render_answer(result)
    turns.append(ChatTurn(question=question, result=result))
    total_tokens = result.usage.input_tokens + result.usage.output_tokens
    st.session_state["session_usage"] = usage.add(
        api_calls=result.usage.api_calls,
        tokens=total_tokens,
    )
    st.rerun()


if __name__ == "__main__":
    main()

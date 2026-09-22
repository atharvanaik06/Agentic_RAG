"""Shared constructors for CLI and web application runtimes."""

from advanced_rag.config import Settings
from advanced_rag.generation.chat import create_chat_model
from advanced_rag.graph import AgenticRAG
from advanced_rag.retrieval.dense import ChromaDenseIndex
from advanced_rag.retrieval.embeddings import create_embedding_provider
from advanced_rag.retrieval.hybrid import HybridRetriever
from advanced_rag.retrieval.rerankers import create_reranker
from advanced_rag.retrieval.sparse import BM25SparseIndex
from advanced_rag.tools import create_search_knowledge_base_tool


def create_dense_index(settings: Settings) -> ChromaDenseIndex:
    """Create the configured persistent dense index."""
    return ChromaDenseIndex(
        path=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embedder=create_embedding_provider(settings),
        write_batch_size=settings.embedding_batch_size,
    )


def create_sparse_index(settings: Settings) -> BM25SparseIndex:
    """Create the configured persistent sparse index."""
    return BM25SparseIndex(
        path=settings.bm25_dir,
        method=settings.bm25_method,
        k1=settings.bm25_k1,
        b=settings.bm25_b,
    )


def create_hybrid_retriever(
    settings: Settings,
    *,
    dense: ChromaDenseIndex | None = None,
    sparse: BM25SparseIndex | None = None,
) -> HybridRetriever:
    """Create one hybrid retriever, optionally reusing initialized indexes."""
    return HybridRetriever(
        dense=dense or create_dense_index(settings),
        sparse=sparse or create_sparse_index(settings),
        reranker=create_reranker(settings),
        candidate_k=settings.hybrid_candidate_k,
        rerank_k=settings.hybrid_rerank_k,
        rrf_k=settings.hybrid_rrf_k,
        context_token_budget=settings.hybrid_context_token_budget,
        max_chunks_per_source=settings.hybrid_max_chunks_per_source,
        max_chunks_per_page=settings.hybrid_max_chunks_per_page,
    )


def create_agent(
    settings: Settings,
    *,
    dense: ChromaDenseIndex | None = None,
    sparse: BM25SparseIndex | None = None,
) -> AgenticRAG:
    """Create the complete bounded LangGraph agent, optionally reusing indexes."""
    retriever = create_hybrid_retriever(settings, dense=dense, sparse=sparse)
    chat_model = create_chat_model(settings)
    return AgenticRAG(
        search_tool=create_search_knowledge_base_tool(retriever),
        chat_model=chat_model,
        evidence_grader=chat_model,
        semantic_evidence_grading=settings.agent_semantic_evidence_grading,
        max_retrieval_attempts=settings.agent_max_retrieval_attempts,
        max_agent_steps=settings.agent_max_steps,
        top_k=settings.agent_top_k,
        scope_description=settings.agent_scope_description,
        scope_terms=settings.agent_scope_term_list,
    )

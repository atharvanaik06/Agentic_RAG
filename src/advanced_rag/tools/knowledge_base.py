"""LangChain-compatible tool wrapper for hybrid knowledge-base search."""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from advanced_rag.ingestion.models import FileType
from advanced_rag.retrieval.hybrid import HybridRetriever
from advanced_rag.retrieval.models import RetrievalFilters


class KnowledgeSearchInput(BaseModel):
    """Arguments exposed to the agent for local evidence retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Specific question or search query")
    top_k: int = Field(default=6, ge=1, le=20, description="Maximum evidence chunks")
    source_id: str | None = Field(default=None, description="Exact source identifier filter")
    filename: str | None = Field(default=None, description="Exact source filename filter")
    file_type: FileType | None = Field(default=None, description="PDF, Markdown, or text")
    page_number: int | None = Field(default=None, ge=1, description="Exact PDF page filter")


def create_search_knowledge_base_tool(retriever: HybridRetriever) -> BaseTool:
    """Wrap one configured hybrid retriever as a structured agent tool."""

    def search_knowledge_base(
        query: str,
        top_k: int = 6,
        source_id: str | None = None,
        filename: str | None = None,
        file_type: FileType | None = None,
        page_number: int | None = None,
    ) -> dict[str, Any]:
        """Search indexed local documents for citation-ready evidence."""
        filters = RetrievalFilters(
            source_id=source_id,
            filename=filename,
            file_type=file_type,
            page_number=page_number,
        )
        response = retriever.search(query, top_k=top_k, filters=filters)
        return response.model_dump(mode="json")

    return StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search the indexed local knowledge base with semantic and keyword retrieval. "
            "Returns reranked evidence with filenames, page numbers, chunk IDs, and confidence "
            "diagnostics. Use this before answering questions about the document corpus."
        ),
        args_schema=KnowledgeSearchInput,
    )

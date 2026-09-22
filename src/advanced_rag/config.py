"""Typed application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from `RAG_` environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "advanced-agentic-rag"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    data_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/indexes")
    chroma_dir: Path = Path("data/indexes/chroma")
    chroma_collection: str = "advanced-rag"

    chunk_size: int = Field(default=500, ge=50)
    chunk_overlap: int = Field(default=75, ge=0)
    min_chunk_size: int = Field(default=50, ge=1)
    token_encoding: str = "cl100k_base"

    embedding_provider: Literal["openai", "deterministic"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)
    embedding_batch_size: int = Field(default=100, gt=0, le=2048)

    bm25_dir: Path = Path("data/indexes/bm25")
    bm25_method: Literal["robertson", "lucene", "atire", "bm25l", "bm25+"] = "lucene"
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)

    hybrid_candidate_k: int = Field(default=15, gt=0, le=100)
    hybrid_rerank_k: int = Field(default=15, gt=0, le=100)
    hybrid_rrf_k: int = Field(default=60, gt=0)
    hybrid_context_token_budget: int = Field(default=3000, gt=0)
    hybrid_max_chunks_per_source: int = Field(default=3, gt=0)
    hybrid_max_chunks_per_page: int = Field(default=2, gt=0)

    reranker_provider: Literal["flashrank", "none"] = "flashrank"
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    reranker_cache_dir: Path = Path("data/models/flashrank")
    reranker_max_length: int = Field(default=512, ge=64, le=8192)

    chat_provider: Literal["openai"] = "openai"
    chat_model: str = "gpt-4o-mini"
    chat_max_output_tokens: int = Field(default=1200, ge=100, le=16000)
    agent_max_retrieval_attempts: int = Field(default=2, ge=1, le=5)
    agent_max_steps: int = Field(default=8, ge=5, le=100)
    agent_top_k: int = Field(default=6, ge=1, le=20)
    agent_semantic_evidence_grading: bool = True
    agent_scope_description: str = "monetary policy, central banking, and macro-financial research"
    agent_scope_terms: str = (
        "monetary,inflation,disinflation,central bank,central banking,federal reserve,fed,fomc,"
        "ecb,bis,interest rate,policy rate,financial stress,financial condition,prices,wages,"
        "profits,employment,labor market,economic growth,macroeconomic,term premium,yield"
    )

    ui_session_question_limit: int = Field(default=10, ge=0, le=1000)
    ui_session_api_call_limit: int = Field(default=20, ge=0, le=10000)
    ui_session_token_budget: int = Field(default=50000, ge=0, le=10000000)
    ui_max_upload_mb: int = Field(default=100, ge=1, le=1000)

    evaluation_dir: Path = Path("reports")
    evaluation_entailment_model: str = "gpt-4o-mini"
    evaluation_hybrid_recall_threshold: float = Field(default=0.75, ge=0, le=1)
    evaluation_citation_validity_threshold: float = Field(default=1.0, ge=0, le=1)
    evaluation_refusal_accuracy_threshold: float = Field(default=0.8, ge=0, le=1)
    evaluation_max_average_attempts: float = Field(default=1.5, ge=1, le=5)

    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    anthropic_api_key: SecretStr | None = Field(default=None, repr=False)
    langsmith_api_key: SecretStr | None = Field(default=None, repr=False)
    langsmith_tracing: bool = False
    enable_web_search: bool = False

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        """Ensure chunking windows can always make forward progress."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must not exceed chunk_size")
        return self

    def ensure_directories(self) -> None:
        """Create local data directories when an entry point needs them."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_dir.mkdir(parents=True, exist_ok=True)
        self.reranker_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def agent_scope_term_list(self) -> tuple[str, ...]:
        """Return normalized comma-separated scope terms for the local domain gate."""
        return tuple(
            term
            for value in self.agent_scope_terms.split(",")
            if (term := value.strip().casefold())
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-local, cached settings instance."""
    return Settings()

"""Application settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Everything the app needs that might reasonably change between environments."""

    chat_deployment: str
    embedding_deployment: str
    api_version: str | None

    collection_name: str = "document_chunks_agentic_rag"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    web_max_results: int = 5

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the process environment (call ``load_dotenv`` first)."""
        return cls(
            chat_deployment=os.getenv("AZURE_COMPLETION_MODEL", "gpt-4o"),
            embedding_deployment=os.getenv(
                "AZURE_EMBEDDING_MODEL", "text-embedding-3-large"
            ),
            api_version=os.getenv("OPENAI_API_VERSION"),
        )

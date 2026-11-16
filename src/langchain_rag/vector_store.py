"""Weaviate connection and vector store helpers."""

from __future__ import annotations

import weaviate
from langchain_core.embeddings import Embeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate import WeaviateClient


def connect_weaviate() -> WeaviateClient:
    """Connect to the local Weaviate instance (HTTP 8080, gRPC 50051)."""
    return weaviate.connect_to_local()


def build_vector_store(
    client: WeaviateClient,
    embeddings: Embeddings,
    collection_name: str,
) -> WeaviateVectorStore:
    """Wrap a collection as a LangChain vector store, creating it if it doesn't exist."""
    return WeaviateVectorStore(
        client=client,
        index_name=collection_name,
        text_key="text",
        embedding=embeddings,
    )


def delete_collection(client: WeaviateClient, collection_name: str) -> None:
    """Delete the collection and all its chunks, if it exists."""
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

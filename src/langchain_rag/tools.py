"""Tools the agents can call."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ddgs import DDGS
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore

_SEPARATOR = "\n\n---\n\n"


def format_passages(documents: Iterable[Document]) -> str:
    """Render retrieved chunks with a source/page header (pages shown 1-based)."""
    return _SEPARATOR.join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"page {int(doc.metadata.get('page', 0)) + 1}]\n{doc.page_content}"
        for doc in documents
    )


def format_web_results(results: Iterable[Mapping[str, Any]]) -> str:
    """Render DDGS text results as markdown links with their snippets."""
    return _SEPARATOR.join(
        f"[{r.get('title', 'Untitled')}]({r.get('href', '')})\n{r.get('body', '')}"
        for r in results
    )


def make_search_document_tool(vector_store: VectorStore, k: int = 4) -> BaseTool:
    """Create a retriever tool bound to ``vector_store``."""

    @tool
    def search_document(query: str) -> str:
        """Search the uploaded PDF document for passages relevant to the query.

        Use this tool for any question about the content of the uploaded document.
        """
        results = vector_store.similarity_search(query, k=k)
        if not results:
            return "No relevant passages found in the document."
        return format_passages(results)

    return search_document


def make_web_search_tool(max_results: int = 5) -> BaseTool:
    """Create a DuckDuckGo-backed web search tool (no API key needed)."""

    @tool
    def web_search(query: str) -> str:
        """Search the web for up-to-date information relevant to the query.

        Use this tool for questions about current events, recent facts, or anything
        you are not confident you know.
        """
        try:
            results = DDGS().text(query, max_results=max_results)
        except Exception as exc:  # network errors, rate limits, bot blocking
            return f"Web search failed: {exc}"

        if not results:
            return "No web results found."
        return format_web_results(results)

    return web_search

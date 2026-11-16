"""Agent construction and invocation."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.vectorstores import VectorStore

from langchain_rag.tools import make_search_document_tool, make_web_search_tool

DOCUMENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about an uploaded PDF document. "
    "Whenever the user asks about the document, call the `search_document` tool to "
    "retrieve relevant passages, and base your answer only on what it returns. "
    "You may call the tool several times with different queries if needed. "
    "Cite page numbers where possible. If the answer is not in the retrieved "
    "passages, say that you could not find it in the document."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly from your own "
    "knowledge when you can. If the question concerns current events, recent or "
    "changing facts, or anything you are not confident about, call the `web_search` "
    "tool and base your answer on its results, citing the source links you used."
)


def build_document_agent(
    llm: BaseChatModel, vector_store: VectorStore, k: int = 4
) -> Any:
    """Agent that answers only from the indexed PDF via `search_document`."""
    return create_agent(
        model=llm,
        tools=[make_search_document_tool(vector_store, k=k)],
        system_prompt=DOCUMENT_SYSTEM_PROMPT,
    )


def build_general_agent(llm: BaseChatModel, max_results: int = 5) -> Any:
    """Agent used when no PDF is uploaded: answers directly, with web search."""
    return create_agent(
        model=llm,
        tools=[make_web_search_tool(max_results=max_results)],
        system_prompt=GENERAL_SYSTEM_PROMPT,
    )


def ask(agent: Any, question: str) -> list[AnyMessage]:
    """Run one question through the agent and return the full message trace.

    The final answer is ``messages[-1].content``.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"]

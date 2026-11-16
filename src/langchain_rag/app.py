"""Streamlit UI for the agentic RAG app.

Run with:  .venv\\Scripts\\streamlit run src\\langchain_rag\\app.py
"""

from __future__ import annotations

import atexit
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile
from weaviate import WeaviateClient

from langchain_rag.agents import ask, build_document_agent, build_general_agent
from langchain_rag.config import Settings
from langchain_rag.ingestion import load_pdf, split_documents
from langchain_rag.models import build_chat_model, build_embeddings
from langchain_rag.vector_store import (
    build_vector_store,
    connect_weaviate,
    delete_collection,
)

load_dotenv()

# Session-state key holding the file_id of the upload currently in Weaviate.
INDEXED_FILE_KEY = "indexed_file_id"


# -------------------------------------------------------------------
# Shared resources (created once per server process)
# -------------------------------------------------------------------


@st.cache_resource
def get_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def get_llm() -> AzureChatOpenAI:
    return build_chat_model(get_settings())


@st.cache_resource
def get_embeddings() -> AzureOpenAIEmbeddings:
    return build_embeddings(get_settings())


@st.cache_resource
def get_weaviate_client() -> WeaviateClient:
    client = connect_weaviate()
    atexit.register(client.close)
    return client


@st.cache_resource
def get_vector_store() -> WeaviateVectorStore:
    return build_vector_store(
        get_weaviate_client(), get_embeddings(), get_settings().collection_name
    )


@st.cache_resource
def get_document_agent() -> Any:
    settings = get_settings()
    return build_document_agent(get_llm(), get_vector_store(), k=settings.retrieval_k)


@st.cache_resource
def get_general_agent() -> Any:
    return build_general_agent(get_llm(), max_results=get_settings().web_max_results)


# -------------------------------------------------------------------
# Indexing
# -------------------------------------------------------------------


def index_pdf(uploaded_file: UploadedFile) -> int:
    """Replace the indexed document with ``uploaded_file``; return the chunk count."""
    settings = get_settings()

    # Parse before touching Weaviate so a bad PDF doesn't wipe the current index.
    pages = load_pdf(uploaded_file.getvalue(), uploaded_file.name)
    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError(
            "No text could be extracted from this PDF (it may be a scanned image)."
        )

    delete_collection(get_weaviate_client(), settings.collection_name)
    # The cached store and agent point at the deleted collection; rebuild them.
    get_vector_store.clear()
    get_document_agent.clear()
    get_vector_store().add_documents(chunks)
    return len(chunks)


def sync_index(uploaded_file: UploadedFile | None) -> bool:
    """Index ``uploaded_file`` if it is new; return whether it is ready to query."""
    if uploaded_file is None:
        return False
    if st.session_state.get(INDEXED_FILE_KEY) == uploaded_file.file_id:
        return True

    with st.spinner("Indexing document..."):
        try:
            chunk_count = index_pdf(uploaded_file)
        except Exception as exc:
            st.session_state[INDEXED_FILE_KEY] = None
            st.error(f"Could not index {uploaded_file.name}: {exc}")
            return False

    st.session_state[INDEXED_FILE_KEY] = uploaded_file.file_id
    st.success(f"Indexed {chunk_count} chunks from {uploaded_file.name}.")
    return True


# -------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------


def render_agent_steps(messages: list[AnyMessage]) -> None:
    with st.expander("Agent steps"):
        for message in messages:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    st.markdown(f"**Tool call:** `{call['name']}({call['args']})`")
            elif message.type == "tool":
                st.text(message.content)


def answer(question: str, uploaded_file: UploadedFile | None, pdf_ready: bool) -> None:
    if pdf_ready and uploaded_file is not None:
        agent = get_document_agent()
        st.caption(f"Answering from: {uploaded_file.name}")
    else:
        agent = get_general_agent()
        st.caption("No PDF uploaded: answering with the model and web search.")

    with st.spinner("Thinking..."):
        try:
            messages = ask(agent, question)
        except Exception as exc:
            st.error(f"The agent failed to answer: {exc}")
            return

    st.markdown(messages[-1].content)
    render_agent_steps(messages)


# -------------------------------------------------------------------
# Page
# -------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Agentic RAG", page_icon="🤖")
    st.title("Agentic RAG with LangChain + Weaviate")

    try:
        get_weaviate_client()
    except Exception as exc:
        st.error(
            "Could not connect to Weaviate on localhost:8080 (gRPC 50051). "
            f"Is it running?\n\n{exc}"
        )
        st.stop()

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    question = st.text_input("Ask a question")

    pdf_ready = sync_index(uploaded_file)

    if question:
        answer(question, uploaded_file, pdf_ready)


if __name__ == "__main__":
    main()

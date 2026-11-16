import os
import tempfile
from pathlib import Path

import streamlit as st
import weaviate
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore

load_dotenv()

st.set_page_config(page_title="Agentic RAG", page_icon="🤖")
st.title("Agentic RAG with LangChain + Weaviate")

collection_name = "document_chunks_agentic_rag"


# -------------------------------------------------------------------
# 1. Create models
# -------------------------------------------------------------------

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_COMPLETION_MODEL", "gpt-4o"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv(
        "AZURE_EMBEDDING_MODEL",
        "text-embedding-3-large",
    ),
    api_version=os.getenv("OPENAI_API_VERSION"),
)


# -------------------------------------------------------------------
# 2. Create Weaviate client and vector store once
# -------------------------------------------------------------------


@st.cache_resource
def get_weaviate_client():
    return weaviate.connect_to_local()


@st.cache_resource
def get_vector_store():
    client = get_weaviate_client()

    return WeaviateVectorStore(
        client=client,
        index_name=collection_name,
        text_key="text",
        embedding=embeddings,
    )


client = get_weaviate_client()
vector_store = get_vector_store()


# -------------------------------------------------------------------
# 3. Helper: reset collection
# -------------------------------------------------------------------


def reset_collection() -> None:
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)


# -------------------------------------------------------------------
# 4. Streamlit inputs
# -------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
question = st.text_input("Ask a question")

if "indexed_file_name" not in st.session_state:
    st.session_state.indexed_file_name = None


# -------------------------------------------------------------------
# 5. Index uploaded PDF
# -------------------------------------------------------------------

if uploaded_file and uploaded_file.name != st.session_state.indexed_file_name:
    with st.spinner("Indexing document..."):
        reset_collection()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = Path(temp_file.name)

        # -------------------------------------------------------------------------------
        # Step 1:
        # Use PyPDFLoader to parse the uploaded pdf file in file_path, split the parsed
        # documents into chunks and store them in the WeaviateVectorStore vector_store.
        # -------------------------------------------------------------------------------
        try:
            docs = PyPDFLoader(str(file_path)).load()
        finally:
            file_path.unlink(missing_ok=True)

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)

        # The collection was just deleted, so rebuild the vector store to recreate it
        get_vector_store.clear()
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        st.session_state.indexed_file_name = uploaded_file.name
        st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}.")


# -------------------------------------------------------------------
# 6. Create retriever tool
# -------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Step 2:
# Create a tool function search_document that takes a user query as input, performs a similarity
# search on the vector_store and returns the retrieved document chunks as a string.
# -------------------------------------------------------------------------------


@tool
def search_document(query: str) -> str:
    """Search the uploaded PDF document for passages relevant to the query.

    Use this tool for any question about the content of the uploaded document.
    """
    results = vector_store.similarity_search(query, k=4)
    if not results:
        return "No relevant passages found in the document."

    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"page {int(doc.metadata.get('page', 0)) + 1}]\n{doc.page_content}"
        for doc in results
    )


# -------------------------------------------------------------------
# 7. Create agent
# -------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Step 3:
# Create the agent using create_agent with the llm, the search tool and a
# system prompt that instructs the agent to use the search tool when the user asks
# about the uploaded PDF.
# -------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about an uploaded PDF document. "
    "Whenever the user asks about the document, call the `search_document` tool to "
    "retrieve relevant passages, and base your answer only on what it returns. "
    "You may call the tool several times with different queries if needed. "
    "Cite page numbers where possible. If the answer is not in the retrieved "
    "passages, say that you could not find it in the document."
)

agent = create_agent(
    model=llm,
    tools=[search_document],
    system_prompt=SYSTEM_PROMPT,
)


# -------------------------------------------------------------------
# 8. Ask the agent
# -------------------------------------------------------------------

if question:
    if st.session_state.indexed_file_name is None:
        st.warning("Please upload a PDF first.")
    else:
        # -------------------------------------------------------------------------------
        # Step 4:
        # Invoke the agent with the user question and display the answer.
        # -------------------------------------------------------------------------------
        with st.spinner("Thinking..."):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

        st.markdown(result["messages"][-1].content)

        with st.expander("Agent steps"):
            for message in result["messages"]:
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    for call in tool_calls:
                        st.markdown(f"**Tool call:** `{call['name']}({call['args']})`")
                elif message.type == "tool":
                    st.text(message.content)

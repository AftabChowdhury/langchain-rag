# CLAUDE.md

Agentic RAG training project: a Streamlit app where a user uploads a PDF and asks questions, and a LangChain agent answers by calling a retriever tool backed by Weaviate.

## Stack

- Python >=3.11, managed with Poetry (`pyproject.toml`, `poetry.lock`); virtualenv in `.venv/`
- LangChain 1.x (`langchain.agents.create_agent`, `langchain.tools.tool`)
- LLM + embeddings: **Azure OpenAI** via `langchain-openai` (`AzureChatOpenAI`, `AzureOpenAIEmbeddings`)
- Vector store: `langchain-weaviate` 0.0.8 + `weaviate-client` 4.x, local Weaviate instance
- PDF loading: `PyPDFLoader` from `langchain-community` (prints a deprecation warning, which can be ignored); splitting: `langchain_text_splitters.RecursiveCharacterTextSplitter`
- UI: Streamlit

## Layout

- `src/langchain_rag/app.py`: the whole app (models, Weaviate client/vector store, PDF indexing, `search_document` tool, agent, Q&A UI)
- `tests/`: empty so far

## Running

Weaviate must be running locally on `localhost:8080` (gRPC 50051); the app uses `weaviate.connect_to_local()`.

```
.venv\Scripts\streamlit run src\langchain_rag\app.py
```

Use `.venv\Scripts\python` (Windows) for scripts and checks.

## Configuration (`.env`, loaded with python-dotenv)

- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`, `OPENAI_API_TYPE`
- `AZURE_COMPLETION_MODEL`: chat deployment name (code falls back to `gpt-4o`)
- `AZURE_EMBEDDING_MODEL`: embedding deployment name (code falls back to `text-embedding-3-large`)

Never print or commit `.env` values. `load_dotenv()` finds `.env` by searching upward from the calling script's directory, so a script run from outside the project (e.g. a scratch dir) must call `load_dotenv("<project>/.env")` explicitly.

## App behavior and gotchas

- Collection name: `document_chunks_agentic_rag`. Each new upload **deletes the collection** (`reset_collection`), so only one PDF is indexed at a time.
- The vector store is cached with `@st.cache_resource`; after deleting the collection, call `get_vector_store.clear()` and rebuild it before `add_documents`.
- Chunking: `chunk_size=1000`, `chunk_overlap=200`, `add_start_index=True`. Chunk metadata includes `source` (uploaded file name) and `page` (0-based from PyPDF; shown 1-based in tool output).
- `search_document` returns the top 4 similarity-search hits as a string with source/page headers; the agent's system prompt tells it to answer only from retrieved passages and cite pages.
- Agent output: `agent.invoke({"messages": [...]})["messages"][-1].content`; intermediate tool calls are shown in an "Agent steps" expander.
- For end-to-end tests against real services, use a throwaway Weaviate collection and delete it afterwards rather than touching `document_chunks_agentic_rag`.

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

- `src/langchain_rag/app.py`: Streamlit UI only: `@st.cache_resource` getters for shared resources, indexing flow, Q&A rendering
- `config.py`: `Settings` dataclass (env vars plus collection name, chunking, k, web result count)
- `models.py`: Azure chat model and embeddings builders
- `vector_store.py`: Weaviate connect, `WeaviateVectorStore` builder, `delete_collection`
- `ingestion.py`: `load_pdf(bytes, name)`, `split_documents`
- `tools.py`: `make_search_document_tool(vector_store, k)`, `make_web_search_tool(max_results)` factories plus formatters
- `agents.py`: system prompts, `build_document_agent`, `build_general_agent`, `ask(agent, question) -> messages`
- Only `app.py` imports Streamlit; the other modules are plain Python and can be tested without it.
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

- Collection name: `document_chunks_agentic_rag`. Each new upload **deletes the collection** (`vector_store.delete_collection`), so only one PDF is indexed at a time.
- The vector store is cached with `@st.cache_resource`; after deleting the collection, `index_pdf` clears `get_vector_store` and `get_document_agent` and rebuilds the store before `add_documents`. Uploads are tracked by `UploadedFile.file_id`, and the PDF is parsed before the old collection is deleted.
- Chunking: `chunk_size=1000`, `chunk_overlap=200`, `add_start_index=True`. Chunk metadata includes `source` (uploaded file name) and `page` (0-based from PyPDF; shown 1-based in tool output).
- `search_document` returns the top 4 similarity-search hits as a string with source/page headers; the agent's system prompt tells it to answer only from retrieved passages and cite pages.
- With no PDF in the uploader, questions go to the general agent (`build_general_agent`), which answers directly and has a `web_search` tool (DuckDuckGo metasearch via the `ddgs` package, no API key). These free search backends sometimes rate-limit or bot-block requests; the tool returns "Web search failed: ..." and the model says it couldn't find the answer.
- Agent output: `agents.ask(agent, question)` returns the message list; the answer is `messages[-1].content`; intermediate tool calls are shown in an "Agent steps" expander.
- For end-to-end tests against real services, use a throwaway Weaviate collection and delete it afterwards rather than touching `document_chunks_agentic_rag`.

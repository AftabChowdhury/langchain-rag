# Agentic RAG with LangChain + Weaviate

A Streamlit app that lets you upload a PDF and ask questions about it. A LangChain agent answers by calling a `search_document` retriever tool, which looks up relevant chunks of the PDF in a Weaviate vector store. The LLM and embeddings come from Azure OpenAI.

## How it works

1. **Upload**: you upload a PDF in the Streamlit UI.
2. **Load**: `PyPDFLoader` parses the PDF into one document per page.
3. **Split**: `RecursiveCharacterTextSplitter` cuts the pages into chunks of 1000 characters with 200 characters of overlap.
4. **Store**: the chunks are embedded with Azure OpenAI and stored in the Weaviate collection `document_chunks_agentic_rag`.
5. **Retrieve**: the `search_document` tool runs a similarity search and returns the top 4 chunks with their file name and page number.
6. **Answer**: an agent built with LangChain's `create_agent` decides when to call the tool, then answers from the retrieved passages and cites page numbers.

Uploading a new PDF replaces the previous one, so only one document is indexed at a time.

## Prerequisites

- Python 3.11 or newer
- [Poetry](https://python-poetry.org/)
- Docker, to run Weaviate locally
- An Azure OpenAI resource with a chat deployment (e.g. `gpt-4o`) and an embedding deployment (e.g. `text-embedding-3-large`)

## Setup

### 1. Install dependencies

```bash
poetry install
```

### 2. Start Weaviate

```bash
docker run -d --name weaviate -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:1.37.4
```

The app connects with `weaviate.connect_to_local()`, which expects HTTP on port 8080 and gRPC on port 50051.

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=<api-version>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_COMPLETION_MODEL=gpt-4o
AZURE_EMBEDDING_MODEL=text-embedding-3-large
```

`AZURE_COMPLETION_MODEL` and `AZURE_EMBEDDING_MODEL` are your Azure **deployment names**. If they are not set, the app falls back to `gpt-4o` and `text-embedding-3-large`.

Do not commit `.env`.

## Run

```bash
poetry run streamlit run src/langchain_rag/app.py
```

On Windows you can also run it from the virtualenv directly:

```powershell
.venv\Scripts\streamlit run src\langchain_rag\app.py
```

Then open the URL that Streamlit prints (usually http://localhost:8501), upload a PDF, wait for the "Indexed N chunks" message, and ask a question. Expand **Agent steps** to see which searches the agent ran and what they returned.

## Project structure

```
langchain-rag/
├── src/langchain_rag/app.py   # Streamlit app: indexing, retriever tool, agent
├── tests/                     # tests (none yet)
├── pyproject.toml             # dependencies
└── .env                       # Azure OpenAI settings (not committed)
```

## Troubleshooting

- **`Missing credentials` error from OpenAI**: `.env` was not found or is missing `AZURE_OPENAI_API_KEY`. Run the app from the project root.
- **Cannot connect to Weaviate**: check that the container is running (`docker ps`) and that ports 8080 and 50051 are free.
- **`DeploymentNotFound` from Azure**: `AZURE_COMPLETION_MODEL` or `AZURE_EMBEDDING_MODEL` doesn't match a deployment name in your Azure resource.
- **`langchain-community` deprecation warning**: harmless; it comes from importing `PyPDFLoader`.

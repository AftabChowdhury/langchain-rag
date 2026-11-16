"""Azure OpenAI chat model and embeddings."""

from __future__ import annotations

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_rag.config import Settings


def build_chat_model(settings: Settings) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=settings.chat_deployment,
        api_version=settings.api_version,
        temperature=0,
    )


def build_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.embedding_deployment,
        api_version=settings.api_version,
    )

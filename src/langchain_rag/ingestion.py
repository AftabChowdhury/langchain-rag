"""Turning an uploaded PDF into chunks ready for the vector store."""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(data: bytes, source_name: str) -> list[Document]:
    """Parse PDF bytes into one document per page, tagged with ``source_name``.

    PyPDFLoader needs a path, so the bytes go through a temporary file that is
    removed afterwards.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "upload.pdf"
        pdf_path.write_bytes(data)
        pages = PyPDFLoader(str(pdf_path)).load()

    for page in pages:
        page.metadata["source"] = source_name
    return pages


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

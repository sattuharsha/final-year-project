"""
MODULE 1 — Knowledge Base Construction.
Clean, chunk, deduplicate documents; store dense embeddings in ChromaDB.
"""
from knowledge.construction import (
    clean_text,
    chunk_document,
    chunk_documents,
    deduplicate_documents,
    prepare_documents_for_indexing,
)

__all__ = [
    "clean_text",
    "chunk_document",
    "chunk_documents",
    "deduplicate_documents",
    "prepare_documents_for_indexing",
]

"""
RAG retrieval using ChromaDB and sentence-transformers.
Provides a search_knowledge_base function for the ADK agent tool.
"""
from __future__ import annotations

import os
from pathlib import Path

# Default path for persistent ChromaDB (relative to project root)
DEFAULT_CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")
# Collection name
COLLECTION_NAME = "fever_ragtruth"

_client = None
_collection = None


def get_chroma_client(path: str | Path | None = None):
    """Lazy singleton Chroma persistent client."""
    global _client
    if _client is None:
        import chromadb
        p = path or DEFAULT_CHROMA_PATH
        _client = chromadb.PersistentClient(path=str(p))
    return _client


def get_embedding_function():
    """Sentence-transformers embedding function for Chroma."""
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def get_collection(force_recreate: bool = False):
    """Get or create the RAG collection."""
    global _collection
    client = get_chroma_client()
    if force_recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    if _collection is None:
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            metadata={"description": "FEVER + RAGTruth RAG index"},
        )
    return _collection


def add_documents(
    documents: list[dict],
    batch_size: int = 256,
    ids: list[str] | None = None,
):
    """
    Add documents to the Chroma collection.
    Each document: {"text": str, "metadata": dict}.
    If ids is provided, it must have the same length as documents (for globally unique IDs).
    """
    coll = get_collection()
    texts = [d["text"] for d in documents]
    metadatas = [d.get("metadata", {}) for d in documents]
    if ids is not None:
        if len(ids) != len(documents):
            raise ValueError("ids length must match documents length")
        use_ids = ids
    else:
        use_ids = [f"doc_{i}" for i in range(len(documents))]
    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i : i + batch_size]
        chunk_metas = metadatas[i : i + batch_size]
        chunk_ids = use_ids[i : i + batch_size]
        coll.add(
            documents=chunk_texts,
            metadatas=chunk_metas,
            ids=chunk_ids,
        )


def search_knowledge_base(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
) -> dict:
    """
    Search the knowledge base for relevant passages to answer factual or
    verification questions. The index contains FEVER (fact verification)
    claims and RAGTruth (QA/summary) content. Use this tool whenever the
    user asks a fact-based question or you need evidence to support or
    refute a claim.

    Args:
        query: The question or claim to look up (natural language).
        top_k: How many passages to return (default 5, max 20).
        source_filter: Optional. Set to 'fever' or 'ragtruth' to filter
            by dataset; omit to search both.

    Returns:
        A dict with status, results (list of text + metadata), and count.
    """
    try:
        coll = get_collection()
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "results": [],
            "count": 0,
        }
    where = None
    if source_filter and source_filter.lower() in ("fever", "ragtruth", "ragtruth_hf"):
        where = {"source": source_filter.lower()}
    try:
        results = coll.query(
            query_texts=[query],
            n_results=min(top_k, 20),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "results": [],
            "count": 0,
        }
    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []
    out = []
    for i, (text, meta, dist) in enumerate(zip(docs, metas, dists)):
        out.append({
            "text": text,
            "metadata": meta,
            "distance": float(dist) if dist is not None else None,
        })
    return {
        "status": "success",
        "results": out,
        "count": len(out),
    }

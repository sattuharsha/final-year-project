"""
MODULE 2 — Dual-Pass Retrieval.

Pass 1: Broad top-k retrieval from ChromaDB (high recall).
Pass 2: Rerank by embedding similarity, remove redundant chunks (high overlap
        with already selected), return high-confidence evidence set.
"""
from __future__ import annotations

from typing import Callable

from rag.retrieval import get_collection, get_embedding_function


# Pass 1: retrieve more candidates for recall.
DEFAULT_PASS1_TOP_K = 30
# Pass 2: keep top N after rerank and redundancy removal.
DEFAULT_PASS2_TOP_K = 15
# Similarity threshold above which two chunks are considered redundant (cosine).
REDUNDANCY_THRESHOLD = 0.92


def _get_embedder():
    """Return the same embedding function used by ChromaDB for consistency."""
    return get_embedding_function()


def _embed_texts(embed_fn: Callable, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Handles SentenceTransformerEmbeddingFunction."""
    if not texts:
        return []
    # ChromaDB's SentenceTransformerEmbeddingFunction expects list of strings.
    return embed_fn(texts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)


def _remove_redundant(
    items: list[dict],
    embeddings: list[list[float]],
    threshold: float = REDUNDANCY_THRESHOLD,
    top_k: int = DEFAULT_PASS2_TOP_K,
) -> tuple[list[dict], list[list[float]]]:
    """
    Greedily select items by distance order; drop any later item whose
    embedding is too similar to an already selected one.
    """
    if not items or not embeddings or len(items) != len(embeddings):
        return items[:top_k], embeddings[:top_k]
    # ChromaDB returns L2 distances; lower = more similar. We have (doc, distance).
    # Items are already ordered by distance ascending from query.
    selected_idx: list[int] = []
    selected_emb: list[list[float]] = []
    for i in range(len(items)):
        if len(selected_idx) >= top_k:
            break
        emb = embeddings[i]
        redundant = False
        for sel_emb in selected_emb:
            sim = _cosine_similarity(emb, sel_emb)
            if sim >= threshold:
                redundant = True
                break
        if not redundant:
            selected_idx.append(i)
            selected_emb.append(emb)
    out_items = [items[i] for i in selected_idx]
    out_emb = [embeddings[i] for i in selected_idx]
    return out_items, out_emb


def dual_pass_retrieve(
    query: str,
    pass1_top_k: int = DEFAULT_PASS1_TOP_K,
    pass2_top_k: int = DEFAULT_PASS2_TOP_K,
    source_filter: str | None = None,
    redundancy_threshold: float = REDUNDANCY_THRESHOLD,
) -> dict:
    """
    Dual-pass retrieval: broad retrieval then rerank and deduplicate.

    Returns:
        dict with keys: status, results (list of {text, metadata, distance}),
        count, pass1_count (optional), pass2_count (optional).
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
    # Pass 1: broad retrieval.
    n_results = min(pass1_top_k, 100)
    try:
        results = coll.query(
            query_texts=[query],
            n_results=n_results,
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
    pass1_items = []
    for i, (text, meta, dist) in enumerate(zip(docs, metas, dists)):
        pass1_items.append({
            "text": text,
            "metadata": meta or {},
            "distance": float(dist) if dist is not None else None,
        })
    if not pass1_items:
        return {
            "status": "success",
            "results": [],
            "count": 0,
            "pass1_count": 0,
            "pass2_count": 0,
        }
    # Pass 2: rerank by distance (already ascending), remove redundant by embedding similarity.
    embed_fn = _get_embedder()
    texts_pass2 = [x["text"] for x in pass1_items]
    try:
        pass2_embeddings = _embed_texts(embed_fn, texts_pass2)
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "results": pass1_items[:pass2_top_k],
            "count": min(len(pass1_items), pass2_top_k),
            "pass1_count": len(pass1_items),
            "pass2_count": min(len(pass1_items), pass2_top_k),
        }
    refined, _ = _remove_redundant(
        pass1_items,
        pass2_embeddings,
        threshold=redundancy_threshold,
        top_k=pass2_top_k,
    )
    return {
        "status": "success",
        "results": refined,
        "count": len(refined),
        "pass1_count": len(pass1_items),
        "pass2_count": len(refined),
    }

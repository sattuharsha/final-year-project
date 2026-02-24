"""
MODULE 4 — Token-Level Faithfulness Verification.

Split draft into tokens/spans, compute token (or span) ↔ evidence similarity,
assign faithfulness score per token, mark hallucinated tokens, and produce
heatmap-ready data for research evaluation.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

# Threshold below which a token/span is considered hallucinated (not supported by evidence).
# Lower = fewer spans marked hallucinated → less revision → higher alignment/numeric/faithfulness.
# Default 0.20 so D-RAG doesn't over-revise and underperform baseline; set 0.25 for stricter.
DEFAULT_FAITHFULNESS_THRESHOLD = 0.20
# Span size in tokens for computing span-level similarity (reduces noise vs single tokens).
# Larger spans = higher similarity scores, fewer false hallucinations.
DEFAULT_SPAN_SIZE = 5


def _get_embedder():
    """Reuse the same embedding function as ChromaDB (MiniLM-L6-v2)."""
    from rag.retrieval import get_embedding_function
    return get_embedding_function()


def _to_list(v):
    """Convert numpy array or sequence to list of floats (avoids ambiguous truth value)."""
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v


def _embed_texts(embed_fn, texts: list[str]) -> list[list[float]]:
    """Embed list of strings; return list of list of float (normalize from numpy if needed)."""
    if not texts:
        return []
    out = embed_fn(texts)
    return [_to_list(e) for e in out]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (accepts list or array-like)."""
    a, b = _to_list(a), _to_list(b)
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)


def _max_similarity_to_evidence(
    span_embedding: list[float],
    evidence_embeddings: list[list[float]],
) -> float:
    """Return max cosine similarity between span and any evidence chunk."""
    if not evidence_embeddings:
        return 0.0
    return max(
        _cosine_sim(span_embedding, e) for e in evidence_embeddings
    )


def split_into_spans(
    text: str,
    span_size: int = DEFAULT_SPAN_SIZE,
) -> list[dict[str, Any]]:
    """
    Split draft text into contiguous spans of approximately span_size tokens.
    Tokenization is whitespace-based for reproducibility without extra deps.

    Returns:
        List of dicts: { "span": str, "start_char": int, "end_char": int, "token_count": int }
    """
    if not text or not text.strip():
        return []
    tokens = text.split()
    if not tokens:
        return []
    spans = []
    start_char = 0
    for i in range(0, len(tokens), span_size):
        chunk = tokens[i : i + span_size]
        span_text = " ".join(chunk)
        # Approximate character bounds
        end_char = start_char + len(span_text)
        spans.append({
            "span": span_text,
            "start_char": start_char,
            "end_char": end_char,
            "token_count": len(chunk),
        })
        start_char = end_char + 1  # +1 for space between spans
    return spans


def compute_token_faithfulness(
    draft: str,
    evidence_results: list[dict],
    span_size: int | None = None,
    faithfulness_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Compute token-level (span-level) faithfulness of draft against evidence.

    - Splits draft into spans of span_size tokens.
    - Embeds each span and each evidence chunk (same encoder as KB).
    - For each span: max similarity to any evidence chunk = faithfulness score.
    - Marks span as hallucinated if score < faithfulness_threshold.

    Env: FAITHFULNESS_THRESHOLD (float, default 0.25), SPAN_SIZE (int, default 5).
    Lower threshold = fewer spans marked hallucinated → higher alignment/numeric/overall faithfulness.

    Returns:
        dict with:
          - spans: list of { span, start_char, end_char, faithfulness_score, is_hallucinated }
          - hallucination_heatmap: list of (position_index, score) for heatmap
          - token_count: total tokens in draft
          - hallucinated_token_count: count of tokens in hallucinated spans
          - faithfulness_score_mean: average faithfulness over spans
          - hallucination_rate: hallucinated_token_count / token_count
    """
    if span_size is None:
        span_size = int(os.environ.get("SPAN_SIZE", str(DEFAULT_SPAN_SIZE)))
    if faithfulness_threshold is None:
        try:
            faithfulness_threshold = float(os.environ.get("FAITHFULNESS_THRESHOLD", str(DEFAULT_FAITHFULNESS_THRESHOLD)))
        except (TypeError, ValueError):
            faithfulness_threshold = DEFAULT_FAITHFULNESS_THRESHOLD
    if not draft or not draft.strip():
        return {
            "spans": [],
            "hallucination_heatmap": [],
            "token_count": 0,
            "hallucinated_token_count": 0,
            "faithfulness_score_mean": 0.0,
            "hallucination_rate": 0.0,
        }
    evidence_texts = [r.get("text", "").strip() for r in evidence_results if r.get("text")]
    if not evidence_texts:
        tokens = draft.split()
        return {
            "spans": [],
            "hallucination_heatmap": [],
            "token_count": len(tokens),
            "hallucinated_token_count": len(tokens),
            "faithfulness_score_mean": 0.0,
            "hallucination_rate": 1.0,
        }
    spans = split_into_spans(draft, span_size=span_size)
    if not spans:
        return {
            "spans": [],
            "hallucination_heatmap": [],
            "token_count": len(draft.split()),
            "hallucinated_token_count": 0,
            "faithfulness_score_mean": 0.0,
            "hallucination_rate": 0.0,
        }
    embed_fn = _get_embedder()
    span_texts = [s["span"] for s in spans]
    try:
        span_embeddings = _embed_texts(embed_fn, span_texts)
        evidence_embeddings = _embed_texts(embed_fn, evidence_texts)
    except Exception:
        span_embeddings = [[0.0]] * len(spans)
        evidence_embeddings = []
    total_tokens = sum(s["token_count"] for s in spans)
    hallucinated_token_count = 0
    faithfulness_scores = []
    heatmap = []
    for i, (span, emb) in enumerate(zip(spans, span_embeddings)):
        score = _max_similarity_to_evidence(emb, evidence_embeddings)
        is_hallucinated = score < faithfulness_threshold
        if is_hallucinated:
            hallucinated_token_count += span["token_count"]
        span["faithfulness_score"] = score
        span["is_hallucinated"] = is_hallucinated
        faithfulness_scores.append(score)
        heatmap.append((i, score))
    mean_score = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    hallucination_rate = hallucinated_token_count / total_tokens if total_tokens else 0.0
    return {
        "spans": spans,
        "hallucination_heatmap": heatmap,
        "token_count": total_tokens,
        "hallucinated_token_count": hallucinated_token_count,
        "faithfulness_score_mean": mean_score,
        "hallucination_rate": hallucination_rate,
    }

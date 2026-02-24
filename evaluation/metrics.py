"""
MODULE 7 — Evaluation Metrics.

- Hallucination rate = (# hallucinated tokens / total tokens)
- Token-level faithfulness score = average faithfulness score over spans
- Numerical accuracy = verified numbers / total numbers
- Evidence alignment quality = similarity(answer, evidence)
- Overall faithfulness score = weighted combination of the above
"""
from __future__ import annotations

import math
from typing import Any

# Weights for overall faithfulness (must sum to 1.0).
WEIGHT_FAITHFULNESS = 0.35
WEIGHT_NUMERIC = 0.25
WEIGHT_ALIGNMENT = 0.25
WEIGHT_HALLUCINATION_PENALTY = 0.15  # (1 - hallucination_rate) contributes


def compute_hallucination_rate(faithfulness_result: dict[str, Any]) -> float:
    """
    Hallucination rate = (# hallucinated tokens / total tokens).
    faithfulness_result from compute_token_faithfulness.
    """
    total = faithfulness_result.get("token_count", 0)
    if total == 0:
        return 0.0
    hall = faithfulness_result.get("hallucinated_token_count", 0)
    return hall / total


def compute_faithfulness_score(faithfulness_result: dict[str, Any]) -> float:
    """
    Token-level faithfulness score = average faithfulness score over spans.
    """
    return faithfulness_result.get("faithfulness_score_mean", 0.0)


def compute_numeric_accuracy(numeric_result: dict[str, Any]) -> float:
    """
    Numerical accuracy = verified numbers / total numbers.
    numeric_result from verify_numerics_against_evidence.
    """
    return numeric_result.get("numeric_accuracy", 1.0)


def compute_alignment_score(
    answer: str,
    evidence_results: list[dict],
) -> float:
    """
    Evidence alignment quality = max similarity(answer, evidence_chunk).
    Uses same embedding model as retrieval for consistency.
    """
    if not answer or not answer.strip():
        return 0.0
    evidence_texts = [r.get("text", "").strip() for r in evidence_results if r.get("text")]
    if not evidence_texts:
        return 0.0
    from rag.retrieval import get_embedding_function
    embed_fn = get_embedding_function()
    try:
        emb_answer = embed_fn([answer])[0]
        emb_evidence = embed_fn(evidence_texts)
    except Exception:
        return 0.0
    # Normalize to list (embedder may return numpy arrays)
    def _to_list(v):
        if hasattr(v, "tolist"):
            return v.tolist()
        return list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v
    emb_answer = _to_list(emb_answer)
    emb_evidence = [_to_list(e) for e in emb_evidence]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    norm = lambda a: math.sqrt(sum(x * x for x in a)) or 1e-12
    sims = [dot(emb_answer, e) / (norm(emb_answer) * norm(e)) for e in emb_evidence]
    return max(sims) if sims else 0.0


def compute_overall_faithfulness(
    faithfulness_result: dict[str, Any],
    numeric_result: dict[str, Any],
    answer: str,
    evidence_results: list[dict],
    weight_faithfulness: float = WEIGHT_FAITHFULNESS,
    weight_numeric: float = WEIGHT_NUMERIC,
    weight_alignment: float = WEIGHT_ALIGNMENT,
    weight_hallucination: float = WEIGHT_HALLUCINATION_PENALTY,
) -> float:
    """
    Overall faithfulness = weighted combination of:
      - token-level faithfulness score
      - numeric accuracy
      - evidence alignment (answer vs evidence)
      - (1 - hallucination_rate) as penalty term
    """
    hall_rate = compute_hallucination_rate(faithfulness_result)
    faith = compute_faithfulness_score(faithfulness_result)
    num_acc = compute_numeric_accuracy(numeric_result)
    align = compute_alignment_score(answer, evidence_results)
    no_hall = 1.0 - hall_rate
    return (
        weight_faithfulness * faith
        + weight_numeric * num_acc
        + weight_alignment * align
        + weight_hallucination * no_hall
    )


def compute_all_metrics(
    answer: str,
    evidence_results: list[dict],
    faithfulness_result: dict[str, Any],
    numeric_result: dict[str, Any],
) -> dict[str, float]:
    """
    Compute all evaluation metrics for a single (answer, evidence) pair.

    Returns:
        dict with: hallucination_rate, faithfulness_score, numeric_accuracy,
        alignment_score, overall_faithfulness_score
    """
    hall_rate = compute_hallucination_rate(faithfulness_result)
    faith = compute_faithfulness_score(faithfulness_result)
    num_acc = compute_numeric_accuracy(numeric_result)
    align = compute_alignment_score(answer, evidence_results)
    overall = compute_overall_faithfulness(
        faithfulness_result,
        numeric_result,
        answer,
        evidence_results,
    )
    return {
        "hallucination_rate": hall_rate,
        "faithfulness_score": faith,
        "numeric_accuracy": num_acc,
        "alignment_score": align,
        "overall_faithfulness_score": overall,
    }

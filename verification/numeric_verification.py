"""
MODULE 5 — Numerical Fact Verification.

Extract numeric entities from the draft, validate each against evidence text,
detect numeric hallucinations (numbers not present or contradicted in evidence),
and output a structured mismatch list for evaluation.
"""
from __future__ import annotations

import re
from typing import Any

# Regex to capture numbers: integers, decimals, percentages, ordinals (e.g. 1st), years.
NUMERIC_PATTERN = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+%?|\d+(?:st|nd|rd|th)?)\b",
    re.IGNORECASE,
)


def _normalize_num(s: str) -> str:
    """Normalize string representation for comparison (e.g. 1,000 -> 1000)."""
    s = s.strip().replace(",", "")
    if s.endswith("%"):
        return s
    if s.lower().endswith(("st", "nd", "rd", "th")):
        return re.sub(r"(?i)(st|nd|rd|th)$", "", s)
    return s


def extract_numeric_entities(text: str) -> list[dict[str, Any]]:
    """
    Extract numeric entities from text with positions.

    Returns:
        List of { "value": str, "normalized": str, "start": int, "end": int }
    """
    if not text:
        return []
    out = []
    for m in NUMERIC_PATTERN.finditer(text):
        value = m.group(1)
        out.append({
            "value": value,
            "normalized": _normalize_num(value),
            "start": m.start(),
            "end": m.end(),
        })
    return out


def _evidence_contains_number(
    normalized: str,
    evidence_texts: list[str],
    tolerance: bool = True,
) -> tuple[bool, str]:
    """
    Check if any evidence chunk contains this number (or equivalent).
    Returns (found: bool, source_snippet: str or "").
    """
    # Exact substring in evidence
    for block in evidence_texts:
        block_norm = block.replace(",", " ")
        if normalized in block_norm:
            return True, block[:200]
        if tolerance and normalized.replace(".", "") in block.replace(",", "").replace(".", ""):
            return True, block[:200]
    # Also try matching normalized form as word boundary
    pattern = re.escape(normalized)
    for block in evidence_texts:
        if re.search(r"\b" + pattern + r"\b", block):
            return True, block[:200]
    return False, ""


def verify_numerics_against_evidence(
    draft: str,
    evidence_results: list[dict],
) -> dict[str, Any]:
    """
    Extract all numerics from draft and verify each against evidence.
    Numerics not found in any evidence chunk are flagged as potential hallucinations.

    Returns:
        dict with:
          - numerics: list of { value, normalized, start, end, verified, source_snippet }
          - total_numerics: int
          - verified_count: int
          - mismatch_list: list of { value, normalized, start, end } for unverified
          - numeric_accuracy: verified_count / total_numerics (0 if 0 total)
    """
    evidence_texts = [r.get("text", "").strip() for r in evidence_results if r.get("text")]
    extracted = extract_numeric_entities(draft)
    if not extracted:
        return {
            "numerics": [],
            "total_numerics": 0,
            "verified_count": 0,
            "mismatch_list": [],
            "numeric_accuracy": 1.0,
        }
    numerics = []
    mismatch_list = []
    for item in extracted:
        found, snippet = _evidence_contains_number(
            item["normalized"],
            evidence_texts,
            tolerance=True,
        )
        record = {
            "value": item["value"],
            "normalized": item["normalized"],
            "start": item["start"],
            "end": item["end"],
            "verified": found,
            "source_snippet": snippet if found else "",
        }
        numerics.append(record)
        if not found:
            mismatch_list.append({
                "value": item["value"],
                "normalized": item["normalized"],
                "start": item["start"],
                "end": item["end"],
            })
    total = len(numerics)
    verified = sum(1 for n in numerics if n["verified"])
    accuracy = verified / total if total else 1.0
    return {
        "numerics": numerics,
        "total_numerics": total,
        "verified_count": verified,
        "mismatch_list": mismatch_list,
        "numeric_accuracy": accuracy,
    }

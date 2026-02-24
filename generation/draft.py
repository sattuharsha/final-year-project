"""
MODULE 3 — Draft Generation.

Uses Gemini API to generate an evidence-grounded draft answer. Prompt enforces
that the model base its answer only on the provided evidence. Returns draft
text and metadata (query, evidence_count, model, etc.) for downstream
verification and evaluation.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"

# Enable caching by default (set USE_API_CACHE=false to disable)
USE_API_CACHE = os.environ.get("USE_API_CACHE", "true").lower() == "true"


def _format_evidence(evidence_results: list[dict]) -> str:
    """Format retrieved evidence chunks for the prompt."""
    parts = []
    for i, item in enumerate(evidence_results, 1):
        text = item.get("text", "").strip()
        if text:
            parts.append(f"[Evidence {i}]\n{text}")
    return "\n\n".join(parts) if parts else "(No evidence provided)"


def _build_evidence_prompt(query: str, evidence_text: str) -> str:
    """Build system-style prompt that enforces evidence-grounded answers."""
    return (
        "You are a factual assistant. Answer the question using ONLY the provided evidence. "
        "Write a clear, concise, and well-structured answer. "
        "If the evidence contains relevant information, synthesize it into a coherent response. "
        "If the evidence does not contain enough information to fully answer the question, "
        "state what can be determined from the evidence and note any gaps. "
        "Do not repeat phrases unnecessarily. Write naturally and avoid awkward phrasing.\n\n"
        "EVIDENCE:\n"
        f"{evidence_text}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "Answer:"
    )


def generate_draft(
    query: str,
    evidence_results: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """
    Generate a draft answer from Gemini conditioned on retrieved evidence.

    Args:
        query: User question or claim.
        evidence_results: List of dicts from dual_pass_retrieve (each has "text", "metadata").
        model: Gemini model name.
        temperature: Lower for more deterministic, evidence-following output.
        max_output_tokens: Cap on generated length.

    Returns:
        dict with keys:
          - draft: str (generated answer text)
          - metadata: dict (query, evidence_count, model, prompt_preview, etc.)
    """
    evidence_text = _format_evidence(evidence_results)
    prompt = _build_evidence_prompt(query, evidence_text)
    
    # Check cache first if enabled
    if USE_API_CACHE:
        try:
            from utils.api_cache import get_cached_response, cache_response
            cached = get_cached_response(prompt, model, temperature, max_output_tokens)
            if cached:
                return {
                    "draft": cached,
                    "metadata": {
                        "query": query,
                        "evidence_count": len(evidence_results),
                        "model": model,
                        "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        "cached": True,
                    },
                }
        except Exception:
            pass  # Fall through to API call if cache fails
    
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "draft": "",
            "metadata": {
                "query": query,
                "evidence_count": len(evidence_results),
                "model": model,
                "error": "GOOGLE_API_KEY or GEMINI_API_KEY not set",
            },
        }
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "draft": "",
            "metadata": {
                "query": query,
                "evidence_count": len(evidence_results),
                "model": model,
                "error": "google-genai not installed",
            },
        }
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        draft = (response.text or "").strip()
        
        # Cache successful response
        if USE_API_CACHE and draft:
            try:
                from utils.api_cache import cache_response
                cache_response(prompt, draft, model, temperature, max_output_tokens)
            except Exception:
                pass  # Continue even if caching fails
        
        if not draft:
            # Check if response was blocked or had issues
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = getattr(response.candidates[0], 'finish_reason', None)
                if finish_reason:
                    return {
                        "draft": "",
                        "metadata": {
                            "query": query,
                            "evidence_count": len(evidence_results),
                            "model": model,
                            "error": f"Generation blocked/failed: finish_reason={finish_reason}",
                        },
                    }
    except Exception as e:
        return {
            "draft": "",
            "metadata": {
                "query": query,
                "evidence_count": len(evidence_results),
                "model": model,
                "error": str(e),
            },
        }
    return {
        "draft": draft,
        "metadata": {
            "query": query,
            "evidence_count": len(evidence_results),
            "model": model,
            "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
        },
    }

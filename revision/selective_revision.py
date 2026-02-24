"""
MODULE 6 — Selective Revision.

Token-level (span-level) detection and correction:
- Detection: spans identify hallucinated regions.
- Correction: revise only hallucinated spans; retrieve at token/span level.
"""
from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"

# Enable caching by default (set USE_API_CACHE=false to disable)
USE_API_CACHE = os.environ.get("USE_API_CACHE", "true").lower() == "true"
# Skip polishing step to save API calls (set SKIP_POLISHING=true to disable polishing)
SKIP_POLISHING = os.environ.get("SKIP_POLISHING", "false").lower() == "true"
# Span-level correction only (retrieve at token/span level for each hallucinated span).
# Pass 2: re-retrieve evidence only for low-faithfulness segments (top_k per segment).
# Higher = more evidence for revision → better numeric accuracy and alignment (e.g. 8–10).
PASS2_RETRIEVE_TOP_K = int(os.environ.get("PASS2_RETRIEVE_TOP_K", "8"))
# Merge Pass 1 evidence with Pass 2 per-segment evidence so numbers/context from initial retrieval are kept.
MERGE_PASS1_PASS2_EVIDENCE = os.environ.get("MERGE_PASS1_PASS2_EVIDENCE", "true").lower() == "true"
# Max Pass 1 chunks to include when merging (keeps prompt size manageable).
PASS1_CHUNKS_IN_MERGE = int(os.environ.get("PASS1_CHUNKS_IN_MERGE", "5"))


def _retrieve_evidence_for_segment(query: str, segment: str, top_k: int = PASS2_RETRIEVE_TOP_K) -> list[dict]:
    """Re-retrieve evidence from KB only for this low-faithfulness segment (Pass 2 targeted retrieval)."""
    try:
        from rag.retrieval import search_knowledge_base
    except Exception:
        return []
    search_text = f"{query} {segment}".strip()
    ret = search_knowledge_base(search_text, top_k=min(top_k, 20))
    if ret.get("status") != "success":
        return []
    return ret.get("results", [])


def _format_evidence(evidence_results: list[dict]) -> str:
    """Format evidence for prompt."""
    parts = []
    for i, item in enumerate(evidence_results, 1):
        text = item.get("text", "").strip()
        if text:
            parts.append(f"[Evidence {i}]\n{text}")
    return "\n\n".join(parts) if parts else "(No evidence)"


def _build_revision_prompt(
    span_to_revise: str,
    query: str,
    evidence_text: str,
    context_before: str,
    context_after: str,
) -> str:
    """Build prompt for rewriting a single hallucinated span using evidence only."""
    return (
        "You are a factual editor. Rewrite the following span from a draft answer "
        "so it flows naturally with the surrounding context and is strictly supported by the evidence.\n"
        "IMPORTANT: (1) If the evidence contains numbers, dates, or facts that answer the question, "
        "include them exactly or paraphrase them—do not remove or replace with 'not specified'. "
        "(2) Use wording and key phrases FROM THE EVIDENCE when rewriting—paraphrase or reuse evidence "
        "text so the revised span stays semantically close to the evidence (this improves alignment). "
        "(3) Only if the evidence truly does not contain relevant information for this span, write a "
        "brief natural phrase or omit; avoid repeating 'Not stated in evidence'.\n\n"
        "EVIDENCE:\n"
        f"{evidence_text}\n\n"
        "QUESTION: " + query + "\n\n"
        "Context before: " + (context_before[-150:] if context_before else "(start)") + "\n\n"
        "Span to revise: " + span_to_revise + "\n\n"
        "Context after: " + (context_after[:150] if context_after else "(end)") + "\n\n"
        "Revised span (use evidence wording/phrases; preserve numbers/dates when present):"
    )


def _call_gemini(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Single Gemini call for revision snippet with caching support."""
    # Check cache first if enabled
    if USE_API_CACHE:
        try:
            from utils.api_cache import get_cached_response, cache_response
            cached = get_cached_response(prompt, model, temperature, max_tokens)
            if cached:
                return cached
        except Exception:
            pass  # Fall through to API call if cache fails
    
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return ""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens),
        )
        result = (response.text or "").strip()
        
        # Cache successful response
        if USE_API_CACHE and result:
            try:
                from utils.api_cache import cache_response
                cache_response(prompt, result, model, temperature, max_tokens)
            except Exception:
                pass  # Continue even if caching fails
        
        return result
    except Exception:
        return ""


def selective_revise(
    draft: str,
    faithfulness_result: dict[str, Any],
    evidence_results: list[dict],
    query: str,
    model: str = DEFAULT_MODEL,
    use_targeted_retrieval: bool = True,
) -> dict[str, Any]:
    """
    Span-level (token-level) detection and correction:
    - Detection: spans mark hallucinated regions.
    - Correction: revise only hallucinated spans; retrieve at token/span level for each.

    Returns:
        dict with: revised_answer, revisions_applied, span_revisions.
    """
    spans = faithfulness_result.get("spans", [])
    evidence_text_fallback = _format_evidence(evidence_results)

    if not spans:
        revised_answer = draft
        revisions_applied = 0
        span_revisions = []
    else:
        # Span-level correction (original behavior)
        revised_parts = []
        span_revisions = []
        for i, span_info in enumerate(spans):
            span_text = span_info.get("span", "")
            start = span_info.get("start_char", 0)
            end = span_info.get("end_char", len(span_text))
            context_before = draft[:start] if start <= len(draft) else ""
            context_after = draft[end:] if end <= len(draft) else ""
            if span_info.get("is_hallucinated"):
                # Pass 2: re-retrieve evidence only for this low-faithfulness segment; optionally merge with Pass 1
                if use_targeted_retrieval:
                    segment_evidence = _retrieve_evidence_for_segment(query, span_text, top_k=PASS2_RETRIEVE_TOP_K)
                    if MERGE_PASS1_PASS2_EVIDENCE and evidence_results:
                        pass1_sub = evidence_results[:PASS1_CHUNKS_IN_MERGE]
                        seen = {r.get("text", "").strip() for r in pass1_sub if r.get("text")}
                        merged = list(pass1_sub)
                        for r in (segment_evidence or []):
                            t = (r.get("text") or "").strip()
                            if t and t not in seen:
                                seen.add(t)
                                merged.append(r)
                        evidence_text = _format_evidence(merged) if merged else evidence_text_fallback
                    else:
                        evidence_text = _format_evidence(segment_evidence) if segment_evidence else evidence_text_fallback
                else:
                    evidence_text = evidence_text_fallback
                prompt = _build_revision_prompt(
                    span_text,
                    query,
                    evidence_text,
                    context_before,
                    context_after,
                )
                revised = _call_gemini(prompt, model=model)
                if not revised:
                    revised = span_text
                revised_parts.append(revised)
                span_revisions.append({
                    "original_span": span_text,
                    "revised_span": revised,
                    "index": i,
                })
            else:
                revised_parts.append(span_text)
        revised_answer = " ".join(revised_parts)
        revisions_applied = len(span_revisions)

    # Post-process: clean up common issues
    revised_answer = _cleanup_answer(revised_answer)

    # Final polishing: rewrite entire answer coherently if it has issues (skip if disabled)
    if not SKIP_POLISHING and revised_answer and len(revised_answer) > 50:
        polished = _polish_answer(revised_answer, query, evidence_results)
        if polished and len(polished) > 10:
            revised_answer = polished

    return {
        "revised_answer": revised_answer,
        "revisions_applied": revisions_applied,
        "span_revisions": span_revisions,
    }


def _cleanup_answer(text: str) -> str:
    """Clean up common formatting issues in the final answer."""
    import re
    if not text:
        return text
    
    # Remove markdown formatting artifacts
    text = re.sub(r'\*\*+', '', text)  # Remove **bold** markers
    
    # Remove repeated phrases with variations (e.g., "The evidence The provided evidence")
    patterns_to_fix = [
        (r'(The\s+(?:provided\s+)?evidence\s+)(?:The\s+(?:provided\s+)?evidence\s+)+', r'\1'),
        (r'(does\s+not\s+)(?:does\s+not\s+)+', r'\1'),
        (r'(The\s+provided\s+evidence\s+does\s+not\s+)(?:The\s+provided\s+evidence\s+does\s+not\s+)+', r'\1'),
        (r'(contain\s+)(?:contain\s+)+', r'\1'),
        (r'(determine\s+)(?:determine\s+)+', r'\1'),
    ]
    for pattern, replacement in patterns_to_fix:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove duplicate phrases (more aggressive)
    text = re.sub(r'\b(\w+(?:\s+\w+){0,4})\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    
    # Fix multiple "Not stated in evidence" or "does not contain" repetitions
    text = re.sub(r'(The\s+(?:provided\s+)?evidence\s+does\s+not\s+(?:contain|determine)[.\s]*){2,}', 
                  'The evidence does not contain sufficient information.', text, flags=re.IGNORECASE)
    
    # Clean up awkward word breaks (e.g., "end War II ended" -> "end of World War II")
    text = re.sub(r'\b(end|war|world)\s+(?:War|World|war|world)\s+(?:II|2)\s+(?:ended|end)\b', 
                  'World War II ended', text, flags=re.IGNORECASE)
    # Fix "7 World War" / "World War 7" typo (model slip) -> "Second World War" / "World War II"
    text = re.sub(r'\bthe\s+7\s+World\s+War\b', 'the Second World War', text, flags=re.IGNORECASE)
    text = re.sub(r'\b7\s+World\s+War\b', 'Second World War', text, flags=re.IGNORECASE)
    text = re.sub(r'\bWorld\s+War\s+7\b', 'World War II', text, flags=re.IGNORECASE)
    
    # Clean up excessive spacing
    text = re.sub(r'\s+', ' ', text)
    
    # Fix awkward punctuation spacing
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1\2', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def _polish_answer(answer: str, query: str, evidence_results: list[dict]) -> str:
    """Final polishing step: use Gemini to rewrite the entire answer coherently."""
    if not answer or len(answer) < 20:
        return answer
    
    evidence_text = _format_evidence(evidence_results)
    prompt = (
        "You are a text editor. Rewrite the following answer to be clear, concise, and well-structured. "
        "Remove all repetitions, awkward phrasing, and formatting issues. "
        "Use wording and key phrases from the EVIDENCE when possible so the polished answer stays "
        "semantically close to the evidence (preserves alignment). Preserve all factual information "
        "and numbers from the evidence.\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "EVIDENCE:\n"
        f"{evidence_text}\n\n"
        "CURRENT ANSWER (needs polishing):\n"
        f"{answer}\n\n"
        "POLISHED ANSWER (clear, natural, evidence-aligned wording, no repetitions):"
    )
    
    polished = _call_gemini(prompt, model=DEFAULT_MODEL)
    if polished and len(polished) > 10:
        return polished.strip()
    return answer  # Fallback to cleaned version if polishing fails

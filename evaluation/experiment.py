"""
MODULE 7 — Experiment Runner and Pipelines.

- baseline_pipeline(query): single-pass RAG → draft → metrics (no revision).
- drag_plus_pipeline(query): dual-pass retrieval → draft → verification → selective revision → metrics.
- run_experiment(dataset, n_samples): run both pipelines on dataset, return structured metrics and comparison.
- Save results to JSON for research evaluation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

# Suppress known multiprocess ResourceTracker shutdown noise on Windows (Python 3.12)
def _quiet_unraisable(hook):
    def wrapper(unraisable):
        try:
            exc_value = getattr(unraisable, "exc_value", None)
            exc_type = getattr(unraisable, "exc_type", None)
            msg = str(exc_value or "")
            # Suppress ResourceTracker / RLock recursion_count errors
            if ("_recursion_count" in msg or 
                "ResourceTracker" in msg or 
                (exc_type and "ResourceTracker" in str(exc_type)) or
                (exc_value and isinstance(exc_value, AttributeError) and "_recursion_count" in str(exc_value))):
                return  # Silently ignore
        except Exception:
            pass  # If suppression itself fails, ignore
        if hook is not None:
            hook(unraisable)
    return wrapper
_default_hook = getattr(sys, "unraisablehook", None)
sys.unraisablehook = _quiet_unraisable(_default_hook)

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Baseline: single-pass RAG + same verification/revision as Dual RAG+ so metrics are comparable.
# Uses same top_k as Dual RAG+ pass2 for comparable numeric accuracy; revision lowers hallucination.
# ---------------------------------------------------------------------------


def baseline_pipeline(query: str) -> dict[str, Any]:
    """
    Single-pass RAG: retrieve top-k, generate draft, then same verification + selective revision
    as Dual RAG+. Metrics computed on revised answer so hallucination rate and numeric accuracy
    are comparable across both models.
    """
    from rag.retrieval import search_knowledge_base
    from generation.draft import generate_draft
    from verification.token_faithfulness import compute_token_faithfulness
    from verification.numeric_verification import verify_numerics_against_evidence
    from revision.selective_revision import selective_revise
    from evaluation.metrics import compute_all_metrics

    # Same retrieval size as Dual RAG+ pass2 for comparable numeric accuracy.
    BASELINE_TOP_K = int(os.environ.get("BASELINE_TOP_K", "15"))
    ret = search_knowledge_base(query, top_k=BASELINE_TOP_K)
    if ret.get("status") != "success":
        evidence_results = []
    else:
        evidence_results = ret.get("results", [])

    gen = generate_draft(query, evidence_results)
    draft = gen.get("draft", "")
    draft_metadata = gen.get("metadata", {})

    if not draft:
        faithfulness_result = {}
        numeric_result = {}
        revision_result = {"revised_answer": "", "revisions_applied": 0, "span_revisions": []}
        answer_for_metrics = ""
    else:
        faithfulness_result = compute_token_faithfulness(draft, evidence_results)
        revision_result = selective_revise(
            draft,
            faithfulness_result,
            evidence_results,
            query,
        )
        revised_answer = revision_result.get("revised_answer", draft)
        answer_for_metrics = revised_answer if (revised_answer and revised_answer.strip()) else draft

    numeric_result = verify_numerics_against_evidence(answer_for_metrics, evidence_results)
    faith_for_metrics = compute_token_faithfulness(answer_for_metrics, evidence_results)
    metrics = compute_all_metrics(
        answer_for_metrics,
        evidence_results,
        faith_for_metrics,
        numeric_result,
    )

    return {
        "answer": revision_result.get("revised_answer", draft) if draft else "",
        "draft": draft,
        "evidence_results": evidence_results,
        "draft_metadata": draft_metadata,
        "faithfulness_result": faithfulness_result if draft else {},
        "numeric_result": numeric_result,
        "revision_result": revision_result,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Dual RAG+: dual-pass retrieval → draft → verification → selective revision → metrics.
# ---------------------------------------------------------------------------


def _pipeline_verbose() -> bool:
    """Whether to print faithfulness score in each pass. Default true; set PIPELINE_VERBOSE=false to disable."""
    return os.environ.get("PIPELINE_VERBOSE", "true").lower() != "false"


def drag_plus_pipeline(query: str) -> dict[str, Any]:
    """
    Dual RAG+ pipeline per proposed architecture:
    - User query -> Retrieve top-k -> Generate initial answer.
    - Verify numerical facts, validate against evidence, flag inconsistent/hallucinated quantities.
    - Pass 1: Compare each generated token/span with retrieved evidence; faithfulness score; identify low-faithfulness segments.
    - Pass 2: Re-retrieve evidence only for low-faithfulness segments; regenerate corrected spans.
    - Final answer + faithfulness score and metrics.

    Returns:
        dict with: answer (revised), draft, evidence_results, draft_metadata,
        faithfulness_result, numeric_result, revision_result, metrics (on revised answer).
    """
    from rag.dual_retrieval import dual_pass_retrieve
    from generation.draft import generate_draft
    from verification.token_faithfulness import compute_token_faithfulness
    from verification.numeric_verification import verify_numerics_against_evidence
    from revision.selective_revision import selective_revise
    from evaluation.metrics import compute_all_metrics

    verbose = _pipeline_verbose()

    # Retrieve top-k -> Generate initial answer
    ret = dual_pass_retrieve(
        query,
        pass1_top_k=30,
        pass2_top_k=15,
    )
    if ret.get("status") != "success":
        evidence_results = []
    else:
        evidence_results = ret.get("results", [])

    gen = generate_draft(query, evidence_results)
    draft = gen.get("draft", "")
    draft_metadata = gen.get("metadata", {})

    if verbose and draft:
        print("\n--- Pass 1: Initial answer ---")
        print(draft[:500] + ("..." if len(draft) > 500 else ""))

    # If draft generation failed, return early with error info (metrics all 0.0).
    if not draft and draft_metadata.get("error"):
        return {
            "answer": "",
            "draft": "",
            "evidence_results": evidence_results,
            "draft_metadata": draft_metadata,
            "faithfulness_result": {},
            "numeric_result": {},
            "revision_result": {},
            "metrics": {
                "hallucination_rate": 0.0,
                "faithfulness_score": 0.0,
                "numeric_accuracy": 0.0,
                "alignment_score": 0.0,
                "overall_faithfulness_score": 0.0,
            },
            "error": f"Draft generation failed: {draft_metadata.get('error')}",
        }

    # Numerical fact verification: validate against evidence, flag inconsistent/hallucinated quantities
    numeric_result = verify_numerics_against_evidence(draft, evidence_results)
    if verbose:
        v_count = numeric_result.get("verified_count", 0)
        total = numeric_result.get("total_numerics", 0)
        mismatch = numeric_result.get("mismatch_list", [])
        print("\n--- Numerical fact verification ---")
        print(f"  Verified: {v_count}/{total} numerics against evidence.")
        if mismatch:
            print("  Flagged inconsistent/hallucinated quantities:", [m.get("value") for m in mismatch])
        else:
            print("  No inconsistent or hallucinated quantities flagged.")

    # Pass 1: Token-level faithfulness — compare each span with evidence; identify low-faithfulness segments
    faithfulness_result = compute_token_faithfulness(draft, evidence_results)
    faith_mean = faithfulness_result.get("faithfulness_score_mean", 0.0)
    spans = faithfulness_result.get("spans", [])
    low_faith_spans = [s for s in spans if s.get("is_hallucinated")]
    if verbose:
        print("\n--- Pass 1: Faithfulness (compare each span with retrieved evidence) ---")
        print(f"  Faithfulness score (mean): {faith_mean:.4f}")
        print(f"  Low-faithfulness segments (hallucinated): {len(low_faith_spans)}")
        for i, s in enumerate(low_faith_spans[:5]):
            score = s.get("faithfulness_score", 0)
            text = (s.get("span", "") or "")[:80]
            print(f"    [{i+1}] score={score:.4f} | \"{text}...\"")
        if len(low_faith_spans) > 5:
            print(f"    ... and {len(low_faith_spans) - 5} more.")

    # Pass 2: Re-retrieve + revise (default: run revision; set DRAG_SKIP_REVISION=true to skip)
    skip_revision = os.environ.get("DRAG_SKIP_REVISION", "false").lower() == "true"
    if skip_revision:
        revision_result = {"revised_answer": draft, "revisions_applied": 0, "span_revisions": []}
        revised_answer = draft
        answer_for_metrics = draft
        metrics = compute_all_metrics(
            draft,
            evidence_results,
            faithfulness_result,
            numeric_result,
        )
        final_answer = draft
    else:
        revision_result = selective_revise(
            draft,
            faithfulness_result,
            evidence_results,
            query,
            use_targeted_retrieval=True,
        )
        revised_answer = revision_result.get("revised_answer", draft)
        n_revisions = revision_result.get("revisions_applied", 0)
        if verbose:
            print("\n--- Pass 2: Targeted re-retrieval + corrected spans ---")
            print(f"  Re-retrieved evidence for low-faithfulness segments; regenerated {n_revisions} corrected span(s).")
        answer_for_metrics = revised_answer if (revised_answer and revised_answer.strip()) else draft
        faith_revised = compute_token_faithfulness(answer_for_metrics, evidence_results)
        if verbose:
            faith_revised_mean = faith_revised.get("faithfulness_score_mean", 0.0)
            print(f"  Pass 2 faithfulness score (mean): {faith_revised_mean:.4f}")
        num_revised = verify_numerics_against_evidence(answer_for_metrics, evidence_results)
        metrics = compute_all_metrics(
            answer_for_metrics,
            evidence_results,
            faith_revised,
            num_revised,
        )
        draft_faith = compute_token_faithfulness(draft, evidence_results)
        draft_num = verify_numerics_against_evidence(draft, evidence_results)
        draft_metrics = compute_all_metrics(draft, evidence_results, draft_faith, draft_num)
        revised_overall = metrics.get("overall_faithfulness_score", 0.0)
        draft_overall = draft_metrics.get("overall_faithfulness_score", 0.0)
        revised_too_short = len((answer_for_metrics or "").strip()) < max(50, 0.25 * len((draft or "").strip()))
        if revised_too_short or revised_overall < draft_overall:
            answer_for_metrics = draft
            metrics = draft_metrics
            final_answer = draft
        else:
            final_answer = revised_answer if (revised_answer and revised_answer.strip()) else draft
    final_faith = metrics.get("faithfulness_score", 0.0)

    if verbose:
        print("\n--- Output: Final faithfulness score & answer ---")
        print(f"  Final faithfulness score: {final_faith:.4f}")
        print("  Final answer:")
        print(final_answer if final_answer else "(empty)")

    return {
        "answer": final_answer,
        "draft": draft,
        "evidence_results": evidence_results,
        "draft_metadata": draft_metadata,
        "faithfulness_result": faithfulness_result,
        "numeric_result": numeric_result,
        "revision_result": revision_result,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Experiment runner: dataset = list of {"query": str, "id": optional}.
# ---------------------------------------------------------------------------


def check_eval_queries_not_in_index(dataset: list[dict], top_k_check: int = 1) -> list[tuple[Any, str]]:
    """
    Check that evaluation queries do not appear in the knowledge base (no leakage).
    For each item with "query", retrieves top_k_check results; if any result text equals
    the query or contains the query as a full substring, that query is flagged as leaked.

    Returns:
        List of (sample_id or index, query) for queries that appear to be in the index.
    """
    from rag.retrieval import search_knowledge_base
    leaked = []
    for i, item in enumerate(dataset):
        query = (item.get("query") or "").strip()
        if not query:
            continue
        ret = search_knowledge_base(query, top_k=top_k_check)
        if ret.get("status") != "success" or not ret.get("results"):
            continue
        for r in ret.get("results", []):
            text = (r.get("text") or "").strip()
            if text == query or (query in text and len(query) > 20):
                leaked.append((item.get("id", i), query))
                break
    return leaked


def run_pipeline_on_dataset(
    dataset: list[dict],
    n_samples: int | None = None,
    pipeline_fn: Callable[[str], dict] | None = None,
) -> list[dict]:
    """
    Run a single pipeline on dataset (or first n_samples). Each item should have
    at least "query". Returns list of per-sample results with metrics.

    If pipeline_fn is None, defaults to drag_plus_pipeline.
    """
    if pipeline_fn is None:
        pipeline_fn = drag_plus_pipeline
    samples = dataset[:n_samples] if n_samples is not None else dataset
    results = []
    for i, item in enumerate(samples):
        query = item.get("query", "").strip()
        if not query:
            continue
        sample_id = item.get("id", i)
        try:
            out = pipeline_fn(query)
            out["sample_id"] = sample_id
            out["query"] = query
            results.append(out)
        except Exception as e:
            results.append({
                "sample_id": sample_id,
                "query": query,
                "error": str(e),
                "metrics": {},
            })
    return results


def _aggregate_metrics(results: list[dict]) -> dict[str, float]:
    """Average metrics over successful results."""
    metrics_list = [r.get("metrics") for r in results if r.get("metrics")]
    if not metrics_list:
        return {
            "hallucination_rate": 0.0,
            "faithfulness_score": 0.0,
            "numeric_accuracy": 0.0,
            "alignment_score": 0.0,
            "overall_faithfulness_score": 0.0,
        }
    keys = list(metrics_list[0].keys())
    agg = {}
    for k in keys:
        vals = [m.get(k, 0.0) for m in metrics_list if isinstance(m.get(k), (int, float))]
        agg[k] = sum(vals) / len(vals) if vals else 0.0
    return agg


def run_baseline_experiment(
    dataset: list[dict],
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Run baseline pipeline on dataset; return results + aggregated metrics."""
    results = run_pipeline_on_dataset(dataset, n_samples=n_samples, pipeline_fn=baseline_pipeline)
    agg = _aggregate_metrics(results)
    return {
        "pipeline": "baseline",
        "n_samples": len(results),
        "results": results,
        "aggregate_metrics": agg,
    }


def run_dual_rag_plus_experiment(
    dataset: list[dict],
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Run Dual RAG+ pipeline on dataset; return results + aggregated metrics."""
    results = run_pipeline_on_dataset(dataset, n_samples=n_samples, pipeline_fn=drag_plus_pipeline)
    agg = _aggregate_metrics(results)
    return {
        "pipeline": "dual_rag_plus",
        "n_samples": len(results),
        "results": results,
        "aggregate_metrics": agg,
    }


def run_experiment_full(
    dataset: list[dict],
    n_samples: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run full experiment: baseline and Dual RAG+ on dataset; return structured
    metrics and comparison table. Optionally save to JSON.

    Returns:
        dict with: comparison_table, comparison_rows, baseline_aggregate_metrics,
        dual_rag_plus_aggregate_metrics, baseline_n_samples, dual_rag_plus_n_samples,
        baseline_results_summary, dual_rag_plus_results_summary.
    """
    return _compare_baseline_vs_dual_rag_plus(
        dataset,
        n_samples=n_samples,
        output_path=output_path,
    )


def compare_baseline_vs_dual_rag_plus(
    dataset: list[dict],
    n_samples: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run both pipelines and return comparison; optionally save to JSON."""
    return _compare_baseline_vs_dual_rag_plus(
        dataset, n_samples=n_samples, output_path=output_path
    )


def _compare_baseline_vs_dual_rag_plus(
    dataset: list[dict],
    n_samples: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run both baseline and Dual RAG+ on the same dataset; return comparison
    table (aggregate metrics side-by-side) and full results. Optionally save to JSON.
    """
    if os.environ.get("CHECK_EVAL_LEAKAGE", "").lower() == "true":
        samples = dataset[:n_samples] if n_samples is not None else dataset
        leaked = check_eval_queries_not_in_index(samples)
        if leaked:
            print(
                f"\nWARNING: {len(leaked)} eval query(s) appear to be in the knowledge base (possible leakage):"
            )
            for sid, q in leaked[:5]:
                print(f"  id={sid}: {q[:80]}...")
            if len(leaked) > 5:
                print(f"  ... and {len(leaked) - 5} more.")
        else:
            print("\nCHECK_EVAL_LEAKAGE: no eval queries found in index (no leakage detected).")
    baseline_out = run_baseline_experiment(dataset, n_samples=n_samples)
    dual_out = run_dual_rag_plus_experiment(dataset, n_samples=n_samples)

    b_agg = baseline_out.get("aggregate_metrics", {})
    d_agg = dual_out.get("aggregate_metrics", {})

    comparison_table = {
        "metric": list(b_agg.keys()),
        "baseline": [b_agg.get(k, 0.0) for k in b_agg],
        "dual_rag_plus": [d_agg.get(k, 0.0) for k in b_agg],
    }
    # Also as row-oriented for readability.
    comparison_rows = [
        {
            "metric": k,
            "baseline": b_agg.get(k, 0.0),
            "dual_rag_plus": d_agg.get(k, 0.0),
        }
        for k in (b_agg.keys() | d_agg.keys())
    ]

    out = {
        "comparison_table": comparison_table,
        "comparison_rows": comparison_rows,
        "baseline_aggregate_metrics": b_agg,
        "dual_rag_plus_aggregate_metrics": d_agg,
        "baseline_n_samples": baseline_out.get("n_samples", 0),
        "dual_rag_plus_n_samples": dual_out.get("n_samples", 0),
        "baseline_results_summary": [
            {"sample_id": r.get("sample_id"), "query": r.get("query", "")[:80], "metrics": r.get("metrics", {})}
            for r in baseline_out.get("results", [])
        ],
        "dual_rag_plus_results_summary": [
            {"sample_id": r.get("sample_id"), "query": r.get("query", "")[:80], "metrics": r.get("metrics", {})}
            for r in dual_out.get("results", [])
        ],
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    return out


# Main API: run_experiment(dataset, n_samples) runs both pipelines and returns comparison.
run_experiment = run_experiment_full


def print_comparison_table(comparison: dict[str, Any]) -> None:
    """Print a formatted comparison table: Baseline vs Dual RAG+."""
    rows = comparison.get("comparison_rows", [])
    if not rows:
        return
    print("\n--- Baseline vs Dual RAG+ ---")
    print(f"{'Metric':<35} {'Baseline':>12} {'Dual RAG+':>12}")
    print("-" * 61)
    for r in rows:
        print(f"{r['metric']:<35} {r['baseline']:>12.4f} {r['dual_rag_plus']:>12.4f}")
    print("-" * 61)
    # Warn if Dual RAG+ shows all zeros (usually empty answers / API rate limit).
    d_agg = comparison.get("dual_rag_plus_aggregate_metrics", {})
    if d_agg and all(d_agg.get(k, 0.0) == 0.0 for k in ("faithfulness_score", "alignment_score", "overall_faithfulness_score")):
        print("\nNote: Dual RAG+ metrics are 0 — often due to empty answers (e.g. 429 rate limit).")
        print("      Wait for API limit reset or check draft_metadata.error in results.")

"""
Dual RAG+ Research Pipeline — Top-level API.

Exposes:
  - drag_plus_pipeline(query)  → full Dual RAG+ pipeline with verification and selective revision.
  - baseline_pipeline(query)   → single-pass RAG baseline.
  - run_experiment(dataset, n_samples=None, output_path=None) → run both, return comparison + save JSON.

Use for research evaluation and large experiment loops.
"""
from __future__ import annotations

import sys

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

from evaluation.experiment import (
    baseline_pipeline,
    drag_plus_pipeline,
    run_experiment,
    run_baseline_experiment,
    run_dual_rag_plus_experiment,
    compare_baseline_vs_dual_rag_plus,
    print_comparison_table,
)

__all__ = [
    "baseline_pipeline",
    "drag_plus_pipeline",
    "run_experiment",
    "run_baseline_experiment",
    "run_dual_rag_plus_experiment",
    "compare_baseline_vs_dual_rag_plus",
    "print_comparison_table",
]

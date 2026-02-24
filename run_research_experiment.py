"""
Run research experiment: Baseline vs Dual RAG+.

Usage (from project root):
  python run_research_experiment.py [n_samples] [output.json]

  n_samples: optional; limit dataset size (default 3 for quick test).
  output.json: optional; path to save results (default experiment_results.json).

Requires: index built (python build_index.py), GOOGLE_API_KEY in .env.
"""
from __future__ import annotations

import json
import os
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

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def get_sample_dataset(n: int = 5) -> list[dict]:
    """Minimal in-memory dataset of queries for testing (no HF/URL required)."""
    queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "When did World War II end?",
    ] + (["Name a country in Europe."] * max(0, n - 3))
    # Take first n queries
    queries = queries[:n]
    return [
        {"id": i, "query": q}
        for i, q in enumerate(queries)
    ]


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    output_path = sys.argv[2] if len(sys.argv) > 2 else "experiment_results.json"

    dataset = get_sample_dataset(n_samples)
    print(f"Running experiment on {len(dataset)} samples...")
    print("  Baseline (single-pass RAG) + Dual RAG+ (dual-pass + verification + revision)")

    from pipeline import run_experiment, print_comparison_table

    result = run_experiment(dataset, n_samples=None, output_path=output_path)
    print(f"\nResults saved to {output_path}")
    print_comparison_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

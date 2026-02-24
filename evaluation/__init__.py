"""
MODULE 7 — Evaluation Framework.

Metrics: hallucination rate, token-level faithfulness, numeric accuracy,
evidence alignment quality, overall faithfulness score.
run_experiment(dataset, n_samples), baseline single-pass RAG, comparison table.
"""
from evaluation.metrics import (
    compute_hallucination_rate,
    compute_faithfulness_score,
    compute_numeric_accuracy,
    compute_alignment_score,
    compute_overall_faithfulness,
    compute_all_metrics,
)
from evaluation.experiment import (
    run_experiment,
    run_experiment_full,
    baseline_pipeline,
    drag_plus_pipeline,
    run_baseline_experiment,
    run_dual_rag_plus_experiment,
    compare_baseline_vs_dual_rag_plus,
    print_comparison_table,
)

__all__ = [
    "compute_hallucination_rate",
    "compute_faithfulness_score",
    "compute_numeric_accuracy",
    "compute_alignment_score",
    "compute_overall_faithfulness",
    "compute_all_metrics",
    "run_experiment",
    "run_experiment_full",
    "baseline_pipeline",
    "drag_plus_pipeline",
    "run_baseline_experiment",
    "run_dual_rag_plus_experiment",
    "compare_baseline_vs_dual_rag_plus",
    "print_comparison_table",
]

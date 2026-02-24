{
  "comparison_table": {
    "metric": [
      "hallucination_rate",
      "faithfulness_score",
      "numeric_accuracy",
      "alignment_score",
      "overall_faithfulness_score"
    ],
    "baseline": [
      0.0962962962962963,
      0.4254439845780967,
      1.0,
      0.6537254221771065,
      0.697892305702166
    ],
    "dual_rag_plus": [
      0.06296296296296296,
      0.05411980402604991,
      0.2,
      0.11704614619296438,
      0.11875902351291412
    ]
  },
  "comparison_rows": [
    {
      "metric": "faithfulness_score",
      "baseline": 0.4254439845780967,
      "dual_rag_plus": 0.05411980402604991
    },
    {
      "metric": "numeric_accuracy",
      "baseline": 1.0,
      "dual_rag_plus": 0.2
    },
    {
      "metric": "hallucination_rate",
      "baseline": 0.0962962962962963,
      "dual_rag_plus": 0.06296296296296296
    },
    {
      "metric": "alignment_score",
      "baseline": 0.6537254221771065,
      "dual_rag_plus": 0.11704614619296438
    },
    {
      "metric": "overall_faithfulness_score",
      "baseline": 0.697892305702166,
      "dual_rag_plus": 0.11875902351291412
    }
  ],
  "baseline_aggregate_metrics": {
    "hallucination_rate": 0.0962962962962963,
    "faithfulness_score": 0.4254439845780967,
    "numeric_accuracy": 1.0,
    "alignment_score": 0.6537254221771065,
    "overall_faithfulness_score": 0.697892305702166
  },
  "dual_rag_plus_aggregate_metrics": {
    "hallucination_rate": 0.06296296296296296,
    "faithfulness_score": 0.05411980402604991,
    "numeric_accuracy": 0.2,
    "alignment_score": 0.11704614619296438,
    "overall_faithfulness_score": 0.11875902351291412
  },
  "baseline_n_samples": 10,
  "dual_rag_plus_n_samples": 10,
  "baseline_results_summary": [
    {
      "sample_id": 0,
      "query": "What is the capital of France?",
      "metrics": {
        "hallucination_rate": 0.6666666666666666,
        "faithfulness_score": 0.2622450465984197,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.5269161250615942,
        "overall_faithfulness_score": 0.5235147975748455
      }
    },
    {
      "sample_id": 1,
      "query": "Who wrote Romeo and Juliet?",
      "metrics": {
        "hallucination_rate": 0.2962962962962963,
        "faithfulness_score": 0.2869455499774451,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.6010654087153452,
        "overall_faithfulness_score": 0.6062528502264976
      }
    },
    {
      "sample_id": 2,
      "query": "When did World War II end?",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.4
      }
    },
    {
      "sample_id": 3,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.5293213213150145,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.7727532411420179,
        "overall_faithfulness_score": 0.7784507727457596
      }
    },
    {
      "sample_id": 4,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.5293213213150145,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.7727532411420179,
        "overall_faithfulness_score": 0.7784507727457596
      }
    },
    {
      "sample_id": 5,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.5293213213150145,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.7727532411420179,
        "overall_faithfulness_score": 0.7784507727457596
      }
    },
    {
      "sample_id": 6,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.5293213213150145,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.7727532411420179,
        "overall_faithfulness_score": 0.7784507727457596
      }
    },
    {
      "sample_id": 7,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.5293213213150145,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.7727532411420179,
        "overall_faithfulness_score": 0.7784507727457596
      }
    },
    {
      "sample_id": 8,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.5293213213150145,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.7727532411420179,
        "overall_faithfulness_score": 0.7784507727457596
      }
    },
    {
      "sample_id": 9,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.5293213213150145,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.7727532411420179,
        "overall_faithfulness_score": 0.7784507727457596
      }
    }
  ],
  "dual_rag_plus_results_summary": [
    {
      "sample_id": 0,
      "query": "What is the capital of France?",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 0.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.0
      }
    },
    {
      "sample_id": 1,
      "query": "Who wrote Romeo and Juliet?",
      "metrics": {
        "hallucination_rate": 0.2962962962962963,
        "faithfulness_score": 0.2869455499774451,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.6010654087153452,
        "overall_faithfulness_score": 0.6062528502264976
      }
    },
    {
      "sample_id": 2,
      "query": "When did World War II end?",
      "metrics": {
        "hallucination_rate": 0.3333333333333333,
        "faithfulness_score": 0.254252490283054,
        "numeric_accuracy": 1.0,
        "alignment_score": 0.5693960532142986,
        "overall_faithfulness_score": 0.5813373849026435
      }
    },
    {
      "sample_id": 3,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 0.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.0
      }
    },
    {
      "sample_id": 4,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 0.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.0
      }
    },
    {
      "sample_id": 5,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 0.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.0
      }
    },
    {
      "sample_id": 6,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 0.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.0
      }
    },
    {
      "sample_id": 7,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 0.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.0
      }
    },
    {
      "sample_id": 8,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 0.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.0
      }
    },
    {
      "sample_id": 9,
      "query": "Name a country in Europe.",
      "metrics": {
        "hallucination_rate": 0.0,
        "faithfulness_score": 0.0,
        "numeric_accuracy": 0.0,
        "alignment_score": 0.0,
        "overall_faithfulness_score": 0.0
      }
    }
  ]
}
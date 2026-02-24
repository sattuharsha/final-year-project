# RAG Agent with FEVER and RAGTruth (Google ADK)

This project uses **Google ADK** (Agent Development Kit) with **Gemini** to build a RAG (Retrieval-Augmented Generation) agent over the **FEVER** and **RAGTruth** datasets, downloaded from Hugging Face and GitHub.

## Setup

### Option A: One script (PowerShell, Windows)

From the project folder, run:

```powershell
.\setup_and_run.ps1
```

This creates `.venv`, installs requirements, builds the index (if needed), and runs a sample query. The first run can take several minutes (downloads FEVER, RAGTruth, and packages).

### Option B: Manual steps (virtual environment)

1. **Create and activate a virtual environment**:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # Windows PowerShell
   ```

   ```bash
   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies** (may take a few minutes):

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Gemini API key** in `.env`:

   ```
   GOOGLE_API_KEY=your_key_here
   ```

## Building the RAG index

Before using the agent, build the vector index from FEVER and RAGTruth:

```bash
python build_index.py
```

- **FEVER** is loaded from Hugging Face: `fever/fever` (v1.0).
- **RAGTruth** is loaded from the official GitHub repo (source_info.jsonl).  
  You can switch to Hugging Face by setting `RAGTRUTH_SOURCE=huggingface` and optionally `RAGTRUTH_HF_DATASET=jakobsnel/RAGTruth_Xtended`.

**Optional environment variables** (for `build_index.py`):

| Variable           | Default | Description                          |
|--------------------|---------|--------------------------------------|
| `FEVER_MAX`        | 5000    | Max FEVER documents to index         |
| `RAGTRUTH_MAX`     | 3000    | Max RAGTruth documents to index      |
| `RAGTRUTH_SOURCE`  | url     | `url` (GitHub) or `huggingface`       |
| `CHROMA_PATH`      | chroma_db | Directory for ChromaDB persistence |

The index is stored under `CHROMA_PATH` (default: `chroma_db/`).

## Running the agent

With the venv activated and the index built:

```bash
python run_agent.py "Your question here"
```

Or use your existing ADK app entrypoint (e.g. `adk run` or `adk web`). The agent is defined in `agent.py` and has:

- **Model**: `gemini-2.5-flash`
- **Tool**: `search_knowledge_base` — semantic search over the FEVER + RAGTruth index

The agent is instructed to call this tool for factual or verification questions and to answer from the retrieved passages.

## Dual RAG+ Research Pipeline (Research-Grade)

The project includes a **Dual RAG+** pipeline with hallucination mitigation and an evaluation framework for academic research.

### Architecture (modular)

| Module | Purpose |
|--------|--------|
| **1. Knowledge Base** | `knowledge/construction.py` — Clean, chunk (512/64), deduplicate; used by `build_index.py` |
| **2. Dual-Pass Retrieval** | `rag/dual_retrieval.py` — Pass 1: broad top-k; Pass 2: rerank + remove redundant chunks |
| **3. Draft Generation** | `generation/draft.py` — Gemini evidence-grounded draft; save draft + metadata |
| **4. Token-Level Faithfulness** | `verification/token_faithfulness.py` — Span↔evidence similarity; faithfulness score; heatmap |
| **5. Numerical Fact Verification** | `verification/numeric_verification.py` — Extract numerics; validate vs evidence; mismatch list |
| **6. Selective Revision** | `revision/selective_revision.py` — Rewrite only hallucinated spans; evidence-conditioned |
| **7. Evaluation** | `evaluation/metrics.py` + `evaluation/experiment.py` — Metrics, baseline vs Dual RAG+, JSON output |

### API (from project root)

```python
from pipeline import drag_plus_pipeline, baseline_pipeline, run_experiment

# Single query: Dual RAG+ (dual-pass retrieval → draft → verification → selective revision)
out = drag_plus_pipeline("Your question?")
# out["answer"], out["metrics"], out["faithfulness_result"], out["revision_result"], ...

# Single query: baseline single-pass RAG (retrieve → draft → metrics only)
out = baseline_pipeline("Your question?")
# out["answer"], out["metrics"], ...

# Full experiment: run both pipelines on a dataset; get comparison table + save JSON
dataset = [{"id": i, "query": q} for i, q in enumerate(["Q1?", "Q2?"])]
result = run_experiment(dataset, n_samples=10, output_path="results.json")
# result["comparison_rows"], result["baseline_aggregate_metrics"], result["dual_rag_plus_aggregate_metrics"]
```

### Metrics (evaluation)

- **hallucination_rate** = # hallucinated tokens / total tokens  
- **faithfulness_score** = average token-level faithfulness (span↔evidence similarity)  
- **numeric_accuracy** = verified numbers / total numbers  
- **alignment_score** = similarity(answer, evidence)  
- **overall_faithfulness_score** = weighted combination of the above  

### Run research experiment

```bash
python run_research_experiment.py [n_samples] [output.json]
```

Uses a small in-memory dataset by default; replace with your own `dataset` (list of `{"query": str, "id": optional}`) for large experiment loops. Results are saved to JSON for paper evaluation.

---

## Project layout

- `agent.py` — ADK agent with RAG tool
- `rag/retrieval.py` — ChromaDB + sentence-transformers retrieval and `search_knowledge_base`
- `rag/dual_retrieval.py` — Dual-pass retrieval for Dual RAG+
- `knowledge/construction.py` — Chunking, cleaning, deduplication for KB
- `generation/draft.py` — Evidence-grounded draft (Gemini)
- `verification/` — Token faithfulness + numeric verification
- `revision/selective_revision.py` — Selective revision of hallucinated spans
- `evaluation/` — Metrics, experiment runner, baseline vs Dual RAG+ comparison
- `pipeline.py` — Top-level API: `drag_plus_pipeline`, `baseline_pipeline`, `run_experiment`
- `data/load_datasets.py` — Load FEVER (Hugging Face) and RAGTruth (URL or HF)
- `build_index.py` — Script to build the vector index (with chunking + dedupe)
- `requirements.txt` — Dependencies (google-adk, chromadb, sentence-transformers, datasets, etc.)

## Datasets

- **FEVER** (Fact Extraction and VERification): claims with labels (SUPPORTS / REFUTES / NOT ENOUGH INFO); from [Hugging Face](https://huggingface.co/datasets/fever/fever).
- **RAGTruth**: source info (QA, summarization, data-to-text) from [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth); loaded via GitHub raw URL or optionally from Hugging Face.

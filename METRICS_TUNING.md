# Improving D-RAG Alignment, Numeric Accuracy, and Overall Faithfulness

If your Dual RAG+ (D-RAG) pipeline has **low alignment score**, **low numeric accuracy**, or **low overall faithfulness score**, use these knobs and practices.

---

## Token-Level Detection + Sentence-Level Correction (Default)

- **Token-level detection**: Uses finer spans (2 tokens when `TOKEN_LEVEL_DETECTION=true`) to identify hallucinated regions.
- **Sentence-level correction**: When `SENTENCE_LEVEL_CORRECTION=true`, revises **entire sentences** containing hallucinated spans (not just the span), producing more coherent output.

**Env vars:**
```env
TOKEN_LEVEL_DETECTION=true   # finer detection (span_size=2)
SENTENCE_LEVEL_CORRECTION=true   # revise whole sentences
```

Set `DRAG_SKIP_REVISION=false` in `.env` to enable revision (and thus sentence-level correction).

---

## Why They Can Be Low

- **Alignment** = max similarity between the **final answer** and **evidence chunks**. Short or generic revised text (e.g. "The evidence does not specify…") is less similar to evidence → lower alignment.
- **Numeric accuracy** = (numbers in the answer that appear in evidence) / (total numbers). If revision removes or changes numbers, this drops.
- **Overall faithfulness** = weighted combo of faithfulness + numeric + alignment + (1 − hallucination_rate). So all three above affect it.

Revision is tuned to **reduce hallucination**, which can make answers more conservative and hurt these metrics. The changes below keep hallucination low while improving alignment, numeric accuracy, and overall faithfulness.

---

## 1. Lower the Faithfulness Threshold (Fewer Spans Revised)

Fewer spans marked as "hallucinated" → less revision → more of the original (evidence-grounded) draft is kept → **higher alignment and faithfulness**.

**In `.env` or environment:**

```env
FAITHFULNESS_THRESHOLD=0.20
```

- Default: `0.25`. Try `0.20` or `0.18`.
- **Trade-off:** Slightly more spans may be left unrevised (hallucination rate can go up a bit).

---

## 2. Merge Pass 1 + Pass 2 Evidence (Already On)

Revision now **merges Pass 1 evidence with Pass 2 (targeted) evidence** for each low-faithfulness segment. That keeps numbers and context from the initial retrieval and improves **numeric accuracy** and **alignment**.

**Default:** `MERGE_PASS1_PASS2_EVIDENCE=true`. To turn off:

```env
MERGE_PASS1_PASS2_EVIDENCE=false
```

---

## 3. Increase Pass 2 Retrieval Size

More evidence per segment gives the model more to paraphrase and more numbers to preserve → **better alignment and numeric accuracy**.

**In `.env`:**

```env
PASS2_RETRIEVE_TOP_K=10
```

- Default: `8`. Try `10` or `12` (higher = more API cost if you re-run often).

---

## 4. Enable Polishing (If You Turned It Off)

The polishing step rewrites the full answer with evidence-aligned wording and can **raise alignment**.

**In `.env`:**

```env
SKIP_POLISHING=false
```

- If you had set `SKIP_POLISHING=true` to save quota, set it back to `false` when tuning for metrics.

---

## 5. Larger Span Size (Optional)

Larger spans often get higher similarity to evidence → fewer spans marked hallucinated → less revision.

**In `.env`:**

```env
SPAN_SIZE=6
```

- Default: `5`. Try `6` or `7`.

---

## 6. More Pass 1 Evidence (Dual RAG+ Retrieval)

The pipeline already uses dual-pass retrieval (e.g. pass1_top_k=30, pass2_top_k=15). If you reduced these, increasing them again gives more evidence for the draft and for metrics → can improve **faithfulness** and **alignment**.

In `evaluation/experiment.py`, `drag_plus_pipeline` uses `dual_pass_retrieve(..., pass1_top_k=30, pass2_top_k=15)`. You can raise these in code or add env vars if you expose them.

---

## Quick Tuning Preset for Higher Alignment / Numeric / Overall Faithfulness

Add to `.env`:

```env
FAITHFULNESS_THRESHOLD=0.20
PASS2_RETRIEVE_TOP_K=10
MERGE_PASS1_PASS2_EVIDENCE=true
SKIP_POLISHING=false
```

Then rebuild nothing; just re-run your experiment or `test_query.py`. Re-run a few times to see the effect; if hallucination rate goes up too much, increase `FAITHFULNESS_THRESHOLD` (e.g. back to `0.25`).

---

## Summary

| Goal | What to do |
|------|------------|
| **Higher alignment** | Lower `FAITHFULNESS_THRESHOLD`, enable polishing, increase `PASS2_RETRIEVE_TOP_K`, keep `MERGE_PASS1_PASS2_EVIDENCE=true`. |
| **Higher numeric accuracy** | Keep merge on, increase `PASS2_RETRIEVE_TOP_K`, lower threshold so fewer spans are revised away. |
| **Higher overall faithfulness** | Same as above (it’s a weighted combo of faithfulness, numeric, alignment, and hallucination). |
| **Keep hallucination low** | Don’t set `FAITHFULNESS_THRESHOLD` too low (e.g. keep ≥ 0.18). |

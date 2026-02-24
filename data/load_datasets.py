"""
Load FEVER and RAGTruth datasets from Hugging Face (and optional RAGTruth from GitHub).
Produces a list of documents (text + metadata) for RAG indexing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

# RAGTruth source_info.jsonl from GitHub (original corpus)
RAGTRUTH_SOURCE_INFO_URL = (
    "https://github.com/ParticleMedia/RAGTruth/raw/main/dataset/source_info.jsonl"
)

# FEVER v1.0 from fever.ai (JSONL; HF dataset uses deprecated loading script)
FEVER_TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"


def load_fever_from_url(
    url: str = FEVER_TRAIN_URL,
    max_examples: int | None = None,
) -> Iterator[dict]:
    """
    Load FEVER dataset from fever.ai JSONL URL.
    Yields documents with 'text' and 'metadata' (claim, label, id, source='fever').
    """
    logger.info("Loading FEVER from %s ...", url)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    count = 0
    for line in r.text.strip().split("\n"):
        if not line:
            continue
        if max_examples is not None and count >= max_examples:
            break
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        claim = row.get("claim", "")
        label = row.get("label", "")
        eid = row.get("id")
        if not claim:
            continue
        # Index only the claim; do NOT include label in text to avoid evaluation leakage.
        text = claim
        yield {
            "text": text,
            "metadata": {
                "source": "fever",
                "id": eid,
                "label": label,
                "claim": claim[:500],
            },
        }
        count += 1
    logger.info("FEVER: yielded %s documents", count)


def load_fever_from_huggingface(
    split: str = "train",
    max_examples: int | None = None,
    trust_remote_code: bool = False,
) -> Iterator[dict]:
    """
    Load FEVER from Hugging Face if available (no script). Else fall back to fever.ai URL.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    try:
        logger.info("Loading FEVER (fever/fever, v1.0) from Hugging Face...")
        ds = load_dataset(
            "fever/fever",
            "v1.0",
            split=split,
            trust_remote_code=trust_remote_code,
        )
        count = 0
        for row in ds:
            if max_examples is not None and count >= max_examples:
                break
            claim = row.get("claim", "")
            label = row.get("label", "")
            eid = row.get("id")
            if not claim:
                continue
            # Index only the claim; do NOT include label in text to avoid evaluation leakage.
            text = claim
            yield {
                "text": text,
                "metadata": {
                    "source": "fever",
                    "id": eid,
                    "label": label,
                    "claim": claim[:500],
                },
            }
            count += 1
        logger.info("FEVER: yielded %s documents", count)
    except RuntimeError as e:
        if "scripts are no longer supported" in str(e) or "Dataset scripts" in str(e):
            logger.info("FEVER HF script not supported, falling back to fever.ai URL.")
            yield from load_fever_from_url(max_examples=max_examples)
        else:
            raise


def load_ragtruth_from_url(
    url: str = RAGTRUTH_SOURCE_INFO_URL,
    max_examples: int | None = None,
) -> Iterator[dict]:
    """
    Load RAGTruth source_info from a URL (e.g. GitHub raw).
    Each line is JSON with source_id, task_type, source_info, prompt.
    We index source_info content (passages, question, or raw text).
    """
    logger.info("Loading RAGTruth from %s ...", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    count = 0
    for line in r.text.strip().split("\n"):
        if not line:
            continue
        if max_examples is not None and count >= max_examples:
            break
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        source_id = obj.get("source_id", "")
        task_type = obj.get("task_type", "")
        source_info = obj.get("source_info")
        prompt = obj.get("prompt", "") or ""
        if source_info is None:
            continue
        if isinstance(source_info, dict):
            # QA: question + passages; Data2txt: structured fields
            question = source_info.get("question", "")
            passages = source_info.get("passages", "")
            if question and passages:
                text = f"Question: {question}\nPassages: {passages}"
            else:
                text = json.dumps(source_info)[:8000]
        else:
            text = str(source_info)[:8000]
        if not text.strip():
            continue
        yield {
            "text": text,
            "metadata": {
                "source": "ragtruth",
                "source_id": source_id,
                "task_type": task_type,
                "prompt_preview": prompt[:200] if prompt else "",
            },
        }
        count += 1
    logger.info("RAGTruth: yielded %s documents", count)


def load_ragtruth_from_huggingface(
    dataset_name: str = "jakobsnel/RAGTruth_Xtended",
    split: str = "train",
    max_examples: int | None = None,
) -> Iterator[dict]:
    """
    Load RAGTruth-related data from Hugging Face (e.g. RAGTruth_Xtended).
    Uses 'question', 'response', or 'source_info' columns if present.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    logger.info("Loading %s from Hugging Face...", dataset_name)
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.warning("Could not load %s: %s. Use RAGTruth from URL instead.", dataset_name, e)
        return
    count = 0
    for row in ds:
        if max_examples is not None and count >= max_examples:
            break
        text = None
        if "source_info" in row and row["source_info"]:
            text = row["source_info"] if isinstance(row["source_info"], str) else json.dumps(row["source_info"])[:8000]
        if not text and "question" in row and row["question"]:
            text = row["question"]
        if not text and "response" in row and row["response"]:
            text = row["response"]
        if not text:
            continue
        yield {
            "text": text[:8000],
            "metadata": {
                "source": "ragtruth_hf",
                "dataset": dataset_name,
                "id": row.get("id", count),
            },
        }
        count += 1
    logger.info("RAGTruth (HF): yielded %s documents", count)


def load_fever_eval_claims(
    split: str = "test",
    max_examples: int | None = None,
) -> Iterator[dict]:
    """
    Load FEVER claims for evaluation only (no indexing).
    Use split='test' or 'dev' so these claims are disjoint from the indexed train split.
    Ensures evaluation claims were not in the knowledge base: index with FEVER_SPLIT=train,
    then evaluate on FEVER test/dev from this loader.
    Yields dicts with keys: id, query (claim), label.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")
    logger.info("Loading FEVER eval claims (split=%s) from Hugging Face...", split)
    ds = load_dataset("fever/fever", "v1.0", split=split, trust_remote_code=False)
    count = 0
    for row in ds:
        if max_examples is not None and count >= max_examples:
            break
        claim = row.get("claim", "").strip()
        if not claim:
            continue
        yield {
            "id": row.get("id", count),
            "query": claim,
            "label": row.get("label", ""),
        }
        count += 1
    logger.info("FEVER eval: yielded %s claims", count)


def load_all_documents(
    fever_split: str = "train",
    fever_max: int | None = 5000,
    ragtruth_source: str = "url",
    ragtruth_max: int | None = 3000,
    ragtruth_hf_dataset: str | None = "jakobsnel/RAGTruth_Xtended",
) -> Iterator[dict]:
    """
    Load both FEVER and RAGTruth and yield unified documents for indexing.
    - fever_max: cap FEVER documents (None = all).
    - ragtruth_source: 'url' (GitHub) or 'huggingface'.
    - ragtruth_hf_dataset: used if ragtruth_source == 'huggingface'.
    """
    for doc in load_fever_from_huggingface(split=fever_split, max_examples=fever_max):
        yield doc
    if ragtruth_source == "url":
        for doc in load_ragtruth_from_url(max_examples=ragtruth_max):
            yield doc
    elif ragtruth_source == "huggingface" and ragtruth_hf_dataset:
        for doc in load_ragtruth_from_huggingface(
            dataset_name=ragtruth_hf_dataset,
            max_examples=ragtruth_max,
        ):
            yield doc

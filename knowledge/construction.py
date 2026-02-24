"""
MODULE 1 — Knowledge Base Construction.

- Clean document text (normalize whitespace, strip noise).
- Chunk long documents into fixed-size overlapping spans for dense retrieval.
- Deduplicate by content hash to avoid storing identical chunks.
- Output: list of {"text", "metadata"} ready for ChromaDB indexing.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterator

# Default chunk size (chars) and overlap for semantic coherence across boundaries.
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


def clean_text(text: str) -> str:
    """
    Normalize document text: collapse whitespace, strip leading/trailing,
    remove null bytes and excessive newlines.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict | None = None,
) -> list[dict]:
    """
    Split a single document into overlapping chunks. Each chunk has "text" and
    "metadata" (with chunk_index, optional parent metadata).
    Optimized: faster boundary detection, prevents infinite loops.
    """
    text = clean_text(text)
    if not text:
        return []
    # For very short texts, return single chunk
    if len(text) <= chunk_size:
        meta = dict(metadata or {})
        return [{"text": text, "metadata": {**meta, "chunk_index": 0}}]
    
    meta = dict(metadata or {})
    chunks = []
    start = 0
    idx = 0
    max_iterations = len(text) // max(1, chunk_size - chunk_overlap) + 10  # Safety limit
    iterations = 0
    
    while start < len(text) and iterations < max_iterations:
        iterations += 1
        end = min(start + chunk_size, len(text))
        
        # Prefer breaking on sentence or word boundary (optimized: only check if needed)
        if end < len(text) and chunk_size > 20:  # Skip boundary detection for tiny chunks
            # Quick check: try sentence boundary first (most common)
            last_dot = text.rfind(". ", start, end + 1)
            if last_dot >= start:
                end = last_dot + 2
            else:
                # Fallback to other boundaries
                for sep in ("? ", "! ", "\n", " "):
                    last = text.rfind(sep, start, end + 1)
                    if last >= start:
                        end = last + len(sep)
                        break
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_meta = {**meta, "chunk_index": idx}
            chunks.append({"text": chunk_text, "metadata": chunk_meta})
            idx += 1
        
        # Ensure progress: move forward by at least (chunk_size - overlap)
        new_start = end if end > start else start + chunk_size
        # Apply overlap, but ensure we make progress
        if new_start > start:
            start = max(start + 1, new_start - chunk_overlap)
        else:
            start = new_start  # Force progress if no movement
    
    return chunks


def chunk_documents(
    documents: Iterator[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[dict]:
    """
    Yield chunked documents from an iterator of {"text", "metadata"}.
    Each yielded item is a single chunk with merged metadata.
    Note: This is kept for compatibility; prepare_documents_for_indexing
    now does chunking inline for better performance.
    """
    for doc in documents:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        for chunk in chunk_document(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=meta,
        ):
            yield chunk


def _content_hash(text: str) -> str:
    """Stable hash of normalized content for deduplication."""
    # Use faster hash for deduplication (text already cleaned during chunking)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def deduplicate_documents(documents: list[dict]) -> list[dict]:
    """
    Remove duplicate chunks by content hash. Preserves first occurrence order.
    Optimized: uses MD5 instead of SHA256 for speed, assumes text already cleaned.
    """
    seen: set[str] = set()
    out = []
    for doc in documents:
        text = doc.get("text", "")
        # Skip empty chunks
        if not text.strip():
            continue
        h = _content_hash(text)
        if h in seen:
            continue
        seen.add(h)
        out.append(doc)
    return out


def prepare_documents_for_indexing(
    documents: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    dedupe: bool = True,
) -> list[dict]:
    """
    Full pipeline: clean and chunk all documents, then deduplicate.
    Returns list of {"text", "metadata"} ready for ChromaDB add_documents.
    """
    import time
    # Chunk documents (with progress feedback for large datasets)
    total = len(documents)
    if total > 100:
        print(f"Chunking {total} documents...", end="", flush=True)
    chunked = []
    start_time = time.time()
    for i, doc in enumerate(documents):
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        chunks = chunk_document(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=meta,
        )
        chunked.extend(chunks)
        # Progress feedback every 50 docs (more frequent)
        if total > 100 and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f". ({i+1}/{total}, {rate:.1f} docs/sec)", end="", flush=True)
    if total > 100:
        elapsed = time.time() - start_time
        print(f"\n  Created {len(chunked)} chunks in {elapsed:.1f}s.")
    
    # Deduplicate if requested
    if dedupe and chunked:
        if len(chunked) > 100:
            print(f"Deduplicating {len(chunked)} chunks...", end="", flush=True)
            start_time = time.time()
        chunked = deduplicate_documents(chunked)
        if len(chunked) > 100:
            elapsed = time.time() - start_time
            print(f" {len(chunked)} unique chunks remaining ({elapsed:.1f}s).")
    
    return chunked

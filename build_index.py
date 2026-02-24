"""
Build the RAG vector index from FEVER and RAGTruth datasets.
Run once (or when you want to refresh the index) before using the agent.

  python build_index.py

Options (env or edit below):
  FEVER_SPLIT     - FEVER split to index: "train" (default). Use train so eval can use test/dev.
  FEVER_MAX       - cap FEVER documents (default 1000)
  RAGTRUTH_MAX    - cap RAGTruth documents (default 500)
  RAGTRUTH_SOURCE - "url" (GitHub) or "huggingface"
  CHROMA_PATH     - path for ChromaDB (default chroma_db)
  CHUNK_SIZE      - chunk size in chars (default 512)
  CHUNK_OVERLAP   - chunk overlap (default 64)

Documents are fed one-by-one: chunk each, global dedupe by content hash, assign globally unique IDs (doc_0, doc_1, ...).
"""
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.load_datasets import load_all_documents
from knowledge.construction import chunk_document, _content_hash
from rag.retrieval import (
    DEFAULT_CHROMA_PATH,
    COLLECTION_NAME,
    get_chroma_client,
    get_collection,
    add_documents,
)


def main():
    fever_max = int(os.environ.get("FEVER_MAX", "1000"))
    ragtruth_max = int(os.environ.get("RAGTRUTH_MAX", "500"))
    ragtruth_source = os.environ.get("RAGTRUTH_SOURCE", "url")
    chroma_path = os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH)

    # Index only FEVER *train* so evaluation can use test/dev without leakage.
    fever_split = os.environ.get("FEVER_SPLIT", "train")
    chunk_size = int(os.environ.get("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "64"))

    # Ensure client uses desired path, then reset collection
    client = get_chroma_client(chroma_path)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    # Force new collection on next get_collection()
    import rag.retrieval as mod
    mod._collection = None

    print("Indexing: one document at a time with global dedupe and unique IDs...")
    seen_hashes = set()
    next_id = 0
    doc_count = 0
    chunk_count = 0
    skipped_dupes = 0

    for doc in load_all_documents(
        fever_split=fever_split,
        fever_max=fever_max,
        ragtruth_max=ragtruth_max,
        ragtruth_source=ragtruth_source,
    ):
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        chunks = chunk_document(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=meta,
        )
        to_add = []
        ids_to_add = []
        for c in chunks:
            t = c.get("text", "").strip()
            if not t:
                continue
            h = _content_hash(t)
            if h in seen_hashes:
                skipped_dupes += 1
                continue
            seen_hashes.add(h)
            to_add.append(c)
            ids_to_add.append(f"doc_{next_id}")
            next_id += 1
        if to_add:
            add_documents(to_add, ids=ids_to_add)
            chunk_count += len(to_add)
        doc_count += 1
        if doc_count % 100 == 0 and doc_count > 0:
            print(f"  Processed {doc_count} documents, {chunk_count} chunks added, {skipped_dupes} duplicates skipped.")

    if doc_count == 0:
        print("No documents to index. Check data sources (HF for FEVER, URL/HF for RAGTruth).")
        return 1

    coll = get_collection()
    print(f"Done. Processed {doc_count} documents; {chunk_count} chunks in collection '{COLLECTION_NAME}', {skipped_dupes} duplicates skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

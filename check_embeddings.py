"""Quick check: verify embedding function works for metrics (faithfulness + alignment)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress ResourceTracker / RLock shutdown noise at exit
def _quiet_unraisable(hook):
    def wrapper(unraisable):
        try:
            exc_value = getattr(unraisable, "exc_value", None)
            msg = str(exc_value or "")
            if "_recursion_count" in msg or "ResourceTracker" in msg:
                return
        except Exception:
            pass
        if hook is not None:
            hook(unraisable)
    return wrapper
_default_hook = getattr(sys, "unraisablehook", None)
sys.unraisablehook = _quiet_unraisable(_default_hook)

def main():
    from rag.retrieval import get_embedding_function
    embed_fn = get_embedding_function()
    texts = ["Paris is the capital of France.", "The capital of France is Paris."]
    try:
        embs = embed_fn(texts)
        print(f"Embedding output type: {type(embs)}, len: {len(embs)}")
        if embs:
            first = embs[0]
            try:
                n = len(first)
            except Exception:
                n = 0
            print(f"First embedding len: {n}")
            # Cosine sim between same-meaning sentences should be high (convert to list for numpy safety)
            a = list(first) if hasattr(first, '__iter__') and not isinstance(first, str) else first
            b = list(embs[1]) if hasattr(embs[1], '__iter__') and not isinstance(embs[1], str) else embs[1]
            dot = sum(x*y for x,y in zip(a,b))
            na = (sum(x*x for x in a)**0.5) or 1e-12
            nb = (sum(x*x for x in b)**0.5) or 1e-12
            sim = dot / (na * nb)
            print(f"Similarity (Paris... vs The capital...): {sim:.4f}")
        print("OK: Embeddings work.")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

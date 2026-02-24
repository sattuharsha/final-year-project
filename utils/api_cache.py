"""
API Response Caching to Reduce Quota Usage.

Caches Gemini API responses based on prompt hash to avoid redundant calls.
Useful when running experiments with repeated queries or similar prompts.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(__file__).parent.parent / ".api_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _hash_prompt(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Generate cache key from prompt and config."""
    key_data = f"{prompt}|{model}|{temperature}|{max_tokens}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def get_cached_response(
    prompt: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Get cached response if available."""
    cache_key = _hash_prompt(prompt, model, temperature, max_tokens)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("response", "")
        except Exception:
            return None
    return None


def cache_response(
    prompt: str,
    response: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> None:
    """Cache API response."""
    cache_key = _hash_prompt(prompt, model, temperature, max_tokens)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt[:200], "response": response, "model": model}, f, indent=2)
    except Exception:
        pass  # Fail silently if cache write fails


def clear_cache() -> None:
    """Clear all cached responses."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception:
                pass


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics."""
    if not CACHE_DIR.exists():
        return {"count": 0, "size_bytes": 0}
    
    cache_files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files if f.exists())
    return {"count": len(cache_files), "size_bytes": total_size}

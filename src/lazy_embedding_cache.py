"""
lazy_embedding_cache.py - On-Demand Embedding Cache Management

Responsible for:
    - Loading embedding model only when first needed
    - Caching embeddings for repeated access
    - Persisting cache to disk to avoid re-computation
    - Providing translator-facing API for embedding retrieval
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_embedding_model = None
_model_available = None
_embedding_cache = {}


def _load_model(model_name: str = "all-MiniLM-L6-v2") -> bool:
    """Attempt to load the SentenceTransformers model once."""
    global _embedding_model, _model_available

    if _model_available is not None:
        return _model_available

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model on-demand: {model_name} …")
        _embedding_model = SentenceTransformer(model_name)
        _model_available = True
        logger.info("Embedding model loaded and ready for use.")
    except Exception as e:
        logger.warning(
            f"Embedding model could not be loaded ({e}). "
            "Embeddings will be unavailable for queries."
        )
        _model_available = False

    return _model_available


class LazyEmbeddingCache:
    """
    On-demand embedding cache with transparent persistence.

    Usage:
        cache = LazyEmbeddingCache("data/embedding_cache.json")
        embedding = cache.get_embedding("machine learning context")
        # First call: loads model and computes embedding
        # Subsequent calls: retrieved from in-memory or disk cache
    """

    def __init__(self, cache_file: str = "data/embedding_cache.json", model_name: str = "all-MiniLM-L6-v2"):
        self.cache_file = Path(cache_file)
        self.model_name = model_name
        self._in_memory_cache: Dict[str, List[float]] = {}
        self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        """Load cached embeddings from disk if file exists."""
        if not self.cache_file.exists():
            logger.info(f"No embedding cache file at {self.cache_file}; starting fresh.")
            return

        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                self._in_memory_cache = data.get("embeddings", {})
                logger.info(f"Loaded {len(self._in_memory_cache)} cached embeddings from disk.")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}. Starting fresh.")
            self._in_memory_cache = {}

    def _save_disk_cache(self) -> None:
        """Persist in-memory cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(
                    {
                        "model_name": self.model_name,
                        "embeddings": self._in_memory_cache,
                        "cache_size": len(self._in_memory_cache),
                    },
                    f,
                    indent=2,
                )
            logger.debug(f"Embedding cache saved ({len(self._in_memory_cache)} entries).")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text, computing on-demand if needed.

        Returns:
            List of floats representing the embedding vector, or None if model unavailable.
        """
        # Check in-memory cache first
        if text in self._in_memory_cache:
            logger.debug(f"Cache hit: {text[:50]}…")
            return self._in_memory_cache[text]

        # Load model if needed
        if not _load_model(self.model_name):
            logger.warning("Cannot embed: model unavailable.")
            return None

        # Compute embedding
        try:
            embedding = _embedding_model.encode(text, convert_to_numpy=False).tolist()
            self._in_memory_cache[text] = embedding
            logger.debug(f"Computed embedding: {text[:50]}… (cache size now: {len(self._in_memory_cache)})")
            return embedding
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None

    def batch_embed(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """Batch embed multiple texts efficiently."""
        results = {}
        texts_to_compute = []
        text_indices = {}

        # Check cache for what's already computed
        for i, text in enumerate(texts):
            if text in self._in_memory_cache:
                results[text] = self._in_memory_cache[text]
            else:
                text_indices[len(texts_to_compute)] = text
                texts_to_compute.append(text)

        if not texts_to_compute:
            logger.debug(f"Batch embed: all {len(texts)} texts in cache.")
            return results

        # Load model if needed
        if not _load_model(self.model_name):
            logger.warning("Cannot embed batch: model unavailable.")
            return results

        # Batch compute
        try:
            embeddings = _embedding_model.encode(texts_to_compute, convert_to_numpy=False).tolist()
            for idx, embedding in enumerate(embeddings):
                text = text_indices[idx]
                results[text] = embedding
                self._in_memory_cache[text] = embedding
            logger.info(f"Batch embed: computed {len(embeddings)} new embeddings (cache size now: {len(self._in_memory_cache)})")
        except Exception as e:
            logger.warning(f"Failed to batch compute embeddings: {e}")

        return results

    def persist(self) -> None:
        """Explicitly save cache to disk."""
        self._save_disk_cache()

    def clear_disk_cache(self) -> None:
        """Remove the disk cache file."""
        if self.cache_file.exists():
            os.remove(self.cache_file)
            logger.info(f"Cleared disk cache at {self.cache_file}")

    def get_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "in_memory_entries": len(self._in_memory_cache),
            "cache_file_exists": self.cache_file.exists(),
        }

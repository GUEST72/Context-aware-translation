"""
embedding_generator.py - Semantic Embedding Generation Module

Responsible for:
    - Generating dense vector embeddings for each term
    - Using SentenceTransformers models
    - Providing graceful fallback when the model is unavailable
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_embedding_model = None
_model_available = None


def _load_model(model_name: str = "all-MiniLM-L6-v2") -> bool:
    """Attempt to load the SentenceTransformers model once."""
    global _embedding_model, _model_available

    if _model_available is not None:
        return _model_available

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name} …")
        _embedding_model = SentenceTransformer(model_name)
        _model_available = True
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.warning(
            f"Embedding model could not be loaded ({e}). "
            "Embeddings will be empty."
        )
        _model_available = False

    return _model_available


class EmbeddingGenerator:
    """
    Generates semantic embeddings for terminology terms.

    Uses SentenceTransformers (all-MiniLM-L6-v2 by default).
    Falls back to empty lists if the model is unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", enabled: bool = True):
        """
        Args:
            model_name: HuggingFace model identifier for SentenceTransformers.
            enabled:    If False, skip embedding generation entirely.
        """
        self.model_name = model_name
        self.enabled = enabled
        self.cache: Dict[str, List[float]] = {}

    def generate(self, term: str) -> List[float]:
        """
        Generate an embedding vector for a single term.

        Returns:
            A list of floats (the embedding), or an empty list on failure.
        """
        if not self.enabled:
            return []

        key = term.lower().strip()
        if key in self.cache:
            return self.cache[key]

        if not _load_model(self.model_name) or _embedding_model is None:
            return []

        try:
            vector = _embedding_model.encode(term).tolist()
            self.cache[key] = vector
            return vector
        except Exception as e:
            logger.warning(f"Embedding generation failed for '{term}': {e}")
            return []

    def generate_batch(self, terms: List[str]) -> Dict[str, List[float]]:
        """
        Generate embeddings for a list of terms (batched for efficiency).

        Returns:
            dict mapping term -> embedding vector (list of floats).
        """
        if not self.enabled:
            return {t: [] for t in terms}

        if not _load_model(self.model_name) or _embedding_model is None:
            return {t: [] for t in terms}

        # Separate cached from uncached
        results: Dict[str, List[float]] = {}
        to_encode: List[str] = []
        for term in terms:
            key = term.lower().strip()
            if key in self.cache:
                results[term] = self.cache[key]
            else:
                to_encode.append(term)

        if to_encode:
            try:
                vectors = _embedding_model.encode(to_encode).tolist()
                for term, vec in zip(to_encode, vectors):
                    key = term.lower().strip()
                    self.cache[key] = vec
                    results[term] = vec
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")
                for term in to_encode:
                    results[term] = []

        return results

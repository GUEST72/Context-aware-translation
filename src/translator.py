"""
translator.py - Term Translation Module (English → Arabic)

Responsible for:
    - Translating individual terms (NOT full paragraphs)
    - Using the Helsinki-NLP/opus-mt-en-ar model via HuggingFace Transformers
    - Providing a graceful fallback when the model is unavailable
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_model = None
_tokenizer = None
_model_available = None  # None = not checked yet


def _load_model() -> bool:
    """Attempt to load the translation model once."""
    global _model, _tokenizer, _model_available

    if _model_available is not None:
        return _model_available

    try:
        from transformers import MarianMTModel, MarianTokenizer

        model_name = "Helsinki-NLP/opus-mt-en-ar"
        logger.info(f"Loading translation model: {model_name} …")
        _tokenizer = MarianTokenizer.from_pretrained(model_name)
        _model = MarianMTModel.from_pretrained(model_name)
        _model_available = True
        logger.info("Translation model loaded successfully.")
    except Exception as e:
        logger.warning(
            f"Translation model could not be loaded ({e}). "
            "Translation will be skipped."
        )
        _model_available = False

    return _model_available


class Translator:
    """
    Translates terminology terms from English to Arabic.

    Uses Helsinki-NLP/opus-mt-en-ar when available.
    Falls back gracefully if the model cannot be loaded.
    """

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: If False, skip all translation attempts.
        """
        self.enabled = enabled
        self.cache: Dict[str, str] = {}

    def translate_term(self, term: str) -> Optional[str]:
        """
        Translate a single English term to Arabic.

        Returns:
            The Arabic translation string, or None if unavailable.
        """
        if not self.enabled:
            return None

        # Check cache first
        key = term.lower().strip()
        if key in self.cache:
            return self.cache[key]

        if not _load_model():
            return None

        try:
            inputs = _tokenizer(term, return_tensors="pt", padding=True, truncation=True)
            translated = _model.generate(**inputs)
            result = _tokenizer.decode(translated[0], skip_special_tokens=True)
            self.cache[key] = result
            return result
        except Exception as e:
            logger.warning(f"Translation failed for '{term}': {e}")
            return None

    def translate_batch(self, terms: list) -> Dict[str, Optional[str]]:
        """
        Translate a list of terms.

        Returns:
            dict mapping original term -> Arabic translation (or None).
        """
        results: Dict[str, Optional[str]] = {}
        for term in terms:
            results[term] = self.translate_term(term)
        return results

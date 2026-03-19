"""
term_extractor.py - Candidate Term Extraction Module

Responsible for:
    - Extracting candidate terms from text using NLP noun-phrase chunking
    - Extracting n-gram candidates (1-4 tokens)
    - Merging candidates from both methods
    - Tracking source locations for each candidate
"""

import re
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

logger = logging.getLogger(__name__)

# Ensure the model is loaded once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error(
        "spaCy model 'en_core_web_sm' not found. "
        "Install it with: python -m spacy download en_core_web_sm"
    )
    raise


class CandidateTerm:
    """Holds a raw candidate term with occurrence metadata."""

    def __init__(self, surface_form: str):
        self.surface_form = surface_form
        self.occurrences: int = 0
        self.source_locations: List[Dict] = []

    def add_occurrence(self, source_location: Dict) -> None:
        self.occurrences += 1
        # Avoid duplicate source locations
        if source_location not in self.source_locations:
            self.source_locations.append(source_location)

    def __repr__(self) -> str:
        return (
            f"CandidateTerm('{self.surface_form}', "
            f"occurrences={self.occurrences})"
        )


class TermExtractor:
    """
    Extracts candidate terminology from text segments using:
        1. spaCy noun-phrase chunking
        2. N-gram extraction (1-4 tokens)
    Combines both and tracks source locations.
    """

    # Tokens that should never form a standalone term
    STOPWORDS: Set[str] = STOP_WORDS | {
        "e.g.", "i.e.", "etc.", "et", "al", "fig", "table",
    }

    # Only keep terms whose POS tags are predominantly nominal
    VALID_POS = {"NOUN", "PROPN", "ADJ"}

    def __init__(self, min_freq: int = 2, max_tokens: int = 5):
        """
        Args:
            min_freq:   Minimum corpus frequency to keep a candidate.
            max_tokens: Maximum number of tokens in a candidate phrase.
        """
        self.min_freq = min_freq
        self.max_tokens = max_tokens
        # surface_lower -> CandidateTerm
        self.candidates: Dict[str, CandidateTerm] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_segments(self, segments) -> Dict[str, CandidateTerm]:
        """
        Run extraction on a list of TextSegment objects.
        Returns the merged candidate dictionary.
        """
        logger.info("Starting candidate term extraction …")
        for seg_idx, seg in enumerate(segments):
            source = seg.to_dict()
            source["segment_index"] = seg_idx
            doc = nlp(seg.text)

            # Method 1: noun-phrase chunks
            self._extract_noun_phrases(doc, source)

            # Method 2: n-grams
            self._extract_ngrams(doc, source)

        logger.info(f"Raw candidates (before filtering): {len(self.candidates)}")
        self._apply_filters()
        logger.info(f"Candidates after filtering: {len(self.candidates)}")
        return self.candidates

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_noun_phrases(self, doc, source: Dict) -> None:
        """Extract noun phrase chunks from a spaCy Doc."""
        for chunk in doc.noun_chunks:
            term_text = self._clean_span(chunk)
            if term_text:
                self._register(term_text, source)

    def _extract_ngrams(self, doc, source: Dict) -> None:
        """Generate token n-grams (1 to max_tokens) and keep nominal ones."""
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        for n in range(1, self.max_tokens + 1):
            for i in range(len(tokens) - n + 1):
                span = tokens[i : i + n]
                # At least one token must be NOUN or PROPN
                if not any(t.pos_ in {"NOUN", "PROPN"} for t in span):
                    continue
                term_text = " ".join(t.text for t in span).strip()
                term_text = self._clean_text(term_text)
                if term_text and len(term_text) > 1:
                    self._register(term_text, source)

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_span(span) -> str:
        """Clean a spaCy Span, stripping leading determiners and whitespace."""
        tokens = [t for t in span if t.pos_ != "DET" and not t.is_space]
        text = " ".join(t.text for t in tokens).strip()
        return TermExtractor._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Lowercase, strip, and remove surrounding punctuation."""
        text = text.lower().strip()
        text = re.sub(r"^[^\w]+|[^\w]+$", "", text)  # strip edge punctuation
        text = re.sub(r"\s{2,}", " ", text)  # collapse whitespace
        return text

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register(self, term_text: str, source: Dict) -> None:
        """Register a candidate or increment its count."""
        key = term_text.lower()
        if key not in self.candidates:
            self.candidates[key] = CandidateTerm(surface_form=key)
        self.candidates[key].add_occurrence(source)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_filters(self) -> None:
        """Remove candidates that fail quality checks."""
        to_remove = []
        for key, cand in self.candidates.items():
            tokens = key.split()
            # Rule 1: reject if > max tokens
            if len(tokens) > self.max_tokens:
                to_remove.append(key)
                continue
            # Rule 2: reject if stopword ratio > 60 %
            sw_ratio = sum(1 for t in tokens if t in self.STOPWORDS) / len(tokens)
            if sw_ratio > 0.6:
                to_remove.append(key)
                continue
            # Rule 3: reject if contains digits or only punctuation
            if re.search(r"\d", key) or not re.search(r"[a-zA-Z]", key):
                to_remove.append(key)
                continue
            # Rule 4: reject if frequency < min_freq
            if cand.occurrences < self.min_freq:
                to_remove.append(key)
                continue
            # Rule 5: reject single-character or empty
            if len(key) <= 2:
                to_remove.append(key)
                continue

        for key in to_remove:
            del self.candidates[key]

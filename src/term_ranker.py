"""
term_ranker.py - Term Ranking Module

Responsible for:
    - Computing Term Frequency (TF)
    - Computing TF-IDF scores
    - Producing a confidence score between 0 and 1 for each candidate term
"""

import math
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class TermRanker:
    """
    Ranks candidate terms by statistical importance using TF and TF-IDF.
    Produces a normalised confidence score in [0, 1].
    """

    def __init__(self):
        self.corpus_size: int = 0
        self.document_freq: Dict[str, int] = {}
        self.term_freq: Dict[str, int] = {}
        self.tfidf_scores: Dict[str, float] = {}
        self.confidence_scores: Dict[str, float] = {}

    def compute_scores(
        self,
        candidates: Dict,
        segments: List,
    ) -> Dict[str, float]:
        """
        Compute TF-IDF-based confidence scores for all candidates.

        Args:
            candidates: dict mapping term_key -> CandidateTerm
            segments:   list of TextSegment objects (each acts as a document)

        Returns:
            dict mapping term_key -> confidence score in [0, 1]
        """
        logger.info("Computing term ranking scores …")

        # Use segment count as corpus size; DF is derived from source locations.
        self.corpus_size = len(segments)

        if self.corpus_size == 0:
            logger.warning("No segments provided for ranking.")
            return {}

        # --- Term Frequency ------------------------------------------------
        total_occurrences = sum(c.occurrences for c in candidates.values())
        if total_occurrences == 0:
            total_occurrences = 1  # avoid division by zero

        for key, cand in candidates.items():
            self.term_freq[key] = cand.occurrences

        # --- Document Frequency --------------------------------------------
        # Large speedup: DF is the number of unique segments where the candidate appears,
        # computed directly from extraction metadata rather than O(terms * documents)
        # substring scans.
        for key, cand in candidates.items():
            segment_ids = {
                loc.get("segment_index")
                for loc in cand.source_locations
                if isinstance(loc.get("segment_index"), int)
            }
            df = len(segment_ids)
            self.document_freq[key] = max(df, 1)

        # --- TF-IDF --------------------------------------------------------
        max_tfidf = 0.0
        for key, cand in candidates.items():
            tf = cand.occurrences / total_occurrences
            idf = math.log((1 + self.corpus_size) / (1 + self.document_freq[key])) + 1
            score = tf * idf
            self.tfidf_scores[key] = score
            if score > max_tfidf:
                max_tfidf = score

        # --- Normalise to [0, 1] -------------------------------------------
        if max_tfidf == 0:
            max_tfidf = 1.0

        for key, score in self.tfidf_scores.items():
            self.confidence_scores[key] = round(score / max_tfidf, 4)

        logger.info(f"Ranked {len(self.confidence_scores)} terms.")
        return self.confidence_scores

    def get_top_terms(self, n: int = 50) -> List[tuple]:
        """Return the top-n terms sorted by confidence score descending."""
        sorted_terms = sorted(
            self.confidence_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_terms[:n]

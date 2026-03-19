"""
context_extractor.py - Context Sentence Extraction Module

Responsible for:
    - Extracting short example sentences for term occurrences
    - Matching source locations to original text segments
"""

import re
from typing import Any, Dict, List


class ContextExtractor:
    """Extracts concise example sentences for terminology entries."""

    _SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def extract_examples(
        self,
        canonical_term: str,
        surface_forms: List[str],
        source_locations: List[Dict[str, Any]],
        segments,
        max_examples: int = 2,
    ) -> List[str]:
        """Return up to max_examples example sentences for a term."""
        examples: List[str] = []
        term_candidates = [canonical_term] + surface_forms

        for location in source_locations:
            text = self._find_segment_text(segments, location)
            if not text:
                continue

            sentence = self._best_sentence_for_term(text, term_candidates)
            if sentence and sentence not in examples:
                examples.append(sentence)

            if len(examples) >= max_examples:
                break

        return examples

    def _find_segment_text(self, segments, location: Dict[str, Any]) -> str:
        """Find segment text that matches a source location."""
        for seg in segments:
            if (
                seg.chapter_id == location.get("chapter_id")
                and seg.chapter_title == location.get("chapter_title")
                and seg.heading == location.get("heading")
                and seg.page_number == location.get("page_number")
            ):
                return seg.text
        return ""

    def _best_sentence_for_term(self, text: str, terms: List[str]) -> str:
        """Select sentence containing the term; fallback to first sentence."""
        sentences = [
            sentence.strip()
            for sentence in self._SENTENCE_SPLIT_PATTERN.split(text.strip())
            if sentence.strip()
        ]
        if not sentences:
            return ""

        lowered_terms = [term.lower() for term in terms if term]
        for sentence in sentences:
            lowered_sentence = sentence.lower()
            if any(term in lowered_sentence for term in lowered_terms):
                return sentence

        return sentences[0]

"""
context_extractor.py - Context Sentence Extraction Module

Responsible for:
    - Ranking candidate sentences for translation usefulness
    - Selecting diverse supporting sentences with MMR
    - Returning explainable score breakdowns
"""

import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from spacy.lang.en.stop_words import STOP_WORDS


class ContextExtractor:
    """Extracts and ranks example sentences for terminology entries."""

    _SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
    _WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")

    # Definitional patterns used in terminology and summarization literature.
    _DEFINITION_PATTERNS = (
        re.compile(r"\b(is|are)\s+(a|an|the)\b", re.IGNORECASE),
        re.compile(r"\brefers\s+to\b", re.IGNORECASE),
        re.compile(r"\b(can\s+be\s+defined\s+as|is\s+defined\s+as)\b", re.IGNORECASE),
        re.compile(r"\bmeans\b", re.IGNORECASE),
    )

    def __init__(self, mmr_lambda: float = 0.7):
        # MMR lambda controls relevance vs diversity trade-off.
        self.mmr_lambda = mmr_lambda

    def extract_examples(
        self,
        canonical_term: str,
        surface_forms: List[str],
        source_locations: List[Dict[str, Any]],
        segments,
        max_examples: int = 2,
    ) -> Dict[str, Any]:
        """Return ranked sentence evidence for a term.

        Output keys:
            - example_sentences
            - primary_example_sentence
            - supporting_example_sentences
            - example_score_breakdown
        """
        term_candidates = [canonical_term] + surface_forms
        candidate_sentences = self._collect_candidate_sentences(
            source_locations=source_locations,
            segments=segments,
        )

        if not candidate_sentences:
            return {
                "example_sentences": [],
                "primary_example_sentence": None,
                "supporting_example_sentences": [],
                "example_score_breakdown": {},
            }

        scored = self._score_candidates(
            canonical_term=canonical_term,
            surface_forms=surface_forms,
            candidate_sentences=candidate_sentences,
        )

        selected_indices = self._select_with_mmr(scored, max_examples=max_examples)
        ordered_selected = [scored[i] for i in selected_indices]

        example_sentences = [item["sentence"] for item in ordered_selected]
        primary = example_sentences[0] if example_sentences else None
        supporting = example_sentences[1:] if len(example_sentences) > 1 else []

        score_breakdown = ordered_selected[0]["breakdown"] if ordered_selected else {}
        score_breakdown["selection_method"] = "hybrid_weighted_score+mmr"
        score_breakdown["term_candidates"] = term_candidates[:5]

        return {
            "example_sentences": example_sentences,
            "primary_example_sentence": primary,
            "supporting_example_sentences": supporting,
            "example_score_breakdown": score_breakdown,
        }

    def _collect_candidate_sentences(
        self,
        source_locations: List[Dict[str, Any]],
        segments,
    ) -> List[str]:
        """Collect unique candidate sentences from all matched source segments."""
        candidates: List[str] = []

        for location in source_locations:
            text = self._find_segment_text(segments, location)
            if not text:
                continue
            sentences = [
                sentence.strip()
                for sentence in self._SENTENCE_SPLIT_PATTERN.split(text.strip())
                if sentence.strip()
            ]
            for sentence in sentences:
                if sentence not in candidates:
                    candidates.append(sentence)

        return candidates

    def _score_candidates(
        self,
        canonical_term: str,
        surface_forms: List[str],
        candidate_sentences: List[str],
    ) -> List[Dict[str, Any]]:
        """Score candidates with weighted translation-utility objective using BM25."""
        term_candidates = [canonical_term] + surface_forms
        query_text = f"{canonical_term} translation context definition technical meaning"

        # Tokenize query and sentences for BM25
        def tokenize(text: str) -> List[str]:
            return [
                token.lower()
                for token in self._WORD_PATTERN.findall(text)
                if token.lower() not in STOP_WORDS
            ]

        query_tokens = tokenize(query_text)
        corpus_tokens = [tokenize(sentence) for sentence in candidate_sentences]

        # Build BM25 index
        bm25 = BM25Okapi(corpus_tokens)
        semantic_scores = bm25.get_scores(query_tokens)

        scored: List[Dict[str, Any]] = []
        for idx, sentence in enumerate(candidate_sentences):
            exact = self._exact_match_score(sentence, term_candidates)
            semantic = float(semantic_scores[idx])
            definitional = self._definitional_score(sentence, term_candidates)
            salience = self._domain_salience_score(sentence)
            quality = self._sentence_quality_score(sentence)

            final_score = (
                0.30 * exact
                + 0.25 * semantic
                + 0.20 * definitional
                + 0.15 * salience
                + 0.10 * quality
            )

            scored.append(
                {
                    "sentence": sentence,
                    "exact": exact,
                    "semantic": semantic,
                    "final_score": float(round(final_score, 4)),
                    "breakdown": {
                        "exact_match_score": round(exact, 4),
                        "semantic_similarity_score": round(semantic, 4),
                        "definitional_score": round(definitional, 4),
                        "domain_salience_score": round(salience, 4),
                        "sentence_quality_score": round(quality, 4),
                        "final_score": float(round(final_score, 4)),
                    },
                }
            )

        # Prefer sentences that either explicitly mention the term or are semantically close.
        relevant = [
            item for item in scored
            if item["exact"] > 0.0 or item["semantic"] >= 0.05
        ]
        if relevant:
            scored = relevant

        scored.sort(key=lambda item: item["final_score"], reverse=True)
        return scored

    def _select_with_mmr(self, scored: List[Dict[str, Any]], max_examples: int) -> List[int]:
        """Select sentences using MMR for relevance + diversity (score-based similarity)."""
        if not scored:
            return []
        if len(scored) <= max_examples:
            return list(range(len(scored)))

        selected = [0]
        while len(selected) < max_examples:
            best_idx = None
            best_mmr = -1.0

            for idx in range(len(scored)):
                if idx in selected:
                    continue

                relevance = scored[idx]["final_score"]
                candidate_semantic = scored[idx]["semantic"]

                # Use semantic score difference as diversity proxy
                # (dissimilar semantic scores = diverse content)
                max_sim = 0.0
                for selected_idx in selected:
                    selected_semantic = scored[selected_idx]["semantic"]
                    # Normalize to [0,1] range
                    sim = abs(candidate_semantic - selected_semantic)
                    if sim > max_sim:
                        max_sim = sim

                mmr_score = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx is None:
                break
            selected.append(best_idx)

        return selected

    def _find_segment_text(self, segments, location: Dict[str, Any]) -> str:
        """Find segment text that matches a source location."""
        segment_index = location.get("segment_index")
        if isinstance(segment_index, int) and 0 <= segment_index < len(segments):
            return segments[segment_index].text

        for seg in segments:
            if (
                seg.chapter_id == location.get("chapter_id")
                and seg.chapter_title == location.get("chapter_title")
                and seg.heading == location.get("heading")
                and seg.page_number == location.get("page_number")
            ):
                return seg.text
        return ""

    def _exact_match_score(self, sentence: str, terms: List[str]) -> float:
        """Return score in [0,1] based on direct term presence."""
        lowered_sentence = sentence.lower()
        matches = 0
        valid_terms = [term.lower().strip() for term in terms if term and term.strip()]

        for term in valid_terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, lowered_sentence):
                matches += 1

        if not valid_terms:
            return 0.0
        return min(1.0, matches / max(1, len(valid_terms)))

    def _definitional_score(self, sentence: str, terms: List[str]) -> float:
        """Boost sentences likely to define the term explicitly."""
        lowered_sentence = sentence.lower()
        term_match = self._exact_match_score(sentence, terms) > 0.0

        if term_match:
            for pattern in self._DEFINITION_PATTERNS:
                if pattern.search(lowered_sentence):
                    return 1.0

        if term_match and ("such as" in lowered_sentence or "including" in lowered_sentence):
            return 0.7
        return 0.0

    def _domain_salience_score(self, sentence: str) -> float:
        """Approximate informativeness by content-word density."""
        tokens = [token.lower() for token in self._WORD_PATTERN.findall(sentence)]
        if not tokens:
            return 0.0

        content_tokens = [
            token
            for token in tokens
            if token not in STOP_WORDS and len(token) > 2
        ]
        return min(1.0, len(content_tokens) / len(tokens))

    def _sentence_quality_score(self, sentence: str) -> float:
        """Reward sentence length that is usually best for translation context."""
        length = len(self._WORD_PATTERN.findall(sentence))
        if length == 0:
            return 0.0
        if 10 <= length <= 30:
            return 1.0
        if 7 <= length <= 35:
            return 0.7
        if 5 <= length <= 40:
            return 0.4
        return 0.2

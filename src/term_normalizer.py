"""
term_normalizer.py - Term Normalization Module

Responsible for:
    - Lowercasing
    - Lemmatization (plural → singular, verb forms → base)
    - Punctuation removal
    - Merging equivalent surface forms into a single canonical term
"""

import re
import logging
from typing import Dict, List, Tuple

import spacy

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error(
        "spaCy model 'en_core_web_sm' not found. "
        "Install it with: python -m spacy download en_core_web_sm"
    )
    raise


class TermNormalizer:
    """
    Normalizes candidate term surface forms to a single canonical form and
    merges duplicates.
    """

    def __init__(self):
        # canonical_form -> list of original surface forms
        self.canonical_map: Dict[str, List[str]] = {}

    def normalize(self, term: str) -> str:
        """
        Produce a canonical normalized form for a term string.

        Steps:
            1. Lowercase
            2. Remove leading/trailing punctuation
            3. Lemmatize each token
            4. Collapse whitespace
        """
        text = term.lower().strip()
        text = re.sub(r"^[^\w]+|[^\w]+$", "", text)

        doc = nlp(text)
        lemmas = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            lemmas.append(token.lemma_)

        normalized = " ".join(lemmas).strip()
        normalized = re.sub(r"\s{2,}", " ", normalized)
        return normalized

    def normalize_candidates(
        self, candidates: Dict
    ) -> Dict[str, Dict]:
        """
        Normalize all candidate terms and merge those that share the same
        canonical form.

        Args:
            candidates: dict  key -> CandidateTerm

        Returns:
            dict mapping canonical_form -> {
                'surface_forms': [str, ...],
                'occurrences': int,
                'source_locations': [dict, ...],
                'surface_form_variants': [
                    {'form': str, 'frequency': int, 'percentage': float},
                    ...
                ],
            }
        """
        logger.info("Normalizing candidate terms …")
        merged: Dict[str, Dict] = {}

        for key, cand in candidates.items():
            canonical = self.normalize(cand.surface_form)
            if not canonical:
                continue

            if canonical not in merged:
                merged[canonical] = {
                    "surface_forms": [],
                    "occurrences": 0,
                    "source_locations": [],
                    "variant_frequencies": {},
                }

            entry = merged[canonical]
            if cand.surface_form not in entry["surface_forms"]:
                entry["surface_forms"].append(cand.surface_form)

            entry["occurrences"] += cand.occurrences
            entry["variant_frequencies"][cand.surface_form] = (
                entry["variant_frequencies"].get(cand.surface_form, 0)
                + cand.occurrences
            )

            for loc in cand.source_locations:
                if loc not in entry["source_locations"]:
                    entry["source_locations"].append(loc)

        for entry in merged.values():
            variant_frequencies = entry.pop("variant_frequencies", {})
            total = sum(variant_frequencies.values())
            variants = []

            for form, freq in sorted(
                variant_frequencies.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                percentage = round((freq / total) * 100, 1) if total > 0 else 0.0
                variants.append(
                    {
                        "form": form,
                        "frequency": freq,
                        "percentage": percentage,
                    }
                )

            entry["surface_form_variants"] = variants

        logger.info(
            f"Normalized {len(candidates)} surface forms → "
            f"{len(merged)} canonical terms."
        )
        self.canonical_map = {k: v["surface_forms"] for k, v in merged.items()}
        return merged

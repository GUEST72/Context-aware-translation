"""
alias_detector.py - Alias / Abbreviation Detection Module

Responsible for:
    - Detecting abbreviations written in parenthetical form
      e.g. "Natural Language Processing (NLP)"
    - Building a mapping from full term → list of aliases
"""

import re
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# Pattern: Full Term (ABBREV)
# Captures the full form and the abbreviation inside parentheses.
_ABBREV_PATTERN = re.compile(
    r"([A-Z][\w\s\-]{2,80}?)\s*\(([A-Z][A-Z0-9\-]{1,15})\)"
)


class AliasDetector:
    """
    Detects abbreviation-style aliases from text.

    Usage:
        detector = AliasDetector()
        detector.detect_from_segments(segments)
        aliases = detector.get_aliases("natural language processing")
    """

    def __init__(self):
        # normalized_term_lower -> set of alias strings
        self.alias_map: Dict[str, Set[str]] = {}

    def detect_from_segments(self, segments) -> Dict[str, Set[str]]:
        """
        Scan all text segments for abbreviation patterns.

        Args:
            segments: list of TextSegment objects.

        Returns:
            dict mapping lowercased full term -> set of alias strings.
        """
        logger.info("Detecting aliases / abbreviations …")

        for seg in segments:
            self._scan_text(seg.text)

        logger.info(f"Detected aliases for {len(self.alias_map)} terms.")
        return self.alias_map

    def _scan_text(self, text: str) -> None:
        """Find all abbreviation patterns in a text string."""
        for match in _ABBREV_PATTERN.finditer(text):
            full_form = match.group(1).strip().lower()
            abbreviation = match.group(2).strip()

            # Validate: abbreviation should roughly match initials
            if self._is_plausible_abbreviation(full_form, abbreviation):
                if full_form not in self.alias_map:
                    self.alias_map[full_form] = set()
                self.alias_map[full_form].add(abbreviation)
                logger.debug(f"Alias detected: '{full_form}' → '{abbreviation}'")

    @staticmethod
    def _is_plausible_abbreviation(full_form: str, abbreviation: str) -> bool:
        """
        Heuristic: first letters of the significant words in the full form
        should roughly correspond to the abbreviation letters.
        """
        words = full_form.split()
        # Filter out very short function words
        significant = [w for w in words if len(w) > 2]
        if not significant:
            significant = words

        initials = "".join(w[0] for w in significant).upper()

        # Accept if the abbreviation matches the initials, or at least
        # shares most letters in order.
        if initials == abbreviation.upper():
            return True

        # Fallback: at least 50 % letter overlap with initials
        overlap = sum(1 for a, b in zip(initials, abbreviation.upper()) if a == b)
        if len(abbreviation) > 0 and overlap / len(abbreviation) >= 0.5:
            return True

        return False

    def get_aliases(self, term: str) -> List[str]:
        """
        Return known aliases (abbreviations) for a term.

        Performs an exact match first, then a substring match as fallback.
        """
        key = term.lower().strip()

        if key in self.alias_map:
            return sorted(self.alias_map[key])

        # Substring fallback
        for stored_key, aliases in self.alias_map.items():
            if key in stored_key or stored_key in key:
                return sorted(aliases)

        return []

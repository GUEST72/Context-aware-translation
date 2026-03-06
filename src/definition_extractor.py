"""
definition_extractor.py - Definition Extraction Module

Responsible for:
    - Scanning paragraphs for definitional patterns
    - Associating extracted definitions with their target term
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Definitional patterns (compiled once)
# ---------------------------------------------------------------------------
# Each pattern captures two groups:
#   group(1) = the term being defined
#   group(2) = the definition text
_PATTERNS = [
    # "X is a …" / "X is the …"
    re.compile(
        r"(?:^|(?<=\.\s))([A-Z][\w\s\-]{2,60}?)\s+is\s+(a|an|the)\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "X refers to …"
    re.compile(
        r"(?:^|(?<=\.\s))([A-Z][\w\s\-]{2,60}?)\s+refers\s+to\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "X can be defined as …"
    re.compile(
        r"(?:^|(?<=\.\s))([A-Z][\w\s\-]{2,60}?)\s+can\s+be\s+defined\s+as\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "X is defined as …"
    re.compile(
        r"(?:^|(?<=\.\s))([A-Z][\w\s\-]{2,60}?)\s+is\s+defined\s+as\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
]


class DefinitionExtractor:
    """
    Extracts definitions from paragraphs using regex-based patterns.

    Usage:
        extractor = DefinitionExtractor()
        extractor.extract_from_segments(segments)
        definition = extractor.get_definition("machine learning")
    """

    def __init__(self):
        # normalized_term_lower -> definition string
        self.definitions: Dict[str, str] = {}

    def extract_from_segments(self, segments) -> Dict[str, str]:
        """
        Scan all text segments for definitional sentences.

        Args:
            segments: list of TextSegment objects.

        Returns:
            dict mapping lowercased term -> definition string.
        """
        logger.info("Extracting definitions from text segments …")

        for seg in segments:
            self._scan_paragraph(seg.text)

        logger.info(f"Extracted {len(self.definitions)} definitions.")
        return self.definitions

    def _scan_paragraph(self, text: str) -> None:
        """Apply all definitional patterns to a paragraph."""
        for pattern in _PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) == 3:
                    # Pattern with article: term, article, definition body
                    term = groups[0].strip()
                    definition = f"{groups[1]} {groups[2]}".strip()
                elif len(groups) == 2:
                    term = groups[0].strip()
                    definition = groups[1].strip()
                else:
                    continue

                term_key = term.lower().strip()
                # Keep the first (usually most explicit) definition
                if term_key not in self.definitions:
                    self.definitions[term_key] = definition
                    logger.debug(f"Definition found: '{term_key}' → '{definition[:60]}…'")

    def get_definition(self, term: str) -> Optional[str]:
        """
        Retrieve the definition for a term.

        The lookup is fuzzy: it checks if the query is a substring of any
        stored key or vice-versa so that "machine learning" matches
        "machine learning" even if the stored key has extra context.
        """
        key = term.lower().strip()

        # Exact match first
        if key in self.definitions:
            return self.definitions[key]

        # Substring match (query inside stored key or stored key inside query)
        for stored_key, definition in self.definitions.items():
            if key in stored_key or stored_key in key:
                return definition

        return None

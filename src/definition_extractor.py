"""
definition_extractor.py - Definition Extraction Module

Responsible for:
    - Scanning paragraphs for definitional patterns
    - Associating extracted definitions with their target term
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

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
        self._definition_candidates: Dict[str, List[Tuple[str, int]]] = {}
        self._token_pattern = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
        self._sentence_split_pattern = re.compile(r"(?<=[.!?])\s+")
        self._generic_term_blacklist: Set[str] = {
            "figure", "section", "chapter", "example", "problem", "discussion"
        }

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
                term_key = self._normalize_term(term_key)
                definition = self._clean_definition(definition)
                if not self._is_valid_definition(term_key, definition):
                    continue

                score = self._score_definition(definition)
                self._definition_candidates.setdefault(term_key, []).append(
                    (definition, score)
                )

                # Keep the current best-scored definition for this term.
                current = self.definitions.get(term_key)
                if current is None or score > self._score_definition(current):
                    self.definitions[term_key] = definition
                    logger.debug(
                        f"Definition found: '{term_key}' → '{definition[:60]}…' (score={score})"
                    )

    def get_definition(
        self,
        term: str,
        surface_forms: Optional[List[str]] = None,
        source_locations: Optional[List[Dict]] = None,
        segments: Optional[List] = None,
    ) -> Optional[str]:
        definitions = self.get_definitions(
            term=term,
            surface_forms=surface_forms,
            source_locations=source_locations,
            segments=segments,
            max_definitions=1,
        )
        return definitions[0] if definitions else None

    def get_definitions(
        self,
        term: str,
        surface_forms: Optional[List[str]] = None,
        source_locations: Optional[List[Dict]] = None,
        segments: Optional[List] = None,
        max_definitions: int = 3,
    ) -> List[str]:
        """
        Retrieve multiple ranked definitions for a term.

        Returns up to max_definitions entries, ranked by lexical fit and
        definition quality heuristics.
        """
        if max_definitions <= 0:
            return []

        key = self._normalize_term(term)
        normalized_surface_forms = {
            self._normalize_term(s) for s in (surface_forms or []) if s and s.strip()
        }
        query_keys = [key] + [sf for sf in normalized_surface_forms if sf != key]

        scored_candidates: List[Tuple[str, float]] = []

        # Exact term/surface candidates extracted from global definitional patterns.
        for qk in query_keys:
            for definition, score in self._definition_candidates.get(qk, []):
                scored_candidates.append((definition, float(100 + score)))

        # Token-overlap fallback with strict threshold to avoid semantic drift.
        query_tokens = self._term_tokens(key) | {
            token
            for sf in normalized_surface_forms
            for token in self._term_tokens(sf)
        }
        if not query_tokens:
            return []

        for stored_key, definition in self.definitions.items():
            stored_tokens = self._term_tokens(stored_key)
            if not stored_tokens:
                continue
            overlap = len(query_tokens & stored_tokens)
            token_score = overlap / max(len(query_tokens), len(stored_tokens))

            # Require strong lexical overlap before accepting fallback.
            if token_score < 0.8:
                continue

            # Prefer closest key length and better definition quality.
            length_penalty = abs(len(stored_tokens) - len(query_tokens)) * 0.05
            candidate_score = token_score - length_penalty + (self._score_definition(definition) / 100.0)
            scored_candidates.append((definition, candidate_score))

        # Term-local fallback: search only in the term's source sentences.
        local_candidates = self._extract_from_local_context(
            term=key,
            surface_forms=list(normalized_surface_forms),
            source_locations=source_locations or [],
            segments=segments or [],
        )
        scored_candidates.extend(local_candidates)

        # Rank, deduplicate, and keep top-k.
        ranked = sorted(scored_candidates, key=lambda item: item[1], reverse=True)
        results: List[str] = []
        seen: Set[str] = set()
        for definition, _ in ranked:
            cleaned = self._clean_definition(definition)
            key_def = cleaned.lower()
            if not cleaned or key_def in seen:
                continue
            seen.add(key_def)
            results.append(cleaned)
            if len(results) >= max_definitions:
                break

        return results

    def _normalize_term(self, term: str) -> str:
        return re.sub(r"\s{2,}", " ", term.lower().strip())

    def _clean_definition(self, definition: str) -> str:
        cleaned = definition.strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = cleaned.strip("-–—:; ")
        return cleaned

    def _term_tokens(self, text: str) -> Set[str]:
        return {token.lower() for token in self._token_pattern.findall(text)}

    def _is_valid_definition(self, term_key: str, definition: str) -> bool:
        if not definition:
            return False

        # Reject extremely short or overly long definitions.
        words = self._token_pattern.findall(definition)
        if len(words) < 4 or len(words) > 55:
            return False

        # Reject bibliography/index-like artifacts and URL-heavy snippets.
        lowered = definition.lower()
        noisy_markers = (
            "http://", "https://", "rfc", "figure", "chapter", "pp.", "usenix"
        )
        if any(marker in lowered for marker in noisy_markers):
            return False

        bad_openings = (
            "to ",
            "by ",
            "in order to",
            "for example",
            "let ",
            "there is ",
            "it is ",
            "we ",
            "what ",
            "same as ",
            "not in ",
            "typically ",
            "also ",
        )
        if lowered.startswith(bad_openings):
            return False

        # Reject question-like fragments and instructional phrasing.
        if "?" in definition:
            return False
        instructional_markers = (
            "let's ", "let us ", "we'll ", "recall ", "consider "
        )
        if any(marker in lowered for marker in instructional_markers):
            return False

        # Reject parenthetical and citation-heavy fragments.
        bracket_count = definition.count("(") + definition.count(")")
        if bracket_count >= 4:
            return False

        # Reject fragments dominated by numbers/symbols.
        non_alpha_ratio = sum(1 for ch in definition if not ch.isalpha() and not ch.isspace()) / max(len(definition), 1)
        if non_alpha_ratio > 0.35:
            return False

        return True

    def _extract_from_local_context(
        self,
        term: str,
        surface_forms: List[str],
        source_locations: List[Dict],
        segments: List,
    ) -> List[Tuple[str, float]]:
        """Extract scored definition candidates from source sentences local to the term."""
        if not source_locations or not segments:
            return []

        forms = [term] + [sf for sf in surface_forms if sf]
        forms = [self._normalize_term(f) for f in forms if f.strip()]
        forms = list(dict.fromkeys(forms))

        candidates: List[Tuple[str, float]] = []
        for sentence in self._collect_source_sentences(source_locations, segments):
            sent = sentence.strip()
            if not sent:
                continue
            score = self._score_definition_sentence(sent, forms)
            if score <= 0:
                continue

            extracted = self._extract_definition_body(sent, forms)
            if extracted and self._is_valid_definition(term, extracted):
                local_score = float(score + self._score_definition(extracted))
                candidates.append((extracted, local_score))

        return candidates

    def _collect_source_sentences(self, source_locations: List[Dict], segments: List) -> List[str]:
        sentences: List[str] = []
        seen = set()

        for location in source_locations:
            segment_text = self._find_segment_text(segments, location)
            if not segment_text:
                continue
            for sentence in self._sentence_split_pattern.split(segment_text.strip()):
                s = sentence.strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                sentences.append(s)

        return sentences

    def _find_segment_text(self, segments: List, location: Dict) -> str:
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

    def _score_definition_sentence(self, sentence: str, forms: List[str]) -> int:
        lowered = sentence.lower()
        score = 0

        if any(re.search(rf"\b{re.escape(form)}\b", lowered) for form in forms if form):
            score += 3

        definitional_cues = (
            " is a ", " is an ", " is the ", " refers to ",
            " defined as ", " means ", " called "
        )
        if any(cue in lowered for cue in definitional_cues):
            score += 4

        noisy_cues = (
            "in the context of", "let's", "figure", "section", "chapter", "rfc"
        )
        if any(cue in lowered for cue in noisy_cues):
            score -= 2

        words = self._token_pattern.findall(sentence)
        if 8 <= len(words) <= 35:
            score += 2
        elif 5 <= len(words) <= 45:
            score += 1

        return score

    def _extract_definition_body(self, sentence: str, forms: List[str]) -> Optional[str]:
        lowered = sentence.lower()
        for form in forms:
            if not form:
                continue
            form_re = re.escape(form)
            patterns = (
                rf"\b{form_re}\b\s+is\s+(?:a|an|the)?\s*(.+?)$",
                rf"\b{form_re}\b\s+refers\s+to\s+(.+?)$",
                rf"\b{form_re}\b\s+(?:can\s+be\s+)?defined\s+as\s+(.+?)$",
                rf"\b{form_re}\b\s+means\s+(.+?)$",
            )
            for pattern in patterns:
                m = re.search(pattern, lowered)
                if m:
                    body = self._clean_definition(m.group(1))
                    if body:
                        return body

        return None

    def _score_definition(self, definition: str) -> int:
        score = 0
        lowered = definition.lower()

        # Prefer definitional cues.
        if any(phrase in lowered for phrase in ("is a", "is an", "is the", "refers to", "defined as", "means")):
            score += 4

        words = self._token_pattern.findall(definition)
        if 8 <= len(words) <= 24:
            score += 3
        elif 5 <= len(words) <= 32:
            score += 2
        else:
            score += 1

        if definition.endswith("."):
            score += 1

        # Light penalty for very generic openings.
        if lowered.startswith(("the ", "this ", "that ")):
            score -= 1

        return score

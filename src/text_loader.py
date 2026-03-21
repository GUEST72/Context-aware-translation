"""
text_loader.py - Book JSON Loader Module

Responsible for:
    - Loading structured book JSON files
    - Validating the JSON structure
    - Extracting text content with source location metadata
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TextSegment:
    """Represents a text segment with its source location metadata."""

    def __init__(
        self,
        text: str,
        chapter_id: int,
        chapter_title: str,
        heading: str,
        page_number: Optional[int] = None,
    ):
        self.text = text
        self.chapter_id = chapter_id
        self.chapter_title = chapter_title
        self.heading = heading
        self.page_number = page_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "chapter_title": self.chapter_title,
            "heading": self.heading,
            "page_number": self.page_number,
        }

    def __repr__(self) -> str:
        return (
            f"TextSegment(chapter={self.chapter_id}, "
            f"page={self.page_number}, "
            f"heading='{self.heading[:30]}...', "
            f"text='{self.text[:50]}...')"
        )


class BookLoader:
    """
    Loads and validates a structured book JSON file.
    Extracts paragraphs as TextSegment objects with source metadata.
    """

    REQUIRED_BOOK_KEYS = {"pages"}
    REQUIRED_PAGE_KEYS = {"page", "paragraphs"}
    REQUIRED_PARAGRAPH_KEYS = {"chapter", "section", "paragraph"}
    _STRUCTURAL_PARAGRAPH_PATTERN = re.compile(
        r"^\s*(chapter|chapters|section|sections|appendix|appendices|references|review\s+questions|exercises|summary)\b(?:\s+\d+[\d\.-]*)?",
        re.IGNORECASE,
    )

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.book_data: Optional[Dict[str, Any]] = None
        self.book_title: str = ""
        self.segments: List[TextSegment] = []

    def load(self) -> "BookLoader":
        """Load and parse the JSON file."""
        logger.info(f"Loading book from: {self.filepath}")
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.book_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {self.filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Malformed JSON in {self.filepath}: {e}")
            raise ValueError(f"Malformed JSON: {e}")

        self._validate_structure()
        if self.book_data is None:
            raise ValueError("Book data is empty after loading.")
        self.book_title = str(self.book_data.get("book_title", "")).strip()
        self._extract_segments()
        book_label = self.book_title or "(no title provided)"
        logger.info(
            f"Loaded '{book_label}' with {len(self.segments)} text segments"
        )
        return self

    def _validate_structure(self) -> None:
        """Validate the top-level JSON structure."""
        if not isinstance(self.book_data, dict):
            raise ValueError("Book JSON root must be a dictionary.")

        missing = self.REQUIRED_BOOK_KEYS - set(self.book_data.keys())
        if missing:
            raise ValueError(f"Missing required book keys: {missing}")

        if not isinstance(self.book_data["pages"], list):
            raise ValueError("'pages' must be a list.")

        for idx, page in enumerate(self.book_data["pages"]):
            self._validate_page(page, idx)

    def _validate_page(self, page: Dict, idx: int) -> None:
        """Validate a single page's structure."""
        if not isinstance(page, dict):
            raise ValueError(f"Page at index {idx} must be a dictionary.")

        missing = self.REQUIRED_PAGE_KEYS - set(page.keys())
        if missing:
            logger.warning(f"Page at index {idx} missing keys: {missing}. Skipping.")
            return

        if not isinstance(page["paragraphs"], list):
            raise ValueError(f"Page {page.get('page', idx)}: 'paragraphs' must be a list.")

        for para_idx, paragraph_entry in enumerate(page["paragraphs"]):
            self._validate_paragraph_entry(
                paragraph_entry,
                page.get("page", idx),
                para_idx,
            )

    def _validate_paragraph_entry(
        self, paragraph_entry: Dict, page_number: int, para_idx: int
    ) -> None:
        """Validate a single paragraph entry structure."""
        if not isinstance(paragraph_entry, dict):
            logger.warning(
                f"Paragraph at index {para_idx} on page {page_number} "
                "is not a dictionary. Skipping."
            )
            return

        missing = self.REQUIRED_PARAGRAPH_KEYS - set(paragraph_entry.keys())
        if missing:
            logger.warning(
                f"Paragraph at index {para_idx} on page {page_number} "
                f"missing keys: {missing}. Skipping."
            )

    def _extract_segments(self) -> None:
        """Extract all text segments from the book."""
        self.segments = []
        seen_texts = set()
        chapter_id_map: Dict[str, int] = {}
        next_chapter_id = 1

        if self.book_data is None:
            return

        for page in self.book_data.get("pages", []):
            if not isinstance(page, dict):
                continue

            page_number = page.get("page")
            paragraphs = page.get("paragraphs", [])

            if not isinstance(paragraphs, list):
                logger.warning(
                    f"Paragraphs on page {page_number} is not a list. Skipping."
                )
                continue

            for entry in paragraphs:
                if not isinstance(entry, dict):
                    continue

                chapter_title = str(entry.get("chapter", "Unknown Chapter")).strip()
                heading = str(entry.get("section", "No Heading")).strip()
                paragraph_text = entry.get("paragraph", "")

                if not chapter_title:
                    chapter_title = "Unknown Chapter"
                if not heading:
                    heading = "No Heading"

                if not isinstance(paragraph_text, str) or not paragraph_text.strip():
                    continue

                paragraph_text = paragraph_text.strip()
                if self._is_structural_paragraph(paragraph_text):
                    logger.debug(
                        "Skipping structural heading paragraph on page %s: '%s'",
                        page_number,
                        paragraph_text[:80],
                    )
                    continue

                if chapter_title not in chapter_id_map:
                    chapter_id_map[chapter_title] = next_chapter_id
                    next_chapter_id += 1
                chapter_id = chapter_id_map[chapter_title]

                # Deduplicate identical paragraphs in the same section
                dedup_key = (chapter_id, heading, paragraph_text.strip())
                if dedup_key in seen_texts:
                    logger.debug(f"Duplicate paragraph skipped in '{heading}'")
                    continue
                seen_texts.add(dedup_key)

                self.segments.append(
                    TextSegment(
                        text=paragraph_text.strip(),
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        heading=heading,
                        page_number=page_number,
                    )
                )

    def _is_structural_paragraph(self, text: str) -> bool:
        """Return True for heading-only structural lines like 'Chapter ...' or 'Section ...'."""
        return bool(self._STRUCTURAL_PARAGRAPH_PATTERN.match(text))

    def get_all_text(self) -> str:
        """Return all paragraph text concatenated (for corpus-level analysis)."""
        return " ".join(seg.text for seg in self.segments)

    def get_segments(self) -> List[TextSegment]:
        """Return all extracted TextSegment objects."""
        return self.segments

    def get_book_title(self) -> str:
        """Return the book title, or an empty string if not provided."""
        return self.book_title

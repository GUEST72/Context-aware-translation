"""
text_loader.py - Book JSON Loader Module

Responsible for:
    - Loading structured book JSON files
    - Validating the JSON structure
    - Extracting text content with source location metadata
"""

import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TextSegment:
    """Represents a text segment with its source location metadata."""

    def __init__(
        self,
        text: str,
        chapter_id: int,
        chapter_title: str,
        heading: str,
    ):
        self.text = text
        self.chapter_id = chapter_id
        self.chapter_title = chapter_title
        self.heading = heading

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "chapter_title": self.chapter_title,
            "heading": self.heading,
        }

    def __repr__(self) -> str:
        return (
            f"TextSegment(chapter={self.chapter_id}, "
            f"heading='{self.heading[:30]}...', "
            f"text='{self.text[:50]}...')"
        )


class BookLoader:
    """
    Loads and validates a structured book JSON file.
    Extracts paragraphs as TextSegment objects with source metadata.
    """

    REQUIRED_BOOK_KEYS = {"book_title", "chapters"}
    REQUIRED_CHAPTER_KEYS = {"chapter_id", "chapter_title", "sections"}
    REQUIRED_SECTION_KEYS = {"heading", "paragraphs"}

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.book_data: Optional[Dict] = None
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
        self.book_title = self.book_data.get("book_title", "Unknown")
        self._extract_segments()
        logger.info(
            f"Loaded '{self.book_title}' with {len(self.segments)} text segments"
        )
        return self

    def _validate_structure(self) -> None:
        """Validate the top-level JSON structure."""
        if not isinstance(self.book_data, dict):
            raise ValueError("Book JSON root must be a dictionary.")

        missing = self.REQUIRED_BOOK_KEYS - set(self.book_data.keys())
        if missing:
            raise ValueError(f"Missing required book keys: {missing}")

        if not isinstance(self.book_data["chapters"], list):
            raise ValueError("'chapters' must be a list.")

        for idx, chapter in enumerate(self.book_data["chapters"]):
            self._validate_chapter(chapter, idx)

    def _validate_chapter(self, chapter: Dict, idx: int) -> None:
        """Validate a single chapter's structure."""
        if not isinstance(chapter, dict):
            raise ValueError(f"Chapter at index {idx} must be a dictionary.")

        missing = self.REQUIRED_CHAPTER_KEYS - set(chapter.keys())
        if missing:
            logger.warning(
                f"Chapter at index {idx} missing keys: {missing}. Skipping."
            )
            return

        if not isinstance(chapter["sections"], list):
            raise ValueError(
                f"Chapter {chapter.get('chapter_id', idx)}: "
                "'sections' must be a list."
            )

        for sec_idx, section in enumerate(chapter["sections"]):
            self._validate_section(section, chapter.get("chapter_id", idx), sec_idx)

    def _validate_section(
        self, section: Dict, chapter_id: int, sec_idx: int
    ) -> None:
        """Validate a single section's structure."""
        if not isinstance(section, dict):
            logger.warning(
                f"Section at index {sec_idx} in chapter {chapter_id} "
                "is not a dictionary. Skipping."
            )
            return

        missing = self.REQUIRED_SECTION_KEYS - set(section.keys())
        if missing:
            logger.warning(
                f"Section at index {sec_idx} in chapter {chapter_id} "
                f"missing keys: {missing}. Skipping."
            )

    def _extract_segments(self) -> None:
        """Extract all text segments from the book."""
        self.segments = []
        seen_texts = set()

        for chapter in self.book_data.get("chapters", []):
            chapter_id = chapter.get("chapter_id", 0)
            chapter_title = chapter.get("chapter_title", "Unknown Chapter")

            for section in chapter.get("sections", []):
                heading = section.get("heading", "No Heading")
                paragraphs = section.get("paragraphs", [])

                if not isinstance(paragraphs, list):
                    logger.warning(
                        f"Paragraphs in '{heading}' (chapter {chapter_id}) "
                        "is not a list. Skipping."
                    )
                    continue

                for para in paragraphs:
                    if not isinstance(para, str) or not para.strip():
                        continue

                    # Deduplicate identical paragraphs in the same section
                    dedup_key = (chapter_id, heading, para.strip())
                    if dedup_key in seen_texts:
                        logger.debug(f"Duplicate paragraph skipped in '{heading}'")
                        continue
                    seen_texts.add(dedup_key)

                    self.segments.append(
                        TextSegment(
                            text=para.strip(),
                            chapter_id=chapter_id,
                            chapter_title=chapter_title,
                            heading=heading,
                        )
                    )

    def get_all_text(self) -> str:
        """Return all paragraph text concatenated (for corpus-level analysis)."""
        return " ".join(seg.text for seg in self.segments)

    def get_segments(self) -> List[TextSegment]:
        """Return all extracted TextSegment objects."""
        return self.segments

    def get_book_title(self) -> str:
        """Return the book title."""
        return self.book_title

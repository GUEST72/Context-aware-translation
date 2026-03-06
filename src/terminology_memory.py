"""
terminology_memory.py - Terminology Memory Storage Module

Responsible for:
    - Assembling final TerminologyEntry records
    - Serializing the terminology memory to JSON
    - Loading / merging terminology memories from disk
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TerminologyEntry:
    """A single entry in the terminology memory database."""

    term: str
    normalized_term: str
    translation_ar: Optional[str] = None
    definition: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    frequency: int = 0
    confidence: float = 0.0
    source_locations: List[Dict[str, Any]] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)

    def to_dict(self, include_embedding: bool = True) -> Dict[str, Any]:
        """Convert to a plain dictionary for JSON serialization."""
        d = asdict(self)
        if not include_embedding:
            d["embedding"] = []
        return d


class TerminologyMemory:
    """
    In-memory terminology database.

    Provides methods to add entries, serialize to JSON,
    and load from a previously saved file.
    """

    def __init__(self):
        self.entries: Dict[str, TerminologyEntry] = {}  # normalized_term -> entry
        self.book_title: str = ""

    def add_entry(self, entry: TerminologyEntry) -> None:
        """Add or merge a terminology entry."""
        key = entry.normalized_term

        if key in self.entries:
            existing = self.entries[key]
            # Merge frequency
            existing.frequency += entry.frequency
            # Merge aliases
            for alias in entry.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            # Merge sources
            for loc in entry.source_locations:
                if loc not in existing.source_locations:
                    existing.source_locations.append(loc)
            # Keep highest confidence
            existing.confidence = max(existing.confidence, entry.confidence)
            # Keep first non-null definition
            if existing.definition is None and entry.definition is not None:
                existing.definition = entry.definition
            # Keep first non-null translation
            if existing.translation_ar is None and entry.translation_ar is not None:
                existing.translation_ar = entry.translation_ar
            # Keep first non-empty embedding
            if not existing.embedding and entry.embedding:
                existing.embedding = entry.embedding
        else:
            self.entries[key] = entry

    def to_dict(self, include_embeddings: bool = True) -> Dict[str, Any]:
        """Serialize the entire memory to a dictionary."""
        return {
            "book_title": self.book_title,
            "total_terms": len(self.entries),
            "terminologies": [
                entry.to_dict(include_embedding=include_embeddings)
                for entry in sorted(
                    self.entries.values(),
                    key=lambda e: e.confidence,
                    reverse=True,
                )
            ],
        }

    def save_json(
        self,
        filepath: str,
        include_embeddings: bool = True,
        indent: int = 2,
    ) -> None:
        """Save the terminology memory to a JSON file."""
        data = self.to_dict(include_embeddings=include_embeddings)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.info(
            f"Terminology memory saved to '{filepath}' "
            f"({len(self.entries)} entries)."
        )

    @classmethod
    def load_json(cls, filepath: str) -> "TerminologyMemory":
        """Load a terminology memory from a previously saved JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory = cls()
        memory.book_title = data.get("book_title", "")
        for item in data.get("terminologies", []):
            entry = TerminologyEntry(
                term=item.get("term", ""),
                normalized_term=item.get("normalized_term", ""),
                translation_ar=item.get("translation_ar"),
                definition=item.get("definition"),
                aliases=item.get("aliases", []),
                frequency=item.get("frequency", 0),
                confidence=item.get("confidence", 0.0),
                source_locations=item.get("source_locations", []),
                embedding=item.get("embedding", []),
            )
            memory.entries[entry.normalized_term] = entry

        logger.info(
            f"Loaded terminology memory from '{filepath}' "
            f"({len(memory.entries)} entries)."
        )
        return memory

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, term: str) -> bool:
        return term.lower().strip() in self.entries

    def get(self, term: str) -> Optional[TerminologyEntry]:
        return self.entries.get(term.lower().strip())

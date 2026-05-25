import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class _SentRecord:
    text:             str
    page:             int
    para_idx:         int    # position inside the combined window (0 = first)
    is_first_in_para: bool
    embedding:        Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class DefinitionResult:
    term:           str
    definition:     str         # best 1-2 sentences joined
    confidence:     float       # composite rerank score, 0.0 – 1.0
    source_page:    int
    candidates:     List[str] = field(default_factory=list)  # top-5 sentences before join


# ── Main class ─────────────────────────────────────────────────────────────────

class DefinitionExtractor:
    """
    RAG-style definition extraction with no section/subsection dependency.

    OFFLINE  build_space(book_json, terms)
    ─────────────────────────────────────
    For each term T, two windows are collected and merged:
      Window A — first occurrence window: the first N body paragraphs starting
                 from T's first appearance in the text (captures IS-A definitions
                 written when the author first introduces T).
      Window B — density window: the N paragraphs around the paragraph with the
                 highest T-mention density (captures dedicated explanation sections
                 that may appear later in the book).
      Window B is only added when it is more than N/2 paragraphs away from
      Window A, avoiding redundant embeddings.

    ONLINE   query(term, space)
    ───────────────────────────
      1. Embed four definition-style queries:
           "What is T?"  |  "T is"  |  "T refers to"  |  "definition of T"
      2. Cosine-similarity search within T's sentence space.
      3. Rerank each sentence by a composite score:
           - max query similarity      (dominant signal, 0.70)
           - soft IS-A verb after T    (0.12)
           - term in subject position  (0.10)
           - is first sentence of para (0.05)
           - position bonus (earlier = higher, 0.03)
      4. Take top-2 by rerank score, re-order by window position for
         readable prose, join as the definition.
    """

    _ISA_VERBS: frozenset = frozenset({
        "is", "are", "was", "were", "refers", "means", "represents",
        "allows", "provides", "defines", "enables", "consists", "denotes",
        "describes", "specifies", "encapsulates", "implements", "involves",
    })

    # Strips leading articles/determiners before checking subject position
    _STRIP_LEAD = re.compile(
        r"^(the|a|an|this|that|these|those|each|every)\s+",
        re.IGNORECASE,
    )

    # Paragraphs that are Table-of-Contents entries: contain 4+ consecutive
    # dots (leader dots) or the § section symbol.
    _TOC_RE = re.compile(r"\.{4,}|§")

    # Running page-header noise: "49 SOLID Principles / Single Responsibility..."
    _PAGE_HEADER_RE = re.compile(r"^\d+\s+\S.+\s/\s*\S")

    # Sentence-ending punctuation — used to detect mid-sentence paragraph splits
    _SENT_END_RE = re.compile(r"[.!?:]\s*$")

    def __init__(
        self,
        embed_model:       str = "all-MiniLM-L6-v2",
        window_paragraphs: int = 20,
    ) -> None:
        self.window_paragraphs = window_paragraphs
        print(f"  Loading embedding model '{embed_model}'...")
        self._embedder = SentenceTransformer(embed_model)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _flat_body_paragraphs(
        self, book_json: dict
    ) -> List[Tuple[str, int]]:
        """
        Flatten book JSON → [(paragraph_text, page_number)] for body paragraphs,
        excluding Table-of-Contents entries (leader dots, § symbol).
        """
        out = []
        for page in book_json.get("pages", []):
            pg = page.get("page", 0)
            for para in page.get("paragraphs", []):
                if para.get("type") == "body":
                    text = para["paragraph"].strip()
                    if (len(text) > 30
                            and not self._TOC_RE.search(text)
                            and not self._PAGE_HEADER_RE.match(text)):
                        out.append((text, pg))
        return out

    def _make_pattern(self, term: str) -> re.Pattern:
        """
        Whole-word pattern for single words; substring for phrases.
        Also normalises slashes/hyphens so 'open closed principle' matches
        'Open/Closed Principle' in the text.
        """
        term = term.strip()
        parts   = re.split(r"[\s/\-]+", term)
        escaped = r"[\s/\-]+".join(re.escape(p) for p in parts)
        if len(parts) == 1:
            return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
        return re.compile(escaped, re.IGNORECASE)

    def _find_first(
        self,
        term:       str,
        paragraphs: List[Tuple[str, int]],
    ) -> int:
        """
        Return the index of the first paragraph that contains `term`.
        Falls back to the first paragraph containing the first word of a phrase.
        Returns -1 if not found.
        """
        pattern = self._make_pattern(term)
        for i, (text, _) in enumerate(paragraphs):
            if pattern.search(text):
                return i

        if " " in term.strip():
            first_word = term.strip().split()[0]
            fb = re.compile(r"\b" + re.escape(first_word) + r"\b", re.IGNORECASE)
            for i, (text, _) in enumerate(paragraphs):
                if fb.search(text):
                    return i

        return -1

    def _best_occurrence(
        self,
        term:       str,
        paragraphs: List[Tuple[str, int]],
    ) -> int:
        """
        Find the paragraph most focused on the term by term density
        (occurrences / word count).  Returns -1 if the term never appears.
        """
        pattern    = self._make_pattern(term)
        best_idx   = -1
        best_score = 0.0

        for i, (text, _) in enumerate(paragraphs):
            hits = len(pattern.findall(text))
            if hits == 0:
                continue
            word_count = max(len(text.split()), 1)
            if word_count < 10:
                continue
            density = hits / word_count
            if density > best_score:
                best_score = density
                best_idx   = i

        if best_idx != -1:
            return max(0, best_idx - 2)

        return -1

    def _window_sentences(
        self,
        paragraphs:      List[Tuple[str, int]],
        start:           int,
        para_idx_offset: int = 0,
    ) -> List[_SentRecord]:
        """
        Extract sentences from `window_paragraphs` paragraphs starting at `start`.

        `para_idx_offset` is added to every record's para_idx so that Window B
        records sort after all Window A records during reranking.

        PDF parsers often split a single sentence across two paragraphs at a
        line break.  We detect this by checking whether a paragraph ends without
        sentence-closing punctuation and merge accordingly.
        """
        end    = min(start + self.window_paragraphs, len(paragraphs))
        window = paragraphs[start:end]

        # Step 1 — merge mid-sentence paragraph splits
        merged: List[Tuple[str, int, int]] = []   # (text, page, para_offset)
        cur_text, cur_page, cur_off = "", 0, 0

        for offset, (text, page) in enumerate(window):
            if cur_text:
                if not self._SENT_END_RE.search(cur_text):
                    cur_text = cur_text.rstrip() + " " + text.lstrip()
                else:
                    merged.append((cur_text, cur_page, cur_off))
                    cur_text, cur_page, cur_off = text, page, offset
            else:
                cur_text, cur_page, cur_off = text, page, offset

        if cur_text:
            merged.append((cur_text, cur_page, cur_off))

        # Step 2 — split into sentences
        records: List[_SentRecord] = []
        for m_text, page, para_offset in merged:
            for s_idx, sent in enumerate(sent_tokenize(m_text)):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                records.append(_SentRecord(
                    text=sent,
                    page=page,
                    para_idx=para_idx_offset + para_offset,
                    is_first_in_para=(s_idx == 0),
                ))
        return records

    def _cosine(self, q: np.ndarray, docs: np.ndarray) -> np.ndarray:
        q    = q    / (np.linalg.norm(q,    axis=-1, keepdims=True) + 1e-9)
        docs = docs / (np.linalg.norm(docs, axis=-1, keepdims=True) + 1e-9)
        return q @ docs.T                               # (n_queries, n_docs)

    def _rerank(
        self,
        sim:    float,
        rec:    _SentRecord,
        term:   str,
        n_para: int,
    ) -> float:
        # Similarity is the dominant signal; structural bonuses are secondary
        score      = 0.70 * sim
        text_lower = rec.text.lower()
        term_lower = term.lower()

        # Soft IS-A verb immediately after the term
        pos = text_lower.find(term_lower)
        if pos != -1:
            after_term = text_lower[pos + len(term_lower):].lstrip(" ,;")
            first_word = after_term.split()[0] if after_term.split() else ""
            if first_word in self._ISA_VERBS:
                score += 0.12

        # Term appears in subject position
        stripped = self._STRIP_LEAD.sub("", rec.text).lower()
        if stripped.startswith(term_lower):
            score += 0.10

        # First sentence of paragraph tends to introduce the concept
        if rec.is_first_in_para:
            score += 0.05

        # Earlier paragraphs in the combined window are slightly preferred
        score += 0.03 * max(0.0, 1.0 - rec.para_idx / max(n_para, 1))

        return score

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_space(
        self,
        book_json: dict,
        terms:     List[str],
    ) -> Dict[str, List[_SentRecord]]:
        """
        Build the sentence-embedding space for every term using a dual-window
        strategy: Window A (first occurrence) + Window B (density peak), merged
        with deduplication.

        Parameters
        ----------
        book_json : dict   Parsed book ({"pages": [...]})
        terms     : list   Term strings from TerminologyExtractor

        Returns
        -------
        Dict[term → List[_SentRecord]]   (embeddings attached to each record)
        """
        paragraphs = self._flat_body_paragraphs(book_json)
        space: Dict[str, List[_SentRecord]] = {}
        half_window = self.window_paragraphs // 2

        for term in terms:
            # Window A — first occurrence
            first = self._find_first(term, paragraphs)
            if first == -1:
                continue
            records_a = self._window_sentences(paragraphs, first, para_idx_offset=0)

            # Window B — density peak (only when far enough from Window A)
            density_start = self._best_occurrence(term, paragraphs)
            records_b: List[_SentRecord] = []
            if density_start != -1 and abs(density_start - first) > half_window:
                offset_b = self.window_paragraphs  # place Window B after Window A
                records_b = self._window_sentences(
                    paragraphs, density_start, para_idx_offset=offset_b
                )

            # Merge with deduplication by sentence text
            seen_texts: set = set()
            records: List[_SentRecord] = []
            for rec in records_a + records_b:
                if rec.text not in seen_texts:
                    seen_texts.add(rec.text)
                    records.append(rec)

            if not records:
                continue

            embs = self._embedder.encode(
                [r.text for r in records],
                show_progress_bar=False,
                batch_size=64,
            )
            for rec, emb in zip(records, embs):
                rec.embedding = emb

            space[term] = records

        return space

    def query(
        self,
        term:   str,
        space:  Dict[str, List[_SentRecord]],
        top_k:  int = 2,
    ) -> Optional[DefinitionResult]:
        """
        Retrieve the best definitional sentences for `term` from its space.

        Returns None if the term has no space entry.
        """
        records = space.get(term)
        if not records:
            return None

        queries = [
            f"What is {term}?",
            f"{term} is",
            f"{term} refers to",
            f"definition of {term}",
        ]
        q_embs  = self._embedder.encode(queries, show_progress_bar=False)
        r_embs  = np.array([r.embedding for r in records])

        sims     = self._cosine(q_embs, r_embs)  # (4, n_sents)
        best_sim = sims.max(axis=0)               # (n_sents,)
        n_para   = max(r.para_idx for r in records) + 1

        scored = sorted(
            enumerate(records),
            key=lambda x: self._rerank(float(best_sim[x[0]]), x[1], term, n_para),
            reverse=True,
        )

        top = scored[:top_k]
        # Re-order by window position so the joined text reads naturally
        top.sort(key=lambda x: (x[1].para_idx, not x[1].is_first_in_para))

        definition = " ".join(r.text for _, r in top)
        confidence = self._rerank(float(best_sim[top[0][0]]), top[0][1], term, n_para)

        return DefinitionResult(
            term=term,
            definition=definition,
            confidence=round(min(confidence, 1.0), 3),
            source_page=top[0][1].page,
            candidates=[records[i].text for i, _ in scored[:5]],
        )

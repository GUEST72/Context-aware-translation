import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import nltk
import spacy
from nltk.corpus import brown, reuters, stopwords, wordnet
from nltk.corpus import words as _nltk_words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for _pkg in (
    "punkt", "punkt_tab",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "stopwords", "brown", "reuters", "wordnet", "omw-1.4", "words",
):
    nltk.download(_pkg, quiet=True)


@dataclass
class TermEntry:
    term: str
    frequency: int
    weirdness: float
    pos: str                          # 'NOUN' or 'ADJ'
    source_pages: List[int] = field(default_factory=list)


class TerminologyExtractor:
    """
    Extracts domain-specific terminology from a structured book JSON.

    Pipeline
    --------
    Single-token terms
      1. POS-tag every body paragraph → keep Nouns and Adjectives only.
      2. Pre-process each token (on its original cased form):
         a. Lingo filter  – drop acronyms, camelCase, code identifiers, etc.
         b. Lemmatize + lowercase
         c. Minimum length guard (default ≥ 4 chars)
      3. Common-word filter – union of:
         - NLTK English stopwords
         - ~500 Google-Books most-frequent English words
         - Top-N words from Brown + Reuters background corpora
      4. Named-entity filter (spaCy NER) – drop tokens that are entity surface
         forms (persons, places, organisations) anywhere in the book.
      5. Fragment guard – reject low-frequency words absent from the English
         dictionary (catches PDF hyphenation tail-fragments).
      6. Weirdness scoring: Weirdness(w) = TF_book(w) / TF_background(w)

    Multi-word phrases (noun chunks)
      7. spaCy dependency parse → noun_chunks over every body paragraph.
      8. Strip leading determiners, apply lingo filter per token.
      9. Keep phrases with frequency ≥ min_phrase_freq.
         Background bigram frequency ≈ 0, so Weirdness naturally ranks them
         alongside high-frequency single tokens.

    Both streams are merged and returned sorted by Weirdness (descending).
    """

    # Penn Treebank POS tags for nouns and adjectives
    _NOUN_TAGS: Set[str] = frozenset({"NN", "NNS", "NNP", "NNPS"})
    _ADJ_TAGS:  Set[str] = frozenset({"JJ", "JJR", "JJS"})

    # Determiners / articles stripped from the front of spaCy noun chunks
    _PHRASE_STRIP_LEAD: Set[str] = frozenset({
        "the","a","an","this","these","those","each","every",
        "its","their","our","your","my","his","her","some","any","all","both",
    })

    # Only block PERSON entities. GPE/LOC/ORG are excluded because they
    # mis-label CS terms: "Singleton" → GPE (English town), "Factory" → ORG.
    _BLOCK_ENT_LABELS: Set[str] = frozenset({"PERSON"})

    # ToC and page-header patterns — paragraphs skipped during NER to avoid
    # polluting the entity blocklist with design-pattern names.
    _TOC_RE = re.compile(r"\.{4,}|§")
    _PAGE_HEADER_RE = re.compile(r"^\d+\s+\S.+\s/\s*\S")

    # Code-like token patterns that invalidate a phrase (applied per token)
    _PHRASE_LINGO_RE = re.compile(r"^\d|\.")

    # Applied on the original (cased) token before lemmatization
    _LINGO_RE = re.compile(
        r"^[A-Z]{2,6}$"        # pure acronyms: TCP, HTTP, LSTM
        r"|^\d"                 # starts with a digit: 3rd, 10x
        r"|[_/\\]"              # path / code separators
        r"|(?<=[a-z])[A-Z]"    # camelCase boundary: backProp, iPhone
        r"|\."                  # any dot: file extensions, method calls, attr access
        r"|^v\d"               # version tokens: v1, v2.3
        r"|-$"                  # PDF hyphenation artifact: "imple-", "com-"
        r"|^-"                  # leading hyphen fragment
    )

    # Standalone fragments from PDF line-break hyphenation.
    # The hyphen-ending half is caught by _LINGO_RE; this set catches the tail half,
    # e.g. "con-\ncrete" → "crete", "com-\nposite" → "posite".
    _SUFFIX_FRAGMENTS: Set[str] = {
        # morphological suffixes (short)
        "tion","tions","sion","ment","ments","ness","ity","ities",
        "ble","ical","ive","ous","ful","less","tor","ple","ling",
        "ing","ers","ings","ated","izer","iser",
        # longer tail fragments seen in practice
        "crete","posite","patible","struction","structions",
        "vides","lection","lections","sive","sively",
        "ware","ponent","ponents","tation","tations",
        # obscure real-words that are actually PDF hyphenation tails
        "mand","mands","scriber","scribers","mentation","mentations",
        "eral","aration","arations","ferent","rithm","rithms",
        "ator","ators","tional","tionals","iors","patible",
    }

    # Google Books ~500 most common English words (hardcoded for offline use)
    _GOOGLE_COMMON: Set[str] = {
        "the","be","to","of","and","a","in","that","have","it","for","not",
        "on","with","he","as","you","do","at","this","but","his","by","from",
        "they","we","say","her","she","or","an","will","my","one","all",
        "would","there","their","what","so","up","out","if","about","who",
        "get","which","go","me","when","make","can","like","time","no","just",
        "him","know","take","people","into","year","your","good","some",
        "could","them","see","other","than","then","now","look","only","come",
        "its","over","think","also","back","after","use","two","how","our",
        "work","first","well","way","even","new","want","because","any",
        "these","give","day","most","us","great","between","need","large",
        "often","hand","high","place","hold","big","play","small","number",
        "off","always","move","try","kind","picture","again","change","live",
        "point","little","may","example","begin","life","those","both","paper",
        "together","got","group","run","important","until","side","car","mile",
        "night","walk","white","sea","began","grow","took","river","four",
        "carry","state","once","book","hear","stop","without","second","later",
        "idea","enough","eat","face","watch","far","real","almost","let",
        "above","girl","sometimes","cut","young","talk","soon","list","song",
        "leave","family","body","music","color","stand","sun","fish","area",
        "mark","dog","horse","problem","complete","room","knew","since","ever",
        "piece","told","usually","friend","easy","heard","order","red","door",
        "sure","become","top","ship","across","today","during","short",
        "better","best","however","low","hours","black","happened","whole",
        "measure","remember","early","fast","several","himself","toward",
        "five","step","morning","passed","true","hundred","against","table",
        "north","slowly","money","map","farm","pulled","draw","voice","power",
        "town","fine","drive","led","cry","dark","machine","note","waited",
        "plan","figure","star","box","field","rest","correct","lead","able",
        "road","decide","thousand","fact","possible","age","quite","deep",
        "next","something","long","word","said",
    }

    def __init__(
        self,
        min_weirdness: float = 1.5,
        min_word_len: int = 4,
        top_bg_common: int = 3000,
        min_non_dict_freq: int = 7,
        min_phrase_freq: int = 2,
    ) -> None:
        self.min_weirdness = min_weirdness
        self.min_word_len = min_word_len
        self.min_non_dict_freq = min_non_dict_freq
        # Multi-word phrases must appear at least this many times in the book.
        # Background bigram freq ≈ 0 for domain phrases, so frequency is the
        # only meaningful filter here.
        self.min_phrase_freq = min_phrase_freq

        self._lemmatizer = WordNetLemmatizer()
        self._stop_words: Set[str] = set(stopwords.words("english"))
        self._dict_words: Set[str] = {w.lower() for w in _nltk_words.words()}

        try:
            # Both tagger (POS) and parser (dep) are required for noun_chunks.
            # Only lemmatizer is unused and disabled for speed.
            self._nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

        self._bg_freq, self._total_bg = self._build_background()
        self._common_set = self._build_common_set(top_bg_common)

    # ── Background corpus ──────────────────────────────────────────────────────

    def _build_background(self) -> Tuple[Counter, int]:
        """Frequency table built from Brown + Reuters (NLTK corpora)."""
        words = [
            w.lower()
            for corpus in (brown, reuters)
            for w in corpus.words()
            if w.isalpha()
        ]
        freq = Counter(words)
        return freq, sum(freq.values())

    def _build_common_set(self, top_n: int) -> Set[str]:
        """Union of stop words, Google-common words, and top-N background words."""
        top_bg = {w for w, _ in self._bg_freq.most_common(top_n)}
        return top_bg | self._stop_words | self._GOOGLE_COMMON

    # ── Per-token predicates ───────────────────────────────────────────────────

    def _is_lingo(self, original_token: str) -> bool:
        """True if the token is a code artifact or a PDF hyphenation fragment."""
        if self._LINGO_RE.search(original_token):
            return True
        # Catch suffix-only fragments produced by PDF line-break splitting
        return original_token.lower() in self._SUFFIX_FRAGMENTS

    def _is_common(self, lemma: str) -> bool:
        return lemma in self._common_set

    # ── Step 1 & 2: POS-tag → lingo filter → lemmatize ────────────────────────

    def _tokenize_paragraph(self, text: str) -> List[Tuple[str, str]]:
        """
        Return (lemma, pos) pairs for nouns/adjectives in text that survive
        the lingo filter and minimum-length guard.
        """
        tagged = nltk.pos_tag(word_tokenize(text))
        out: List[Tuple[str, str]] = []
        for word, tag in tagged:
            if tag not in self._NOUN_TAGS and tag not in self._ADJ_TAGS:
                continue
            if self._is_lingo(word):
                continue
            pos = "NOUN" if tag in self._NOUN_TAGS else "ADJ"
            wn_pos = wordnet.NOUN if pos == "NOUN" else wordnet.ADJ
            lemma = self._lemmatizer.lemmatize(word.lower(), pos=wn_pos)
            if len(lemma) >= self.min_word_len:
                out.append((lemma, pos))
        return out

    # ── Steps 4 & 7: single spaCy pass — NER + noun-chunk phrases ────────────

    def _analyse_paragraphs(
        self, pages: list
    ) -> Tuple[Set[str], Counter, Dict[str, Set[int]]]:
        """
        One pass over every body paragraph with the full spaCy pipeline:
          - NER   → entity token blocklist (step 4)
          - Dependency parse → noun_chunks → phrase candidates (step 7)

        Returns
        -------
        entities   : Set[str]          lowercased entity tokens
        phrase_freq: Counter           phrase → count across whole book
        phrase_pages: Dict[str,Set]    phrase → set of page numbers
        """
        entities:     Set[str]           = set()
        phrase_freq:  Counter            = Counter()
        phrase_pages: Dict[str, Set[int]] = {}

        for page in pages:
            page_num = page.get("page", 0)
            for para in page.get("paragraphs", []):
                if para.get("type") in ("chapter", "section", "subsection"):
                    continue
                text = para["paragraph"]
                is_noise = (self._TOC_RE.search(text)
                            or self._PAGE_HEADER_RE.match(text))
                doc = self._nlp(text)

                # NER — collect PERSON tokens as a blocklist (author names, etc.).
                # Skip ToC and page-header paragraphs: they cause spaCy to
                # label design-pattern names as ORG/GPE (e.g. "§ Singleton" → GPE,
                # "44 … / Favor Composition Over Inheritance" → ORG).
                if not is_noise:
                    for ent in doc.ents:
                        if ent.label_ not in self._BLOCK_ENT_LABELS:
                            continue
                        # Skip entities whose surface starts with a non-letter
                        # (PDF bullet symbols like  make spaCy tag
                        # "✧ Inheritance" as PERSON, causing false blocklist hits).
                        surface = ent.text.strip()
                        if not surface or not surface[0].isalpha():
                            continue
                        for tok in surface.lower().split():
                            if tok.isalpha():
                                entities.add(tok)

                # Noun chunks — multi-word phrase candidates
                for chunk in doc.noun_chunks:
                    tokens = [
                        t for t in chunk
                        if not t.is_punct
                        and t.text.lower() not in self._PHRASE_STRIP_LEAD
                    ]
                    if len(tokens) < 2:
                        continue
                    # Drop chunks that contain code-like tokens
                    if any(self._PHRASE_LINGO_RE.search(t.text) for t in tokens):
                        continue
                    phrase = " ".join(t.text.lower() for t in tokens)
                    if len(phrase) < 6:
                        continue
                    phrase_freq[phrase] += 1
                    phrase_pages.setdefault(phrase, set()).add(page_num)

        return entities, phrase_freq, phrase_pages

    # ── Step 5: Weirdness scoring ──────────────────────────────────────────────

    def _weirdness(self, lemma: str, book_count: int, total_book: int) -> float:
        """
        Weirdness(w) = TF_book(w) / TF_background(w)

        A score > 1.0 means the word appears proportionally more in this book
        than in the general-purpose background corpus — i.e. it is domain-specific.
        """
        tf_book = book_count / total_book
        tf_bg   = self._bg_freq.get(lemma, 0) / self._total_bg
        return tf_book / (tf_bg + 1e-9)   # epsilon avoids division by zero

    # ── Main entry point ───────────────────────────────────────────────────────

    def extract(self, book_json: dict) -> List[TermEntry]:
        """
        Run the full pipeline on a parsed book JSON dict.

        Parameters
        ----------
        book_json : dict
            The structure produced by the parser: {"pages": [...]}

        Returns
        -------
        List[TermEntry]
            Domain-specific terms ranked by weirdness score (descending).
        """
        pages = book_json.get("pages", [])

        # Pass 1 — accumulate per-lemma frequency and page occurrences
        book_freq:  Counter               = Counter()
        token_pos:  Dict[str, str]        = {}
        term_pages: Dict[str, Set[int]]   = {}

        for page in pages:
            page_num = page.get("page", 0)
            for para in page.get("paragraphs", []):
                # Skip structural headings — they bias frequency counts
                if para.get("type") in ("chapter", "section", "subsection"):
                    continue
                for lemma, pos in self._tokenize_paragraph(para["paragraph"]):
                    book_freq[lemma] += 1
                    token_pos.setdefault(lemma, pos)
                    term_pages.setdefault(lemma, set()).add(page_num)

        total_book = sum(book_freq.values()) or 1

        # Pass 2 — single spaCy pass: entity blocklist + noun-chunk phrases
        entities, phrase_freq, phrase_pages = self._analyse_paragraphs(pages)

        # Pass 3 — filter and score single-token terms
        results: List[TermEntry] = []
        for lemma, freq in book_freq.items():
            if self._is_common(lemma):
                continue
            if lemma in entities:
                continue
            # Fragment guard: PDF hyphenation produces tail-fragments ("rithm",
            # "scriber") that look like nouns but are not real words. Reject any
            # lemma absent from the English dictionary unless it occurs often
            # enough to be a genuine domain term (e.g. "flyweight", "foreach").
            if lemma not in self._dict_words and freq < self.min_non_dict_freq:
                continue
            score = self._weirdness(lemma, freq, total_book)
            if score >= self.min_weirdness:
                results.append(TermEntry(
                    term=lemma,
                    frequency=freq,
                    weirdness=round(score, 4),
                    pos=token_pos[lemma],
                    source_pages=sorted(term_pages[lemma]),
                ))

        # Pass 4 — score and add multi-word phrases
        # Background bigram frequency is 0 for domain phrases, so _weirdness()
        # naturally produces a large score comparable to rare single tokens.
        for phrase, freq in phrase_freq.items():
            if freq < self.min_phrase_freq:
                continue
            score = self._weirdness(phrase, freq, total_book)
            results.append(TermEntry(
                term=phrase,
                frequency=freq,
                weirdness=round(score, 4),
                pos="PHRASE",
                source_pages=sorted(phrase_pages[phrase]),
            ))

        results.sort(key=lambda e: e.weirdness, reverse=True)
        return results

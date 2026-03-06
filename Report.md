# Walkthrough

## Terminology Memory Extraction System:

This document explains the entire system **step by step**, module by module, function by function. 

---

## Table of Contents

1. [What is Terminology Extraction?](#1-what-is-terminology-extraction)
2. [Key NLP Concepts](#2-key-nlp-concepts)
3. [System Overview](#3-system-overview)
4. [Module-by-Module Walkthrough](#4-module-by-module-walkthrough)
   - [4.1 text_loader.py](#41-text_loaderpy)
   - [4.2 term_extractor.py](#42-term_extractorpy)
   - [4.3 term_ranker.py](#43-term_rankerpy)
   - [4.4 term_normalizer.py](#44-term_normalizerpy)
   - [4.5 alias_detector.py](#45-alias_detectorpy)
   - [4.6 definition_extractor.py](#46-definition_extractorpy)
   - [4.7 translator.py](#47-translatorpy)
   - [4.8 embedding_generator.py](#48-embedding_generatorpy)
   - [4.9 terminology_memory.py](#49-terminology_memorypy)
   - [4.10 pipeline.py](#410-pipelinepy)
   - [4.11 main.py](#411-mainpy)
5. [Design Decisions Explained](#5-design-decisions-explained)
6. [Glossary of Technical Terms](#6-glossary-of-technical-terms)

---

## 1. What is Terminology Extraction?

**Terminology extraction** is the process of automatically identifying domain-specific terms from a body of text.

### Why is it important?

When translating a technical book (e.g., a machine learning textbook), you'll encounter hundreds of specialised terms like:

- "neural network"
- "gradient descent"
- "attention mechanism"

Each of these must be translated **consistently** across the entire book. If one translator writes "شبكة عصبية" and another writes "شبكة عصبونية" for "neural network", readers will be confused.

A **Terminology Memory** solves this by providing a single, structured database where every term is recorded with its canonical form, definition, translation, and location.

### What this system does

```
Input:  A structured JSON file representing an academic book
Output: A Terminology Memory JSON file containing every detected term
```

The system does NOT translate the full book — it only extracts and organises the terminology.

---

## 2. Key NLP Concepts

Before diving into the code, let's understand the core NLP concepts used in this system.

### 2.1 Tokenization

**Tokenization** splits text into individual units (tokens).

```
Input:  "Machine learning is powerful."
Tokens: ["Machine", "learning", "is", "powerful", "."]
```

We use spaCy's tokenizer, which handles edge cases like contractions and hyphenated words.

### 2.2 Part-of-Speech (POS) Tagging

Each token is assigned a grammatical tag:

| Token      | POS Tag |
|------------|---------|
| Machine    | NOUN    |
| learning   | NOUN    |
| is         | AUX     |
| powerful   | ADJ     |

We use POS tags to filter candidates. Terms should be **noun-heavy** — we don't want verbs or function words.

### 2.3 Noun Phrase Chunking

spaCy can identify **noun phrases** — contiguous groups of words centred around a noun:

```
"The attention mechanism processes the input sequence"
 ^^^^^^^^^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^^^^^^
 noun phrase                       noun phrase
```

This is one of our primary extraction methods.

### 2.4 Lemmatization

Lemmatization reduces a word to its dictionary form:

```
"networks"  → "network"
"processing" → "processing" (gerund, kept as-is in nominal contexts)
"learned"   → "learn"
```

This is how we merge "Neural Networks" and "neural network" into a single entry.

### 2.5 TF-IDF

**Term Frequency–Inverse Document Frequency** measures how important a word is within a collection of documents.

```
TF(t)  = (Number of times term t appears in a document) / (Total terms in the document)
IDF(t) = log((Total documents + 1) / (Documents containing t + 1)) + 1
TF-IDF = TF × IDF
```

A term that appears frequently in a few documents but not everywhere gets a **high** TF-IDF score. Generic words like "the" or "is" get low scores because they appear everywhere.

### 2.6 Word Embeddings

Word embeddings are **dense vector representations** of words/phrases:

```
"machine learning" → [0.12, -0.45, 0.78, ..., 0.33]  (384 dimensions)
```

Semantically similar terms have similar vectors:
- cos_similarity("machine learning", "deep learning") ≈ 0.85
- cos_similarity("machine learning", "table tennis") ≈ 0.05

We use these for downstream semantic search and clustering.

---

## 3. System Overview

### Data Flow

```
┌──────────────┐
│  Book JSON   │  ← The input file
└──────┬───────┘
       │
       ▼
  text_loader.py        → Parse JSON, validate, extract TextSegments
       │
       ▼
  term_extractor.py     → Noun phrase chunking + n-gram extraction
       │
       ▼
  term_ranker.py        → Compute TF-IDF, produce confidence scores
       │
       ▼
  term_normalizer.py    → Lemmatize, lowercase, merge duplicates
       │
       ▼
  alias_detector.py     → Find abbreviations like "NLP", "CNN"
       │
       ▼
  definition_extractor.py → Find "X is Y" patterns
       │
       ▼
  translator.py         → Translate terms to Arabic (optional)
       │
       ▼
  embedding_generator.py → Generate semantic vectors (optional)
       │
       ▼
  terminology_memory.py → Assemble and save the final JSON
```

### Why this order?

1. **Load first** — you can't process what you haven't read.
2. **Extract candidates** — get a raw list of potential terms.
3. **Rank** — score them before normalizing, because ranking uses the raw surface forms.
4. **Normalize** — merge duplicates after ranking so we can aggregate scores.
5. **Aliases & Definitions** — these run on the original text, not on candidates, so they can run at any point.
6. **Translate & Embed** — these are the most expensive steps, so we do them last and only on the final term list.

---

## 4. Module-by-Module Walkthrough

---

### 4.1 `text_loader.py`

**Purpose:** Load a structured book JSON and convert it into a list of `TextSegment` objects.

#### Key Classes

**`TextSegment`**
```python
class TextSegment:
    def __init__(self, text, chapter_id, chapter_title, heading):
        self.text = text                # The paragraph text
        self.chapter_id = chapter_id    # e.g. 1
        self.chapter_title = chapter_title  # e.g. "Machine Learning"
        self.heading = heading          # e.g. "Introduction"
```

Each paragraph becomes one `TextSegment`. We attach metadata so that later, when a term is found, we know **where** in the book it came from.

**`BookLoader`**

This class:
1. Opens and parses the JSON file
2. Validates the structure (checks for required keys)
3. Extracts every paragraph into a `TextSegment`
4. De-duplicates identical paragraphs in the same section

#### Validation Logic (line by line)

```python
REQUIRED_BOOK_KEYS = {"book_title", "chapters"}
```
The top-level JSON **must** have these two keys. If either is missing, we raise an error.

```python
missing = self.REQUIRED_BOOK_KEYS - set(self.book_data.keys())
if missing:
    raise ValueError(f"Missing required book keys: {missing}")
```
We use set subtraction to find which required keys are absent.

#### De-duplication

```python
dedup_key = (chapter_id, heading, para.strip())
if dedup_key in seen_texts:
    continue
seen_texts.add(dedup_key)
```

We create a unique key from (chapter, heading, paragraph text). If we've seen this exact combination before, we skip it. This handles cases where the input JSON accidentally contains duplicate paragraphs.

#### Why TextSegment instead of plain strings?

We need to track **where** each term was found. If we just extracted raw text, we'd lose the chapter/section information. By wrapping each paragraph in a `TextSegment`, we carry the metadata through the entire pipeline.

---

### 4.2 `term_extractor.py`

**Purpose:** Extract candidate terminology from text using two NLP methods.

#### Method 1: Noun Phrase Chunking

```python
def _extract_noun_phrases(self, doc, source):
    for chunk in doc.noun_chunks:
        term_text = self._clean_span(chunk)
        if term_text:
            self._register(term_text, source)
```

spaCy's `doc.noun_chunks` yields grammatical noun phrases. For example:

```
"Machine learning is a field of artificial intelligence."
 ^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^
 chunk 1                       chunk 2
```

We clean each chunk (remove determiners like "a", "the") and register it as a candidate.

#### Method 2: N-gram Extraction

```python
def _extract_ngrams(self, doc, source):
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    for n in range(1, self.max_tokens + 1):
        for i in range(len(tokens) - n + 1):
            span = tokens[i : i + n]
```

An **n-gram** is a contiguous sequence of n tokens. We generate all 1-grams, 2-grams, 3-grams, and 4-grams:

```
Text: "deep learning model"
1-grams: "deep", "learning", "model"
2-grams: "deep learning", "learning model"
3-grams: "deep learning model"
```

We only keep n-grams where **at least one token is a noun** — this prevents us from extracting phrases like "very quickly".

#### Why both methods?

Noun phrase chunking is **precise** — it uses grammatical analysis. But it can miss terms that span unusual grammatical structures.

N-gram extraction is **broad** — it catches everything. But it generates many false positives.

By combining both, we get high recall (catch most terms) while using filtering to improve precision.

#### Filtering Rules

```python
def _apply_filters(self):
    # Rule 1: reject if > max tokens
    if len(tokens) > self.max_tokens:        # default 5

    # Rule 2: reject if stopword ratio > 60%
    sw_ratio = sum(1 for t in tokens if t in self.STOPWORDS) / len(tokens)
    if sw_ratio > 0.6:

    # Rule 3: reject if contains digits
    if re.search(r"\d", key):

    # Rule 4: reject if frequency < min_freq
    if cand.occurrences < self.min_freq:     # default 2

    # Rule 5: reject single-character or empty
    if len(key) <= 2:
```

**Why these thresholds?**

- **Max 5 tokens:** Legitimate terms rarely exceed 5 words. "Bidirectional encoder representations from transformers" (5 words) is about the longest common term.
- **60% stopwords:** "the process of the" is 75% stopwords — not a term. "machine learning" is 0% — definitely a term.
- **No digits:** "Chapter 3" or "Section 2.1" are not terms.
- **Min frequency 2:** A term appearing only once is likely noise or not domain-specific.

#### CandidateTerm Class

```python
class CandidateTerm:
    def __init__(self, surface_form):
        self.surface_form = surface_form  # e.g. "neural network"
        self.occurrences = 0              # how many times it appeared
        self.source_locations = []        # where it appeared
```

We track the surface form (exact text), count, and locations. This gets passed to the ranker and normalizer.

---

### 4.3 `term_ranker.py`

**Purpose:** Rank terms by importance using TF-IDF and produce a confidence score.

#### How TF-IDF Works Here

```python
# Term Frequency: how often the term appears relative to all terms
tf = cand.occurrences / total_occurrences

# Inverse Document Frequency: penalise terms that appear in every segment
idf = math.log((1 + corpus_size) / (1 + document_freq[key])) + 1

# Combined score
score = tf * idf
```

**Example:**

Suppose "machine learning" appears 12 times across 30 segments, and appears in 8 of those segments.

```
tf  = 12 / 500 = 0.024  (500 total term occurrences)
idf = log((1 + 30) / (1 + 8)) + 1 = log(3.44) + 1 ≈ 2.24
tfidf = 0.024 × 2.24 ≈ 0.054
```

A generic word like "model" might appear in 25 of 30 segments:

```
idf = log((1 + 30) / (1 + 25)) + 1 = log(1.19) + 1 ≈ 1.18
```

The lower IDF makes generic words score lower.

#### Normalization to [0, 1]

```python
# Find the highest TF-IDF score
max_tfidf = max(self.tfidf_scores.values())

# Divide every score by the maximum
confidence = score / max_tfidf
```

The highest-ranked term gets confidence = 1.0, and all others are proportionally lower.

---

### 4.4 `term_normalizer.py`

**Purpose:** Convert surface forms into canonical normalised terms and merge duplicates.

#### Normalization Steps

```python
def normalize(self, term):
    # Step 1: Lowercase
    text = term.lower().strip()          # "Neural Networks" → "neural networks"

    # Step 2: Remove edge punctuation
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)

    # Step 3: Lemmatize each token
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        lemmas.append(token.lemma_)       # "networks" → "network"

    # Step 4: Rejoin
    normalized = " ".join(lemmas)          # "neural network"
```

#### Merging Duplicates

```python
def normalize_candidates(self, candidates):
    merged = {}
    for key, cand in candidates.items():
        canonical = self.normalize(cand.surface_form)

        if canonical not in merged:
            merged[canonical] = {
                "surface_forms": [],
                "occurrences": 0,
                "source_locations": [],
            }

        entry = merged[canonical]
        entry["surface_forms"].append(cand.surface_form)
        entry["occurrences"] += cand.occurrences
        # ... merge source locations
```

**Example:**

Before normalization:
```
"Neural Networks"  → occurrences: 3
"neural network"   → occurrences: 5
"neural networks"  → occurrences: 2
```

After normalization, all three map to `"neural network"`:
```
"neural network" → occurrences: 10, surface_forms: ["neural networks", "neural network", ...]
```

**Why normalize?** Without normalization, the same concept would appear as three separate entries in the terminology memory. By merging, we get accurate frequency counts and a single canonical form.

---

### 4.5 `alias_detector.py`

**Purpose:** Detect abbreviations/aliases from patterns like `Full Term (ABBR)`.

#### The Regex Pattern

```python
_ABBREV_PATTERN = re.compile(
    r"([A-Z][\w\s\-]{2,80}?)\s*\(([A-Z][A-Z0-9\-]{1,15})\)"
)
```

Let's break this down:

| Part                    | Meaning                                              |
|-------------------------|------------------------------------------------------|
| `([A-Z][\w\s\-]{2,80}?)` | Capture group 1: starts with uppercase, 2-80 word/space/hyphen chars (lazy) |
| `\s*`                   | Optional whitespace before the parenthesis            |
| `\(`                    | Literal opening parenthesis                           |
| `([A-Z][A-Z0-9\-]{1,15})` | Capture group 2: abbreviation — all caps, 2-16 chars |
| `\)`                    | Literal closing parenthesis                           |

**Example match:**
```
Text: "Natural Language Processing (NLP) is a subfield"
Group 1: "Natural Language Processing"
Group 2: "NLP"
```

#### Plausibility Check

Not every parenthetical is an abbreviation. We verify:

```python
def _is_plausible_abbreviation(full_form, abbreviation):
    words = full_form.split()
    significant = [w for w in words if len(w) > 2]
    initials = "".join(w[0] for w in significant).upper()

    # "Natural Language Processing" → initials "NLP"
    # abbreviation "NLP" → match!
    return initials == abbreviation.upper()
```

This prevents false positives like `"example text (EXAMPLE)"` from being treated as an abbreviation.

---

### 4.6 `definition_extractor.py`

**Purpose:** Find definitions in the text using regex patterns.

#### Patterns

```python
_PATTERNS = [
    # "X is a/an/the Y"
    re.compile(r"([A-Z][\w\s\-]{2,60}?)\s+is\s+(a|an|the)\s+(.+?)(?:\.|$)", re.IGNORECASE),

    # "X refers to Y"
    re.compile(r"([A-Z][\w\s\-]{2,60}?)\s+refers\s+to\s+(.+?)(?:\.|$)", re.IGNORECASE),

    # "X can be defined as Y"
    re.compile(r"([A-Z][\w\s\-]{2,60}?)\s+can\s+be\s+defined\s+as\s+(.+?)(?:\.|$)", re.IGNORECASE),

    # "X is defined as Y"
    re.compile(r"([A-Z][\w\s\-]{2,60}?)\s+is\s+defined\s+as\s+(.+?)(?:\.|$)", re.IGNORECASE),
]
```

**Example:**

```
Text: "Machine learning is a branch of artificial intelligence."
             |                    |
         term (group 1)     definition (groups 2+3)

Result: term = "machine learning"
        def  = "a branch of artificial intelligence"
```

#### Why regex and not ML-based extraction?

For a **baseline system**, regex is:
- **Deterministic** — the same input always gives the same output.
- **Fast** — no model loading or inference.
- **Transparent** — you can read the pattern and understand what it matches.

The drawback is that it only catches simple definitional sentences. Complex definitions spanning multiple sentences are missed. This is acknowledged as a limitation.

#### Fuzzy Lookup

```python
def get_definition(self, term):
    key = term.lower().strip()

    # Exact match
    if key in self.definitions:
        return self.definitions[key]

    # Substring match
    for stored_key, definition in self.definitions.items():
        if key in stored_key or stored_key in key:
            return definition
```

Why substring matching? The extracted definition key might be "machine learning" while the normalized term is "machine learning" — an exact match. But sometimes the regex captures extra context, e.g., "deep machine learning" as the key. The substring fallback handles these mismatches.

---

### 4.7 `translator.py`

**Purpose:** Translate individual terms from English to Arabic.

#### Model: Helsinki-NLP/opus-mt-en-ar

This is a **MarianMT** model trained on the OPUS parallel corpus. It's relatively small (~300 MB) and runs on CPU.

#### Lazy Loading

```python
_model = None
_tokenizer = None
_model_available = None  # None = not checked yet

def _load_model():
    global _model, _tokenizer, _model_available

    if _model_available is not None:
        return _model_available  # already loaded (or already failed)

    try:
        from transformers import MarianMTModel, MarianTokenizer
        _tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        _model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        _model_available = True
    except Exception:
        _model_available = False

    return _model_available
```

**Why lazy loading?**

1. The model is large — we don't want to load it if the user disabled translation.
2. If the model isn't installed, we gracefully skip translation instead of crashing.
3. We load it exactly once (singleton pattern using module-level globals).

#### Caching

```python
class Translator:
    def __init__(self):
        self.cache = {}

    def translate_term(self, term):
        key = term.lower().strip()
        if key in self.cache:
            return self.cache[key]
        # ... translate and store in cache
        self.cache[key] = result
```

If the same term appears multiple times (after normalization merging), we don't re-translate it. This is important for performance when processing books with thousands of terms.

---

### 4.8 `embedding_generator.py`

**Purpose:** Generate dense vector representations for each term.

#### Model: all-MiniLM-L6-v2

This is a **SentenceTransformers** model that maps text to 384-dimensional vectors. It's fast and produces high-quality embeddings.

#### Batch Processing

```python
def generate_batch(self, terms):
    # Separate cached from uncached
    to_encode = [t for t in terms if t.lower() not in self.cache]

    if to_encode:
        vectors = _embedding_model.encode(to_encode).tolist()
        for term, vec in zip(to_encode, vectors):
            self.cache[term.lower()] = vec
```

**Why batch?** The model can encode many terms at once using efficient matrix operations. Encoding 100 terms in a batch is much faster than encoding them one by one.

#### Why embeddings?

Embeddings enable:

1. **Semantic search** — find terms similar to a query (e.g., "What terms relate to attention?").
2. **Clustering** — automatically group related terms (e.g., all neural network variants).
3. **Duplicate detection** — two terms with cosine similarity > 0.95 are likely duplicates.

These capabilities are not implemented in this baseline but the embeddings are stored for future use.

---

### 4.9 `terminology_memory.py`

**Purpose:** Define the data model and handle JSON serialization.

#### Data Model

```python
@dataclass
class TerminologyEntry:
    term: str                           # Original term
    normalized_term: str                # Canonical form
    translation_ar: Optional[str]       # Arabic translation (or None)
    definition: Optional[str]           # Extracted definition (or None)
    aliases: List[str]                  # ["NLP", "nlp"]
    frequency: int                      # Total corpus occurrences
    confidence: float                   # TF-IDF score [0, 1]
    source_locations: List[Dict]        # Where this term appears
    embedding: List[float]              # Semantic vector
```

We use Python's `@dataclass` decorator for clean, minimal boilerplate.

#### Merging Logic

```python
def add_entry(self, entry):
    if key in self.entries:
        existing = self.entries[key]
        existing.frequency += entry.frequency        # sum frequencies
        existing.confidence = max(...)                # keep highest confidence
        if existing.definition is None:
            existing.definition = entry.definition   # keep first non-null
```

**Why merge?** The normalizer may produce entries that map to the same canonical form. The memory merges them intelligently: sum frequencies, keep the best confidence, keep the first definition found.

#### Serialization

```python
def save_json(self, filepath, include_embeddings=True):
    data = self.to_dict(include_embeddings=include_embeddings)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

Note `ensure_ascii=False` — this is critical because Arabic characters must be written as Unicode, not as `\uXXXX` escape sequences.

---

### 4.10 `pipeline.py`

**Purpose:** Orchestrate all modules in the correct order.

#### PipelineConfig

```python
class PipelineConfig:
    def __init__(
        self,
        input_path="data/sample_book.json",
        output_path="data/terminology_memory.json",
        min_term_freq=2,
        max_term_tokens=5,
        enable_translation=True,
        enable_embeddings=True,
        include_embeddings_in_output=False,
        embedding_model="all-MiniLM-L6-v2",
    ):
```

All tuneable parameters are centralised in one configuration class. This makes it easy to:
- Change defaults in one place
- Pass configuration from the CLI
- Save/load configurations

#### The `run()` Method

The 9-step pipeline:

```python
def run(self):
    # Step 1: Load book
    loader = BookLoader(self.config.input_path).load()
    segments = loader.get_segments()

    # Step 2: Extract candidates
    extractor = TermExtractor(min_freq=..., max_tokens=...)
    candidates = extractor.extract_from_segments(segments)

    # Step 3: Rank
    ranker = TermRanker()
    confidence_scores = ranker.compute_scores(candidates, segments)

    # Step 4: Normalize
    normalizer = TermNormalizer()
    normalized = normalizer.normalize_candidates(candidates)

    # Step 5: Detect aliases
    alias_detector = AliasDetector()
    alias_detector.detect_from_segments(segments)

    # Step 6: Extract definitions
    def_extractor = DefinitionExtractor()
    def_extractor.extract_from_segments(segments)

    # Step 7: Translate (optional)
    translator = Translator(enabled=self.config.enable_translation)

    # Step 8: Generate embeddings (optional)
    embedder = EmbeddingGenerator(enabled=self.config.enable_embeddings)
    embeddings = embedder.generate_batch(list(normalized.keys()))

    # Step 9: Assemble entries
    for canonical, info in normalized.items():
        entry = TerminologyEntry(
            term=canonical,
            normalized_term=canonical,
            translation_ar=translator.translate_term(canonical),
            definition=def_extractor.get_definition(canonical),
            aliases=alias_detector.get_aliases(canonical),
            frequency=info["occurrences"],
            confidence=...,
            source_locations=info["source_locations"],
            embedding=embeddings.get(canonical, []),
        )
        self.memory.add_entry(entry)

    # Save to disk
    self.memory.save_json(self.config.output_path)
    return self.memory
```

**Why this assembly order?**

The pipeline object creates each module, calls it, and collects results into local variables. In step 9, it combines all results into `TerminologyEntry` objects. This ensures each module runs independently — they don't depend on each other's internal state.

---

### 4.11 `main.py`

**Purpose:** CLI entry point — parses arguments, sets up logging, runs the pipeline, and prints a summary.

#### Argument Parsing

```python
parser.add_argument("--input", default="data/sample_book.json")
parser.add_argument("--no-translate", action="store_true")
parser.add_argument("--no-embed", action="store_true")
```

`action="store_true"` means the flag is `False` by default, and becomes `True` if the user passes `--no-translate`.

#### Summary Output

After the pipeline runs, we print the top 15 terms:

```
============================================================
  TERMINOLOGY MEMORY — SUMMARY
============================================================
  Book:          Foundations of NLP and Machine Learning
  Total terms:   42
  Output file:   data/terminology_memory.json

  Top 15 terms by confidence:
  --------------------------------------------------
    1. machine learning                    conf=1.0000  freq=12   aliases=[ML]   ar=تعلم الآلة
    2. neural network                      conf=0.8723  freq=8    aliases=[—]    ar=شبكة عصبية
    ...
```

---

## 5. Design Decisions Explained

### 5.1 Why spaCy and not NLTK?

| Aspect       | spaCy                          | NLTK                       |
|--------------|--------------------------------|----------------------------|
| Speed        | Very fast (Cython-based)       | Slower (pure Python)       |
| API          | Object-oriented, pipeline-based| Functional, fragmented     |
| NLP features | Tokenization, POS, NER, chunking all in one call | Separate installations |
| Production   | Designed for production use    | Designed for education     |

spaCy is the better choice for a production-oriented system.

### 5.2 Why lazy model loading?

Translation and embedding models are **large** (hundreds of MB). Loading them:
- Takes several seconds.
- Consumes significant RAM.
- May fail if the model isn't downloaded.

By loading lazily (only when first needed), we:
- Avoid loading unused models.
- Fail gracefully with a warning instead of crashing.
- Keep startup time fast.

### 5.3 Why JSON output and not a database?

JSON is:
- **Portable** — can be shared, version-controlled, and diff'd.
- **Human-readable** — reviewers can inspect the output.
- **Easy to integrate** — every programming language can parse JSON.

For larger-scale deployments, the system could be extended to write to SQLite or a vector database, but JSON is the right starting point.

### 5.4 Why regex for definitions?

Alternatives:
- **ML-based definition extraction** — requires training data and a trained model.
- **LLM-based extraction** — powerful but slow and expensive.

Regex is **deterministic, fast, and sufficient** for the common definitional patterns found in academic writing. It's the right choice for a baseline system.

### 5.5 Why both noun phrases AND n-grams?

Noun phrase chunking is **precision-oriented** — it extracts grammatically correct phrases but may miss unconventional terms.

N-gram extraction is **recall-oriented** — it casts a wide net but generates many false positives.

Combining them and then filtering gives us the best of both worlds.

### 5.6 Why normalize AFTER ranking?

Ranking uses the raw surface forms to compute TF-IDF. If we normalized first, we'd lose information about which specific surface form appeared in which segment. After ranking, we normalize and merge — carrying forward the best confidence score from any surface form.

### 5.7 Why a pipeline architecture?

Each module is:
- **Independently testable** — you can unit-test the normalizer without running extraction.
- **Replaceable** — swap the regex definition extractor for an ML-based one without changing anything else.
- **Configurable** — enable/disable translation or embeddings via a flag.

This is a standard software engineering pattern called the **pipeline pattern** or **chain of responsibility**.

---

## 6. Glossary of Technical Terms

| Term                    | Definition                                                                 |
|-------------------------|---------------------------------------------------------------------------|
| **Tokenization**        | Splitting text into individual units (tokens).                            |
| **POS Tagging**         | Assigning grammatical categories (noun, verb, etc.) to tokens.            |
| **Noun Phrase**         | A phrase centered around a noun (e.g., "neural network").                 |
| **N-gram**              | A contiguous sequence of n tokens from text.                              |
| **Lemmatization**       | Reducing words to their dictionary form (e.g., "networks" → "network").   |
| **TF-IDF**              | A score measuring how important a term is in a document collection.       |
| **Embedding**           | A dense vector representation of text that captures semantic meaning.     |
| **Confidence Score**    | A normalized [0, 1] score indicating how likely a candidate is a real term.|
| **Alias**               | An abbreviation or alternate form of a term (e.g., "NLP" for "Natural Language Processing"). |
| **Canonical Form**      | The standardized, normalized version of a term used as the unique key.    |
| **Surface Form**        | The original text as it appears in the document (before normalization).   |
| **Corpus**              | The entire collection of text being analyzed.                             |
| **MarianMT**            | A family of machine translation models from Helsinki NLP.                 |
| **SentenceTransformers**| A Python library for computing sentence/text embeddings.                  |
| **spaCy**               | An industrial-strength NLP library for Python.                            |
| **Lazy Loading**        | Deferring the initialization of an object until it's first needed.        |
| **Singleton Pattern**   | Ensuring only one instance of a resource exists (e.g., the loaded model). |

---

## Final Notes

This system is a **baseline implementation**. It demonstrates the core concepts of:

1. **NLP-based information extraction** — using linguistic analysis to find terms.
2. **Statistical ranking** — using TF-IDF to separate important terms from noise.
3. **Text normalization** — standardizing variant forms into canonical entries.
4. **Modular software design** — clean separation of concerns with a pipeline architecture.

Students are encouraged to:
- Modify the filtering thresholds and observe the effect on output quality.
- Add new definition patterns to improve recall.
- Replace the translation model with a different language pair.
- Experiment with different embedding models and measure similarity between terms.
- Build a simple web UI to browse and validate the extracted terminology.

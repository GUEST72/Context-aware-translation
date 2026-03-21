"""
pipeline.py - Main Processing Pipeline

Orchestrates:
    1. Text Loading
    2. Candidate Term Extraction
    3. Term Ranking
    4. Term Normalization
    5. Alias Detection
    6. Definition Extraction
    7. Translation (optional)
    8. Embedding Generation (optional)
    9. Terminology Memory Assembly & Storage
"""

import logging
import time
import re
from typing import List, Optional

from src.text_loader import BookLoader
from src.term_extractor import TermExtractor
from src.term_ranker import TermRanker
from src.term_normalizer import TermNormalizer
from src.alias_detector import AliasDetector
from src.context_extractor import ContextExtractor
from src.definition_extractor import DefinitionExtractor
from src.translator import Translator
from src.embedding_generator import EmbeddingGenerator
from src.lazy_embedding_cache import LazyEmbeddingCache
from src.terminology_memory import TerminologyMemory, TerminologyEntry

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for the extraction pipeline."""

    def __init__(
        self,
        input_path: str = "data/sample_book.json",
        output_path: str = "data/terminology_memory.json",
        min_term_freq: int = 2,
        max_term_tokens: int = 5,
        max_output_terms: int = 1500,
        pre_normalization_multiplier: int = 8,
        enable_translation: bool = True,
        enable_embeddings: bool = True,
        include_embeddings_in_output: bool = True,
        include_source_locations_in_output: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        lazy_embeddings: bool = False,
        term_importance_weight: float = 0.7,
        context_score_weight: float = 0.3,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.min_term_freq = min_term_freq
        self.max_term_tokens = max_term_tokens
        self.max_output_terms = max(1, int(max_output_terms))
        self.pre_normalization_multiplier = max(1, int(pre_normalization_multiplier))
        self.enable_translation = enable_translation
        # In lazy mode, skip embeddings during pipeline (they're done on-demand later)
        self.enable_embeddings = enable_embeddings and not lazy_embeddings
        self.include_embeddings_in_output = include_embeddings_in_output
        self.include_source_locations_in_output = include_source_locations_in_output
        self.embedding_model = embedding_model
        self.lazy_embeddings = lazy_embeddings
        total_weight = term_importance_weight + context_score_weight
        if total_weight <= 0:
            self.term_importance_weight = 0.7
            self.context_score_weight = 0.3
        else:
            # Normalize weights so they are stable even if user inputs arbitrary values.
            self.term_importance_weight = term_importance_weight / total_weight
            self.context_score_weight = context_score_weight / total_weight


class Pipeline:
    """
    End-to-end terminology memory extraction pipeline.

    Usage:
        config = PipelineConfig(input_path="data/sample_book.json")
        pipeline = Pipeline(config)
        memory = pipeline.run()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.memory = TerminologyMemory()

    _GENERIC_NON_TERMS = {
        "figure",
        "figures",
        "section",
        "sections",
        "chapter",
        "chapters",
        "example",
        "examples",
        "order",
        "problem",
        "in order",
        "for example",
    }
    _REPEATED_TOKEN_PATTERN = re.compile(r"^([a-z]{1,3})(?:\s+\1){2,}$")

    @classmethod
    def _is_noise_term(cls, term: str) -> bool:
        """Return True when term is likely formatting or generic narrative noise."""
        t = term.lower().strip()
        if not t:
            return True
        if t in cls._GENERIC_NON_TERMS:
            return True
        if cls._REPEATED_TOKEN_PATTERN.match(t):
            return True

        tokens = t.split()
        if not tokens:
            return True

        # Drop phrases dominated by very short alphabetic fragments.
        short_alpha = [tok for tok in tokens if tok.isalpha() and len(tok) <= 2]
        if len(short_alpha) >= max(2, len(tokens) - 1):
            return True

        # Exclude leading discourse phrases rather than domain terms.
        leading_noise = {"the", "a", "an", "in", "for", "to", "of"}
        if len(tokens) >= 2 and tokens[0] in leading_noise:
            return True

        return False

    def run(self) -> TerminologyMemory:
        """Execute the full pipeline and return the populated TerminologyMemory."""
        start = time.time()
        logger.info("=" * 60)
        logger.info("TERMINOLOGY MEMORY EXTRACTION PIPELINE")
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        # Step 1 – Load book
        # ------------------------------------------------------------------
        logger.info("[1/9] Loading book …")
        loader = BookLoader(self.config.input_path).load()
        segments = loader.get_segments()
        self.memory.book_title = loader.get_book_title()
        book_label = self.memory.book_title or "(no title provided)"
        logger.info(f"      Book: '{book_label}' — {len(segments)} segments")

        # ------------------------------------------------------------------
        # Step 2 – Extract candidate terms
        # ------------------------------------------------------------------
        logger.info("[2/9] Extracting candidate terms …")
        extractor = TermExtractor(
            min_freq=self.config.min_term_freq,
            max_tokens=self.config.max_term_tokens,
        )
        candidates = extractor.extract_from_segments(segments)
        logger.info(f"      {len(candidates)} candidates found")

        # ------------------------------------------------------------------
        # Step 3 – Rank terms
        # ------------------------------------------------------------------
        logger.info("[3/9] Ranking terms …")
        ranker = TermRanker()
        confidence_scores = ranker.compute_scores(candidates, segments)

        # Reduce noise and runtime by keeping only top-ranked candidate surface forms.
        max_pre_normalization = self.config.max_output_terms * self.config.pre_normalization_multiplier
        if len(candidates) > max_pre_normalization:
            ranked_keys = sorted(
                candidates.keys(),
                key=lambda key: (
                    confidence_scores.get(key, 0.0),
                    candidates[key].occurrences,
                ),
                reverse=True,
            )
            keep_keys = set(ranked_keys[:max_pre_normalization])
            candidates = {
                key: cand for key, cand in candidates.items() if key in keep_keys
            }
            logger.info(
                "      Pre-normalization limit: %s -> top %s candidates",
                len(ranked_keys),
                len(candidates),
            )

        # ------------------------------------------------------------------
        # Step 4 – Normalize terms
        # ------------------------------------------------------------------
        logger.info("[4/9] Normalizing terms …")
        normalizer = TermNormalizer()
        normalized = normalizer.normalize_candidates(candidates)

        # Keep only top normalized terms by best candidate confidence to reduce noise.
        if len(normalized) > self.config.max_output_terms:
            logger.info(
                "      Limiting normalized terms: %s -> top %s by rank",
                len(normalized),
                self.config.max_output_terms,
            )

            def _best_confidence(item):
                _, info = item
                best = 0.0
                for sf in info.get("surface_forms", []):
                    if sf in confidence_scores:
                        best = max(best, confidence_scores[sf])
                return best

            ranked_normalized = sorted(
                normalized.items(),
                key=_best_confidence,
                reverse=True,
            )
            selected = []
            for item in ranked_normalized:
                canonical, _ = item
                if self._is_noise_term(canonical):
                    continue
                selected.append(item)
                if len(selected) >= self.config.max_output_terms:
                    break

            # Fallback: keep best available terms even if many are filtered.
            if len(selected) < self.config.max_output_terms:
                for item in ranked_normalized:
                    if item in selected:
                        continue
                    selected.append(item)
                    if len(selected) >= self.config.max_output_terms:
                        break

            normalized = dict(selected)

        # ------------------------------------------------------------------
        # Step 5 – Detect aliases
        # ------------------------------------------------------------------
        logger.info("[5/9] Detecting aliases …")
        alias_detector = AliasDetector()
        alias_detector.detect_from_segments(segments)

        # ------------------------------------------------------------------
        # Step 6 – Extract definitions
        # ------------------------------------------------------------------
        logger.info("[6/9] Extracting definitions …")
        def_extractor = DefinitionExtractor()
        def_extractor.extract_from_segments(segments)

        # ------------------------------------------------------------------
        # Step 7 – Translate terms (optional)
        # ------------------------------------------------------------------
        translator = Translator(enabled=self.config.enable_translation)
        if self.config.enable_translation:
            logger.info("[7/9] Translating terms …")
        else:
            logger.info("[7/9] Translation disabled — skipping.")

        # ------------------------------------------------------------------
        # Step 8 – Generate embeddings (optional)
        # ------------------------------------------------------------------
        embedder = EmbeddingGenerator(
            model_name=self.config.embedding_model,
            enabled=self.config.enable_embeddings,
        )
        if self.config.enable_embeddings:
            logger.info("[8/9] Generating embeddings …")
        else:
            logger.info("[8/9] Embeddings disabled — skipping.")

        # Batch-generate embeddings if enabled
        all_terms = list(normalized.keys())
        embeddings = embedder.generate_batch(all_terms) if self.config.enable_embeddings else {}

        # ------------------------------------------------------------------
        # Step 9 – Assemble Terminology Memory
        # ------------------------------------------------------------------
        logger.info("[9/9] Assembling terminology memory …")
        context_extractor = ContextExtractor()

        for canonical, info in normalized.items():
            # Pick the best confidence from the original surface forms
            best_confidence = 0.0
            for sf in info["surface_forms"]:
                if sf in confidence_scores:
                    best_confidence = max(best_confidence, confidence_scores[sf])

            # Retrieve ancillary data
            definition_candidates = def_extractor.get_definitions(
                canonical,
                surface_forms=info.get("surface_forms", []),
                source_locations=info.get("source_locations", []),
                segments=segments,
                max_definitions=3,
            )
            definition = definition_candidates[0] if definition_candidates else None
            aliases = alias_detector.get_aliases(canonical)
            translation = translator.translate_term(canonical)
            embedding = embeddings.get(canonical, [])
            example_data = context_extractor.extract_examples(
                canonical_term=canonical,
                surface_forms=info.get("surface_forms", []),
                source_locations=info.get("source_locations", []),
                segments=segments,
                max_examples=2,
            )
            example_sentences = example_data.get("example_sentences", [])
            primary_example_sentence = example_data.get("primary_example_sentence")
            supporting_example_sentences = example_data.get(
                "supporting_example_sentences", []
            )
            example_score_breakdown = example_data.get("example_score_breakdown", {})
            context_final_score = float(example_score_breakdown.get("final_score", 0.0))
            hybrid_rank_score = (
                self.config.term_importance_weight * best_confidence
                + self.config.context_score_weight * context_final_score
            )
            example_score_breakdown["term_importance_score"] = round(best_confidence, 4)
            example_score_breakdown["context_final_score"] = round(context_final_score, 4)
            example_score_breakdown["term_rank_score"] = round(hybrid_rank_score, 4)
            example_score_breakdown["term_importance_weight"] = round(self.config.term_importance_weight, 4)
            example_score_breakdown["context_score_weight"] = round(self.config.context_score_weight, 4)
            surface_form_variants = info.get("surface_form_variants", [])

            entry = TerminologyEntry(
                term=canonical,
                normalized_term=canonical,
                translation_ar=translation,
                definition=definition,
                definition_candidates=definition_candidates,
                aliases=aliases,
                frequency=info["occurrences"],
                confidence=round(hybrid_rank_score, 4),
                source_locations=info["source_locations"],
                embedding=embedding,
                example_sentences=example_sentences,
                primary_example_sentence=primary_example_sentence,
                supporting_example_sentences=supporting_example_sentences,
                example_score_breakdown=example_score_breakdown,
                surface_form_variants=surface_form_variants,
            )
            self.memory.add_entry(entry)

        # ------------------------------------------------------------------
        # Save output
        # ------------------------------------------------------------------
        self.memory.save_json(
            self.config.output_path,
            include_embeddings=self.config.include_embeddings_in_output,
            include_source_locations=self.config.include_source_locations_in_output,
        )

        elapsed = time.time() - start
        logger.info(f"Pipeline completed in {elapsed:.2f}s — "
                     f"{len(self.memory)} terms stored.")
        logger.info("=" * 60)
        return self.memory

    def embed_terms_lazy(self, terms: Optional[List[str]] = None, cache_path: str = "data/embedding_cache.json") -> LazyEmbeddingCache:
        """
        Embed terminology terms on-demand using lazy cache.

        This method is called AFTER pipeline completes. Embeddings are stored in a separate
        cache file, not in the main terminology_memory.json, keeping extraction fast.

        Args:
            terms: List of terms to embed. If None, embeds all terms in memory.
            cache_path: Path to store embedding cache.

        Returns:
            LazyEmbeddingCache instance with populated embeddings.
        """
        if not self.memory:
            logger.warning("No terminology memory loaded; cannot embed. Run pipeline first.")
            return LazyEmbeddingCache(cache_path, self.config.embedding_model)

        cache = LazyEmbeddingCache(cache_path, self.config.embedding_model)

        if terms is None:
            terms = list(self.memory.entries.keys())

        logger.info(f"[LAZY] Embedding {len(terms)} terms on-demand …")
        
        # Use batch embedding for efficiency
        embeddings = cache.batch_embed(terms)
        
        cached_count = sum(1 for emb in embeddings.values() if emb is not None)
        logger.info(f"[LAZY] Computed/cached {cached_count} embeddings.")
        
        # Persist cache to disk
        cache.persist()
        logger.info(f"[LAZY] Embedding cache saved to {cache_path} ({len(embeddings)} entries).")
        
        return cache

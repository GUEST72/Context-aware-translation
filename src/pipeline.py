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
from typing import Optional

from src.text_loader import BookLoader
from src.term_extractor import TermExtractor
from src.term_ranker import TermRanker
from src.term_normalizer import TermNormalizer
from src.alias_detector import AliasDetector
from src.definition_extractor import DefinitionExtractor
from src.translator import Translator
from src.embedding_generator import EmbeddingGenerator
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
        enable_translation: bool = True,
        enable_embeddings: bool = True,
        include_embeddings_in_output: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.min_term_freq = min_term_freq
        self.max_term_tokens = max_term_tokens
        self.enable_translation = enable_translation
        self.enable_embeddings = enable_embeddings
        self.include_embeddings_in_output = include_embeddings_in_output
        self.embedding_model = embedding_model


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
        logger.info(f"      Book: '{self.memory.book_title}' — {len(segments)} segments")

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

        # ------------------------------------------------------------------
        # Step 4 – Normalize terms
        # ------------------------------------------------------------------
        logger.info("[4/9] Normalizing terms …")
        normalizer = TermNormalizer()
        normalized = normalizer.normalize_candidates(candidates)

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

        for canonical, info in normalized.items():
            # Pick the best confidence from the original surface forms
            best_confidence = 0.0
            for sf in info["surface_forms"]:
                if sf in confidence_scores:
                    best_confidence = max(best_confidence, confidence_scores[sf])

            # Retrieve ancillary data
            definition = def_extractor.get_definition(canonical)
            aliases = alias_detector.get_aliases(canonical)
            translation = translator.translate_term(canonical)
            embedding = embeddings.get(canonical, [])

            entry = TerminologyEntry(
                term=canonical,
                normalized_term=canonical,
                translation_ar=translation,
                definition=definition,
                aliases=aliases,
                frequency=info["occurrences"],
                confidence=round(best_confidence, 4),
                source_locations=info["source_locations"],
                embedding=embedding,
            )
            self.memory.add_entry(entry)

        # ------------------------------------------------------------------
        # Save output
        # ------------------------------------------------------------------
        self.memory.save_json(
            self.config.output_path,
            include_embeddings=self.config.include_embeddings_in_output,
        )

        elapsed = time.time() - start
        logger.info(f"Pipeline completed in {elapsed:.2f}s — "
                     f"{len(self.memory)} terms stored.")
        logger.info("=" * 60)
        return self.memory

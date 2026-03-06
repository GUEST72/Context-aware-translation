#!/usr/bin/env python3
"""
main.py - Entry Point for the Terminology Memory Extraction System

Run:
    python main.py                        # defaults
    python main.py --input data/book.json # custom input
    python main.py --no-translate         # skip translation
    python main.py --no-embed             # skip embeddings
"""

import argparse
import logging
import sys
import os

# Ensure project root is on the path so `src.*` imports work when
# running the script directly (python main.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import Pipeline, PipelineConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminology Memory Extraction System for Academic Books"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_book.json",
        help="Path to the structured book JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/terminology_memory.json",
        help="Path for the output terminology memory JSON.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum term frequency threshold (default: 2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5,
        help="Maximum tokens per term phrase (default: 5).",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Disable Arabic translation.",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Disable embedding generation.",
    )
    parser.add_argument(
        "--no-include-embeddings",
        action="store_true",
        help="Exclude embedding vectors from the output JSON (they are included by default).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model name for embeddings.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        min_term_freq=args.min_freq,
        max_term_tokens=args.max_tokens,
        enable_translation=not args.no_translate,
        enable_embeddings=not args.no_embed,
        include_embeddings_in_output=not args.no_include_embeddings,
        embedding_model=args.embedding_model,
    )

    pipeline = Pipeline(config)
    memory = pipeline.run()

    # Print summary
    print("\n" + "=" * 60)
    print("  TERMINOLOGY MEMORY — SUMMARY")
    print("=" * 60)
    print(f"  Book:          {memory.book_title}")
    print(f"  Total terms:   {len(memory)}")
    print(f"  Output file:   {config.output_path}")
    print()
    print("  Top 15 terms by confidence:")
    print("  " + "-" * 50)
    sorted_entries = sorted(
        memory.entries.values(), key=lambda e: e.confidence, reverse=True
    )
    for i, entry in enumerate(sorted_entries[:15], 1):
        aliases_str = ", ".join(entry.aliases) if entry.aliases else "—"
        trans_str = entry.translation_ar or "—"
        print(
            f"  {i:>3}. {entry.normalized_term:<35} "
            f"conf={entry.confidence:.4f}  freq={entry.frequency:<4} "
            f"aliases=[{aliases_str}]  ar={trans_str}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

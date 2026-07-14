"""Shared helpers for the async-vs-threadpool-vs-celery upload_pdf benchmark.

Each benchmark variant app writes its uploaded file and its parsed JSON
output to a UUID-named path so concurrent requests never collide, and never
touches the shared BOOK_PATH/BOOK_DATA global used by the real app.
"""
import uuid
from pathlib import Path

import pymupdf
from parser.exporter import export_to_json

BASE_DIR = Path(__file__).resolve().parent
BENCH_UPLOAD_DIR = BASE_DIR / "bench_uploads"
BENCH_OUTPUT_DIR = BASE_DIR / "bench_outputs"
BENCH_UPLOAD_DIR.mkdir(exist_ok=True)
BENCH_OUTPUT_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def unique_paths(original_filename: str) -> tuple[Path, Path]:
    suffix = Path(original_filename).suffix or ".pdf"
    token = uuid.uuid4().hex
    upload_path = BENCH_UPLOAD_DIR / f"{token}{suffix}"
    output_path = BENCH_OUTPUT_DIR / f"{token}.json"
    return upload_path, output_path


def parse_pdf(upload_path: Path, output_path: Path) -> dict:
    doc = pymupdf.open(str(upload_path))
    try:
        return export_to_json(doc, str(output_path))
    finally:
        doc.close()

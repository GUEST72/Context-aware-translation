#!/usr/bin/env python3
"""Profiles the GIL-release split in the parsing hot path.

Two questions:
  1. How is wall time split between pymupdf.open() (C extension, may release
     the GIL) and export_to_json() (pure Python, holds the GIL)?
  2. Does that split predict how variant B (threadpool) scales with
     concurrent threads? If pymupdf release the GIL for a real share of the
     work, running N parses across N threads should be faster than N
     sequential parses by roughly that share; if export_to_json dominates
     and holds the GIL, threaded wall time should be ~= sequential wall time.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import pymupdf  # noqa: E402
from parser.exporter import export_to_json  # noqa: E402
from parser.parser import get_spans_from_page, group_spans_into_lines, group_lines_into_paragraphs  # noqa: E402
from parser.classifier import classify_paragraphs  # noqa: E402

PDF_PATH = Path(__file__).resolve().parent / "fixtures" / "bench_book.pdf"
N = 8


def timed_parse(tag: str) -> dict:
    t0 = time.perf_counter()
    doc = pymupdf.open(str(PDF_PATH))
    t1 = time.perf_counter()
    export_to_json(doc, f"/tmp/gil_profile_{tag}.json")
    t2 = time.perf_counter()
    doc.close()
    return {"open_s": t1 - t0, "export_s": t2 - t1, "total_s": t2 - t0}


def timed_parse_instrumented() -> dict:
    """Same work as export_to_json, but timing the MuPDF-touching phase
    (get_spans_from_page, which calls the C-level page.get_text('dict') and
    then builds Span objects) separately from the pure-Python grouping /
    classification that follows it."""
    t_open_start = time.perf_counter()
    doc = pymupdf.open(str(PDF_PATH))
    open_s = time.perf_counter() - t_open_start

    mupdf_s = 0.0
    python_s = 0.0
    for page in doc:
        t0 = time.perf_counter()
        spans = get_spans_from_page(page)
        t1 = time.perf_counter()
        mupdf_s += t1 - t0

        lines = group_spans_into_lines(spans)
        paragraphs = group_lines_into_paragraphs(lines)
        classify_paragraphs(paragraphs)
        python_s += time.perf_counter() - t1
    doc.close()
    return {"open_s": open_s, "mupdf_touching_s": mupdf_s, "python_grouping_s": python_s}


def main() -> None:
    print("--- single-parse split (average of 5) ---")
    splits = [timed_parse(f"warm{i}") for i in range(5)]
    avg_open = sum(s["open_s"] for s in splits) / len(splits)
    avg_export = sum(s["export_s"] for s in splits) / len(splits)
    avg_total = avg_open + avg_export
    print(f"pymupdf.open:    {avg_open*1000:7.1f} ms  ({avg_open/avg_total*100:5.1f}% of total)")
    print(f"export_to_json:  {avg_export*1000:7.1f} ms  ({avg_export/avg_total*100:5.1f}% of total)")
    print(f"total:           {avg_total*1000:7.1f} ms")

    print("\n--- fine-grained split inside export_to_json (average of 5) ---")
    fine = [timed_parse_instrumented() for _ in range(5)]
    avg_open2 = sum(s["open_s"] for s in fine) / len(fine)
    avg_c = sum(s["mupdf_touching_s"] for s in fine) / len(fine)
    avg_py = sum(s["python_grouping_s"] for s in fine) / len(fine)
    avg_total2 = avg_open2 + avg_c + avg_py
    print(f"doc.open:                          {avg_open2*1000:7.1f} ms  ({avg_open2/avg_total2*100:5.1f}%)")
    print(f"get_spans_from_page (MuPDF C call): {avg_c*1000:7.1f} ms  ({avg_c/avg_total2*100:5.1f}%)")
    print(f"pure-Python grouping/classify:      {avg_py*1000:7.1f} ms  ({avg_py/avg_total2*100:5.1f}%)")
    print(f"total:                              {avg_total2*1000:7.1f} ms")

    print(f"\n--- sequential vs threaded scaling, N={N} parses ---")
    t0 = time.perf_counter()
    for i in range(N):
        timed_parse(f"seq{i}")
    seq_wall = time.perf_counter() - t0
    print(f"sequential wall time for {N} parses: {seq_wall:.2f}s ({seq_wall/N*1000:.1f}ms/parse)")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N) as pool:
        list(pool.map(lambda i: timed_parse(f"thr{i}"), range(N)))
    thr_wall = time.perf_counter() - t0
    print(f"threaded ({N} workers) wall time for {N} parses: {thr_wall:.2f}s ({thr_wall/N*1000:.1f}ms/parse)")

    speedup = seq_wall / thr_wall if thr_wall > 0 else 0
    print(f"\nspeedup from threading: {speedup:.2f}x  (1.0x = fully GIL-bound, {N}.0x = fully parallel)")
    implied_gil_released_share = max(0.0, 1 - 1 / speedup) if speedup > 0 else 0.0
    print(f"implied GIL-released share of total work: ~{implied_gil_released_share*100:.0f}%")


if __name__ == "__main__":
    main()

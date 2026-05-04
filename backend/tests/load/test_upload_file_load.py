#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def _default_pdf_path() -> Path | None:
    backend_dir = Path(__file__).resolve().parents[2]
    uploads_dir = backend_dir / "app" / "uploads"
    pdf_files = sorted(uploads_dir.glob("*.pdf"))
    return pdf_files[0] if pdf_files else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = math.ceil((percentile / 100) * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def _summarize_body(body: Any) -> str:
    if isinstance(body, dict):
        if "detail" in body:
            return f"detail={body['detail']}"
        if "message" in body:
            pages = body.get("pages_parsed")
            if pages is not None:
                return f"message={body['message']}, pages_parsed={pages}"
            return f"message={body['message']}"
        return json.dumps(body, ensure_ascii=True)[:180]
    return str(body)[:180]


def _send_request(index: int, url: str, filename: str, file_bytes: bytes, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    upload_name = f"loadtest_{index:04d}_{filename}"
    files = {"file": (upload_name, file_bytes, "application/pdf")}
    try:
        response = requests.post(url, files=files, timeout=timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            body = response.json()
        except ValueError:
            body = response.text[:200]
        return {
            "request_id": index,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "ok": 200 <= response.status_code < 300,
            "body": body,
        }
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "request_id": index,
            "status_code": "ERROR",
            "elapsed_ms": elapsed_ms,
            "ok": False,
            "body": str(exc),
        }


def _parse_args() -> argparse.Namespace:
    default_pdf = _default_pdf_path()
    parser = argparse.ArgumentParser(
        description="Simple load test for upload_file/upload_pdf endpoint with per-response output."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/upload_pdf",
        help="Upload endpoint URL. Example: http://127.0.0.1:8000/upload_file",
    )
    parser.add_argument(
        "--file",
        default=str(default_pdf) if default_pdf else None,
        help="Path to a PDF file to upload.",
    )
    parser.add_argument("--requests", type=int, default=20, help="Total number of upload requests.")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel workers.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Timeout per request in seconds.")
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report output path (example: backend/tests/load/report.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.file:
        print("No PDF file found. Pass one with --file /path/to/file.pdf")
        return 2

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"PDF file not found: {file_path}")
        return 2

    if args.requests < 1:
        print("--requests must be >= 1")
        return 2

    if args.concurrency < 1:
        print("--concurrency must be >= 1")
        return 2

    concurrency = min(args.concurrency, args.requests)
    file_bytes = file_path.read_bytes()

    print(f"Target endpoint: {args.url}")
    print(f"PDF file: {file_path}")
    print(f"Total requests: {args.requests}")
    print(f"Concurrency: {concurrency}")
    print("-" * 72)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_send_request, i + 1, args.url, file_path.name, file_bytes, args.timeout)
            for i in range(args.requests)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"request={result['request_id']:03d} "
                f"status={result['status_code']} "
                f"time={result['elapsed_ms']:.1f}ms "
                f"{_summarize_body(result['body'])}"
            )

    results.sort(key=lambda item: item["request_id"])

    latencies = [item["elapsed_ms"] for item in results]
    success_count = sum(1 for item in results if item["ok"])
    failure_count = len(results) - success_count
    status_counts: dict[str, int] = {}
    for item in results:
        key = str(item["status_code"])
        status_counts[key] = status_counts.get(key, 0) + 1

    summary = {
        "total_requests": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "status_counts": status_counts,
        "latency_ms": {
            "min": min(latencies) if latencies else 0.0,
            "avg": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else 0.0,
        },
    }

    print("-" * 72)
    print("Summary")
    print(f"  success: {summary['success_count']}/{summary['total_requests']}")
    print(f"  failures: {summary['failure_count']}")
    print(f"  status counts: {summary['status_counts']}")
    print(
        "  latency(ms): "
        f"min={summary['latency_ms']['min']:.1f}, "
        f"avg={summary['latency_ms']['avg']:.1f}, "
        f"p95={summary['latency_ms']['p95']:.1f}, "
        f"max={summary['latency_ms']['max']:.1f}"
    )

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print(f"  report saved to: {report_path}")

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

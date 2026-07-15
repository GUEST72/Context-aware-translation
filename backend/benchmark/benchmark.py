#!/usr/bin/env python3
"""upload_pdf benchmark: async inline vs. threadpool vs. Celery -- all in one file.

Subcommands (`python benchmark.py <command> --help` for details on each):

  serve           run one variant's FastAPI app under uvicorn (app_a/app_b/app_c below)
  celery-worker   run the Celery worker used by variant c, in-process
  drive           one load-test trial against an already-running server (fires N
                  concurrent requests, probes /health, samples CPU/RAM/threads)
  run             orchestrate the full variant x load x trial matrix: starts/stops
                  each server (+ Celery worker for c) fresh, runs 1 discarded
                  warm-up + TRIALS_PER_LEVEL trials per (variant, load) cell
  gil-profile     standalone micro-benchmark: splits parse time between the MuPDF
                  C call and pure-Python grouping, and measures the actual
                  GIL-released share via sequential-vs-threaded scaling
  aggregate       reads runs/**/trial_*.json into a mean+-stddev summary table
                  (runs/summary.json) and writes runs/plots/*.png

See ANALYSIS.md for the full write-up and README.md for how to reproduce.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import statistics as stats
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import httpx
import psutil
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from celery.result import AsyncResult

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths & shared config
# --------------------------------------------------------------------------- #

THIS_FILE = Path(__file__).resolve()
BENCHMARK_DIR = THIS_FILE.parent
BACKEND_DIR = BENCHMARK_DIR.parent
APP_DIR = BACKEND_DIR / "app"
VENV_PY = BACKEND_DIR / "venv" / "bin" / "python"
PDF_PATH = BENCHMARK_DIR / "fixtures" / "bench_book.pdf"
RUNS_DIR = BENCHMARK_DIR / "runs"

# The real app's modules (parser, celery_config, tasks) use bare imports that
# assume backend/app is on sys.path -- same as when main.py itself is run.
sys.path.insert(0, str(APP_DIR))

import pymupdf  # noqa: E402
from parser.exporter import export_to_json  # noqa: E402
from parser.parser import get_spans_from_page, group_lines_into_paragraphs, group_spans_into_lines  # noqa: E402
from parser.classifier import classify_paragraphs  # noqa: E402
from celery_config import celery_app  # noqa: E402
from tasks import parse_pdf_task  # noqa: E402

VARIANT_IDS = ["a", "b", "c"]
VARIANT_PORTS = {"a": 8001, "b": 8002, "c": 8003}
VARIANT_NEEDS_CELERY = {"a": False, "b": False, "c": True}
VARIANT_LABELS = {"a": "A: async inline", "b": "B: threadpool", "c": "C: celery"}
VARIANT_COLORS = {"a": "#d64545", "b": "#3b82c4", "c": "#3fa34d"}

LOAD_LEVELS = [10, 100, 1000]
TRIALS_PER_LEVEL = 3
WARMUP_RUNS = 1
REQUEST_TIMEOUT_S = 180.0
CELERY_CONCURRENCY = os.cpu_count() or 4
REDIS_ENV = {
    "CELERY_BROKER_URL": "redis://127.0.0.1:6379/0",
    "CELERY_RESULT_BACKEND": "redis://127.0.0.1:6379/1",
}

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
BENCH_UPLOAD_DIR = APP_DIR / "bench_uploads"
BENCH_OUTPUT_DIR = APP_DIR / "bench_outputs"
BENCH_UPLOAD_DIR.mkdir(exist_ok=True)
BENCH_OUTPUT_DIR.mkdir(exist_ok=True)


def unique_paths(original_filename: str) -> tuple[Path, Path]:
    """Every request gets its own upload + output path so concurrent requests
    never collide, and the real app's shared BOOK_PATH/BOOK_DATA global is
    never touched."""
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


# --------------------------------------------------------------------------- #
# Variant A: async def upload_pdf, parsing runs inline on the event-loop thread.
#
# Reproduces the current production shape of a CPU-bound handler declared
# `async def` with no offloading: nothing here yields control back to the
# event loop during the pymupdf.open()/export_to_json() call, so a single
# in-flight parse blocks every other coroutine (including /health) on this
# worker.
# --------------------------------------------------------------------------- #

app_a = FastAPI(title="Benchmark Variant A - async inline")


@app_a.get("/health")
async def health_a():
    return {"status": "ok"}


@app_a.post("/upload_pdf")
async def upload_pdf_a(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")

    upload_path, output_path = unique_paths(file.filename)
    with open(upload_path, "wb") as f:
        f.write(content)

    parsed = parse_pdf(upload_path, output_path)

    return {
        "message": "PDF parsed",
        "pages_parsed": len(parsed.get("pages", [])),
        "output_json": str(output_path),
    }


# --------------------------------------------------------------------------- #
# Variant B: plain `def` upload_pdf, offloaded by Starlette to its AnyIO
# threadpool.
#
# Identical logic to variant A, but declaring the path operation as a
# synchronous function makes FastAPI run it via `anyio.to_thread.run_sync`
# instead of on the event-loop thread, so the event loop (and /health) stays
# responsive while parsing runs on a worker thread. `await file.read()` is
# replaced with the sync equivalent (`file.file.read()`) since this function
# can't await.
# --------------------------------------------------------------------------- #

app_b = FastAPI(title="Benchmark Variant B - threadpool")


@app_b.get("/health")
async def health_b():
    return {"status": "ok"}


@app_b.post("/upload_pdf")
def upload_pdf_b(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = file.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")

    upload_path, output_path = unique_paths(file.filename)
    with open(upload_path, "wb") as f:
        f.write(content)

    parsed = parse_pdf(upload_path, output_path)

    return {
        "message": "PDF parsed",
        "pages_parsed": len(parsed.get("pages", [])),
        "output_json": str(output_path),
    }


# --------------------------------------------------------------------------- #
# Variant C: enqueue to Celery, parse in a separate worker process (own
# interpreter, own GIL).
#
# The endpoint itself only saves the upload and enqueues a task -- true
# end-to-end latency (what `drive`/cmd_drive measures) is enqueue time plus
# polling /upload_pdf/status/{task_id} until the worker pool finishes parsing.
# --------------------------------------------------------------------------- #

app_c = FastAPI(title="Benchmark Variant C - celery")


@app_c.get("/health")
async def health_c():
    return {"status": "ok"}


@app_c.post("/upload_pdf", status_code=202)
async def upload_pdf_c(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")

    upload_path, output_path = unique_paths(file.filename)
    with open(upload_path, "wb") as f:
        f.write(content)

    task = parse_pdf_task.delay(str(upload_path), str(output_path))

    return {
        "message": "PDF uploaded. Parsing has been queued.",
        "task_id": task.id,
        "task_status": "PENDING",
    }


@app_c.get("/upload_pdf/status/{task_id}")
def upload_pdf_status_c(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "state": task.state}
    if task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.result)
    return response


VARIANT_APPS = {"a": app_a, "b": app_b, "c": app_c}


def cmd_serve(args: argparse.Namespace) -> None:
    app = VARIANT_APPS[args.variant]
    port = args.port or VARIANT_PORTS[args.variant]
    uvicorn.run(app, host=args.host, port=port)


def cmd_celery_worker(args: argparse.Namespace) -> None:
    celery_app.worker_main(argv=[
        "worker", f"--loglevel={args.loglevel}", f"--concurrency={args.concurrency}", "-P", "prefork",
    ])


# --------------------------------------------------------------------------- #
# `drive`: one load-test trial. Fires `--concurrency` concurrent POST
# /upload_pdf requests against a single running server variant, measures
# end-to-end latency per request (for the celery variant this includes
# polling /upload_pdf/status/{task_id} until the worker finishes -- not just
# enqueue time), probes GET /health every ~100ms throughout the run to
# quantify event-loop responsiveness, and samples CPU/RAM/thread-or-process
# count of the given PIDs (and their descendants) throughout the run.
#
# One process running `drive` = one trial. Warm-up vs. real trials, and
# restarting the server between variants, is handled by `run`/cmd_run.
# --------------------------------------------------------------------------- #

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(0, min(math.ceil((p / 100) * len(s)) - 1, len(s) - 1))
    return s[rank]


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


async def _one_request_a_b(client: httpx.AsyncClient, base_url: str, file_bytes: bytes,
                            filename: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/upload_pdf",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
        return {"ok": resp.status_code == 200, "status_code": resp.status_code,
                "elapsed_s": elapsed, "timed_out": False}
    except httpx.TimeoutException:
        return {"ok": False, "status_code": "TIMEOUT", "elapsed_s": time.perf_counter() - started,
                "timed_out": True}
    except httpx.HTTPError as exc:
        return {"ok": False, "status_code": f"ERROR:{exc.__class__.__name__}",
                "elapsed_s": time.perf_counter() - started, "timed_out": False}


async def _one_request_c(client: httpx.AsyncClient, base_url: str, file_bytes: bytes,
                          filename: str, timeout: float, poll_interval: float) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + timeout
    try:
        resp = await client.post(
            f"{base_url}/upload_pdf",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=timeout,
        )
        if resp.status_code != 202:
            return {"ok": False, "status_code": resp.status_code,
                    "elapsed_s": time.perf_counter() - started, "timed_out": False}
        task_id = resp.json()["task_id"]

        while True:
            now = time.perf_counter()
            if now > deadline:
                return {"ok": False, "status_code": "TIMEOUT",
                        "elapsed_s": now - started, "timed_out": True}
            status_resp = await client.get(f"{base_url}/upload_pdf/status/{task_id}",
                                            timeout=timeout)
            state = status_resp.json().get("state")
            if state == "SUCCESS":
                return {"ok": True, "status_code": 200,
                        "elapsed_s": time.perf_counter() - started, "timed_out": False}
            if state == "FAILURE":
                return {"ok": False, "status_code": "TASK_FAILURE",
                        "elapsed_s": time.perf_counter() - started, "timed_out": False}
            await asyncio.sleep(poll_interval)
    except httpx.TimeoutException:
        return {"ok": False, "status_code": "TIMEOUT", "elapsed_s": time.perf_counter() - started,
                "timed_out": True}
    except httpx.HTTPError as exc:
        return {"ok": False, "status_code": f"ERROR:{exc.__class__.__name__}",
                "elapsed_s": time.perf_counter() - started, "timed_out": False}


async def _health_prober(base_url: str, interval_s: float, stop_event: asyncio.Event,
                          samples: list[dict[str, Any]]) -> None:
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                resp = await client.get(f"{base_url}/health", timeout=10.0)
                elapsed = time.perf_counter() - t0
                samples.append({"t": t0, "elapsed_s": elapsed, "ok": resp.status_code == 200})
            except httpx.HTTPError:
                samples.append({"t": t0, "elapsed_s": time.perf_counter() - t0, "ok": False})
            await asyncio.sleep(interval_s)


def _current_pids(root_pids: list[int]) -> set[int]:
    pids = set()
    for pid in root_pids:
        try:
            p = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue
        pids.add(pid)
        try:
            pids.update(c.pid for c in p.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
    return pids


async def _resource_monitor(root_pids: list[int], interval_s: float, stop_event: asyncio.Event,
                             samples: list[dict[str, Any]]) -> None:
    # psutil.Process.cpu_percent() reports usage *since the previous call on
    # that same instance* -- creating a fresh Process each tick always looks
    # like a first call and reports 0. Keep one Process object per pid alive
    # across ticks so the delta is meaningful, and reconcile as children
    # come and go (celery prefork pool).
    tracked: dict[int, psutil.Process] = {}
    for pid in _current_pids(root_pids):
        try:
            p = psutil.Process(pid)
            p.cpu_percent(interval=None)
            tracked[pid] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    while not stop_event.is_set():
        for pid in _current_pids(root_pids) - tracked.keys():
            try:
                p = psutil.Process(pid)
                p.cpu_percent(interval=None)
                tracked[pid] = p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        cpu_total = 0.0
        rss_total = 0
        threads_total = 0
        alive = 0
        dead_pids = []
        for pid, p in tracked.items():
            try:
                cpu_total += p.cpu_percent(interval=None)
                rss_total += p.memory_info().rss
                threads_total += p.num_threads()
                alive += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                dead_pids.append(pid)
        for pid in dead_pids:
            tracked.pop(pid, None)

        samples.append({
            "t": time.time(),
            "cpu_percent_sum": cpu_total,
            "rss_mb": rss_total / (1024 * 1024),
            "thread_count": threads_total,
            "process_count": alive,
        })
        await asyncio.sleep(interval_s)


async def run_driver(args: argparse.Namespace) -> dict[str, Any]:
    file_bytes = Path(args.pdf).read_bytes()
    filename = Path(args.pdf).name

    stop_event = asyncio.Event()
    health_samples: list[dict[str, Any]] = []
    resource_samples: list[dict[str, Any]] = []

    monitor_tasks = []
    if args.health_url:
        monitor_tasks.append(asyncio.create_task(
            _health_prober(args.health_url, args.health_interval_ms / 1000, stop_event, health_samples)))
    root_pids = [int(x) for x in args.monitor_pids.split(",")] if args.monitor_pids else []
    if root_pids:
        monitor_tasks.append(asyncio.create_task(
            _resource_monitor(root_pids, args.resource_interval_ms / 1000, stop_event, resource_samples)))

    limits = httpx.Limits(max_connections=args.concurrency + 10,
                           max_keepalive_connections=args.concurrency + 10)
    wall_start = time.perf_counter()
    async with httpx.AsyncClient(limits=limits) as client:
        if args.variant == "c":
            coros = [_one_request_c(client, args.base_url, file_bytes, filename, args.timeout,
                                     args.poll_interval_ms / 1000)
                     for _ in range(args.concurrency)]
        else:
            coros = [_one_request_a_b(client, args.base_url, file_bytes, filename, args.timeout)
                     for _ in range(args.concurrency)]
        results = await asyncio.gather(*coros)
    wall_elapsed = time.perf_counter() - wall_start

    stop_event.set()
    for t in monitor_tasks:
        try:
            await asyncio.wait_for(t, timeout=2.0)
        except asyncio.TimeoutError:
            t.cancel()

    latencies = [r["elapsed_s"] * 1000 for r in results if r["ok"]]
    success = sum(1 for r in results if r["ok"])
    timeouts = sum(1 for r in results if r["timed_out"])
    errors = len(results) - success - timeouts
    status_counts: dict[str, int] = {}
    for r in results:
        key = str(r["status_code"])
        status_counts[key] = status_counts.get(key, 0) + 1

    health_latencies_ms = [s["elapsed_s"] * 1000 for s in health_samples if s["ok"]]

    summary = {
        "variant": args.variant,
        "concurrency": args.concurrency,
        "wall_clock_s": wall_elapsed,
        "throughput_rps": len(results) / wall_elapsed if wall_elapsed > 0 else 0.0,
        "success_count": success,
        "timeout_count": timeouts,
        "error_count": errors,
        "status_counts": status_counts,
        "latency_ms": _latency_stats(latencies),
        "health_probe": {
            "sample_count": len(health_samples),
            "failed_count": sum(1 for s in health_samples if not s["ok"]),
            "latency_ms": _latency_stats(health_latencies_ms),
        },
        "resources": {
            "peak_cpu_percent_sum": max((s["cpu_percent_sum"] for s in resource_samples), default=0.0),
            "peak_rss_mb": max((s["rss_mb"] for s in resource_samples), default=0.0),
            "peak_thread_count": max((s["thread_count"] for s in resource_samples), default=0),
            "peak_process_count": max((s["process_count"] for s in resource_samples), default=0),
        },
    }

    return {
        "summary": summary,
        "results": results,
        "health_samples": health_samples,
        "resource_samples": resource_samples,
    }


def cmd_drive(args: argparse.Namespace) -> None:
    if not args.health_url:
        args.health_url = args.base_url
    raw = asyncio.run(run_driver(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    s = raw["summary"]
    print(f"variant={s['variant']} concurrency={s['concurrency']} "
          f"wall={s['wall_clock_s']:.2f}s throughput={s['throughput_rps']:.2f}rps "
          f"success={s['success_count']} timeout={s['timeout_count']} error={s['error_count']} "
          f"p50={s['latency_ms']['p50']:.0f}ms p95={s['latency_ms']['p95']:.0f}ms "
          f"health_p95={s['health_probe']['latency_ms']['p95']:.0f}ms")


# --------------------------------------------------------------------------- #
# `run`: orchestrates the full variant x load-level x trial matrix.
#
# For each variant (a: async inline, b: threadpool, c: celery) this:
#   1. starts a fresh server (and, for c, a fresh Celery worker) on its own port
#   2. waits for /health to come up
#   3. for each load level, runs one discarded warm-up then N real trials via
#      `drive`, saving each trial's raw JSON under runs/<variant>/<load>/
#   4. tears the server (and worker) down before moving to the next variant
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_healthy(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.3)
    raise RuntimeError(f"server at {url} never became healthy")


def start_server(variant: str) -> subprocess.Popen:
    port = VARIANT_PORTS[variant]
    env = os.environ.copy()
    if VARIANT_NEEDS_CELERY[variant]:
        env.update(REDIS_ENV)
    log(f"starting server for variant {variant} on port {port}")
    proc = subprocess.Popen(
        [str(VENV_PY), str(THIS_FILE), "serve", "--variant", variant, "--port", str(port), "--host", "127.0.0.1"],
        env=env,
        stdout=open(RUNS_DIR / f"server_{variant}.log", "w"), stderr=subprocess.STDOUT,
    )
    wait_healthy(f"http://127.0.0.1:{port}")
    log(f"server for variant {variant} healthy (pid={proc.pid})")
    return proc


def start_celery_worker_proc() -> subprocess.Popen:
    env = os.environ.copy()
    env.update(REDIS_ENV)
    log(f"starting celery worker (concurrency={CELERY_CONCURRENCY})")
    proc = subprocess.Popen(
        [str(VENV_PY), str(THIS_FILE), "celery-worker", "--concurrency", str(CELERY_CONCURRENCY)],
        env=env,
        stdout=open(RUNS_DIR / "celery_worker.log", "w"), stderr=subprocess.STDOUT,
    )
    time.sleep(6)  # mingle/boot time before it's ready to accept tasks
    log(f"celery worker started (pid={proc.pid})")
    return proc


def stop_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    log(f"stopping {name} (pid={proc.pid})")
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        log(f"{name} did not exit in time, killing")
        proc.kill()
        proc.wait(timeout=5)


def clean_bench_dirs() -> None:
    for d in (BENCH_UPLOAD_DIR, BENCH_OUTPUT_DIR):
        if d.exists():
            for f in d.iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass


def run_trial(variant: str, concurrency: int, monitor_pids: list[int], out_path: Path) -> dict:
    base_url = f"http://127.0.0.1:{VARIANT_PORTS[variant]}"
    cmd = [
        str(VENV_PY), str(THIS_FILE), "drive",
        "--variant", variant, "--base-url", base_url,
        "--pdf", str(PDF_PATH), "--concurrency", str(concurrency),
        "--timeout", str(REQUEST_TIMEOUT_S),
        "--monitor-pids", ",".join(str(p) for p in monitor_pids),
        "--out", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        log(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        log(f"  driver stderr: {result.stderr[-2000:]}")
    return json.loads(out_path.read_text())


def run_variant(variant: str) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    clean_bench_dirs()

    celery_proc = start_celery_worker_proc() if VARIANT_NEEDS_CELERY[variant] else None
    server_proc = start_server(variant)
    monitor_pids = [server_proc.pid] + ([celery_proc.pid] if celery_proc else [])

    try:
        for load in LOAD_LEVELS:
            level_dir = RUNS_DIR / variant / str(load)
            level_dir.mkdir(parents=True, exist_ok=True)

            for w in range(WARMUP_RUNS):
                log(f"variant={variant} load={load} warmup {w + 1}/{WARMUP_RUNS}")
                run_trial(variant, load, monitor_pids, level_dir / f"warmup_{w}.json")
                clean_bench_dirs()
                time.sleep(2)

            for t in range(TRIALS_PER_LEVEL):
                log(f"variant={variant} load={load} trial {t + 1}/{TRIALS_PER_LEVEL}")
                run_trial(variant, load, monitor_pids, level_dir / f"trial_{t}.json")
                clean_bench_dirs()
                time.sleep(2)
    finally:
        stop_process(server_proc, f"server[{variant}]")
        if celery_proc:
            stop_process(celery_proc, "celery_worker")
        time.sleep(1)


def cmd_run(args: argparse.Namespace) -> None:
    variants = args.variants or list(VARIANT_IDS)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "machine_info.json").write_text(json.dumps({
        "cpu_count": os.cpu_count(),
        "celery_concurrency": CELERY_CONCURRENCY,
        "load_levels": LOAD_LEVELS,
        "trials_per_level": TRIALS_PER_LEVEL,
        "warmup_runs": WARMUP_RUNS,
        "pdf": str(PDF_PATH),
        "pdf_size_bytes": PDF_PATH.stat().st_size,
    }, indent=2))
    for variant in variants:
        log(f"=== variant {variant} ===")
        run_variant(variant)
    log("all done")


# --------------------------------------------------------------------------- #
# `gil-profile`: standalone micro-benchmark for the GIL-release split in the
# parsing hot path.
#
# Two questions:
#   1. How is wall time split between the MuPDF C call (get_spans_from_page,
#      via page.get_text('dict')) and the pure-Python grouping/classification
#      that follows it?
#   2. Does that split predict how variant B (threadpool) scales with
#      concurrent threads? If the MuPDF call releases the GIL for a real
#      share of the work, running N parses across N threads should be faster
#      than N sequential parses by roughly that share; if it holds the GIL,
#      threaded wall time should be ~= sequential wall time.
# --------------------------------------------------------------------------- #

_GIL_PROFILE_N = 8


def _gil_timed_parse(tag: str) -> dict:
    t0 = time.perf_counter()
    doc = pymupdf.open(str(PDF_PATH))
    t1 = time.perf_counter()
    export_to_json(doc, f"/tmp/gil_profile_{tag}.json")
    t2 = time.perf_counter()
    doc.close()
    return {"open_s": t1 - t0, "export_s": t2 - t1, "total_s": t2 - t0}


def _gil_timed_parse_instrumented() -> dict:
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


def cmd_gil_profile(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    print("--- single-parse split (average of 5) ---")
    splits = [_gil_timed_parse(f"warm{i}") for i in range(5)]
    avg_open = sum(s["open_s"] for s in splits) / len(splits)
    avg_export = sum(s["export_s"] for s in splits) / len(splits)
    avg_total = avg_open + avg_export
    print(f"pymupdf.open:    {avg_open*1000:7.1f} ms  ({avg_open/avg_total*100:5.1f}% of total)")
    print(f"export_to_json:  {avg_export*1000:7.1f} ms  ({avg_export/avg_total*100:5.1f}% of total)")
    print(f"total:           {avg_total*1000:7.1f} ms")

    print("\n--- fine-grained split inside export_to_json (average of 5) ---")
    fine = [_gil_timed_parse_instrumented() for _ in range(5)]
    avg_open2 = sum(s["open_s"] for s in fine) / len(fine)
    avg_c = sum(s["mupdf_touching_s"] for s in fine) / len(fine)
    avg_py = sum(s["python_grouping_s"] for s in fine) / len(fine)
    avg_total2 = avg_open2 + avg_c + avg_py
    print(f"doc.open:                          {avg_open2*1000:7.1f} ms  ({avg_open2/avg_total2*100:5.1f}%)")
    print(f"get_spans_from_page (MuPDF C call): {avg_c*1000:7.1f} ms  ({avg_c/avg_total2*100:5.1f}%)")
    print(f"pure-Python grouping/classify:      {avg_py*1000:7.1f} ms  ({avg_py/avg_total2*100:5.1f}%)")
    print(f"total:                              {avg_total2*1000:7.1f} ms")

    n = _GIL_PROFILE_N
    print(f"\n--- sequential vs threaded scaling, N={n} parses ---")
    t0 = time.perf_counter()
    for i in range(n):
        _gil_timed_parse(f"seq{i}")
    seq_wall = time.perf_counter() - t0
    print(f"sequential wall time for {n} parses: {seq_wall:.2f}s ({seq_wall/n*1000:.1f}ms/parse)")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda i: _gil_timed_parse(f"thr{i}"), range(n)))
    thr_wall = time.perf_counter() - t0
    print(f"threaded ({n} workers) wall time for {n} parses: {thr_wall:.2f}s ({thr_wall/n*1000:.1f}ms/parse)")

    speedup = seq_wall / thr_wall if thr_wall > 0 else 0
    print(f"\nspeedup from threading: {speedup:.2f}x  (1.0x = fully GIL-bound, {n}.0x = fully parallel)")
    implied_gil_released_share = max(0.0, 1 - 1 / speedup) if speedup > 0 else 0.0
    print(f"implied GIL-released share of total work: ~{implied_gil_released_share*100:.0f}%")


# --------------------------------------------------------------------------- #
# `aggregate`: reads raw per-trial JSON (runs/<variant>/<load>/trial_*.json)
# into a mean +/- stddev summary table, and writes plots (latency/throughput/
# health vs concurrency). Warm-up runs (warmup_*.json) are excluded.
# --------------------------------------------------------------------------- #

def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], 0.0)
    return (stats.mean(values), stats.stdev(values))


def _load_trials(variant: str, load: int) -> list[dict]:
    level_dir = RUNS_DIR / variant / str(load)
    trials = []
    for f in sorted(level_dir.glob("trial_*.json")):
        trials.append(json.loads(f.read_text()))
    return trials


def _build_summary_table() -> dict:
    table = {}
    for variant in VARIANT_IDS:
        variant_dir = RUNS_DIR / variant
        if not variant_dir.exists():
            continue
        loads = sorted(int(p.name) for p in variant_dir.iterdir() if p.is_dir())
        table[variant] = {}
        for load in loads:
            trials = _load_trials(variant, load)
            if not trials:
                continue
            summaries = [t["summary"] for t in trials]

            def metric(path_fn):
                return [path_fn(s) for s in summaries]

            wall = metric(lambda s: s["wall_clock_s"])
            throughput = metric(lambda s: s["throughput_rps"])
            p50 = metric(lambda s: s["latency_ms"]["p50"])
            p95 = metric(lambda s: s["latency_ms"]["p95"])
            p99 = metric(lambda s: s["latency_ms"]["p99"])
            lat_max = metric(lambda s: s["latency_ms"]["max"])
            lat_min = metric(lambda s: s["latency_ms"]["min"])
            success = metric(lambda s: s["success_count"])
            timeout = metric(lambda s: s["timeout_count"])
            error = metric(lambda s: s["error_count"])
            health_min = metric(lambda s: s["health_probe"]["latency_ms"]["min"])
            health_p50 = metric(lambda s: s["health_probe"]["latency_ms"]["p50"])
            health_p95 = metric(lambda s: s["health_probe"]["latency_ms"]["p95"])
            health_p99 = metric(lambda s: s["health_probe"]["latency_ms"]["p99"])
            health_max = metric(lambda s: s["health_probe"]["latency_ms"]["max"])
            cpu = metric(lambda s: s["resources"]["peak_cpu_percent_sum"])
            cpu_per_core = metric(lambda s: s["resources"]["peak_cpu_percent_sum"] / (os.cpu_count() or 1))
            rss = metric(lambda s: s["resources"]["peak_rss_mb"])
            threads = metric(lambda s: s["resources"]["peak_thread_count"])
            procs = metric(lambda s: s["resources"]["peak_process_count"])
            total_requests = metric(lambda s: s["success_count"] + s["timeout_count"] + s["error_count"])

            table[variant][load] = {
                "n_trials": len(trials),
                "wall_clock_s": _mean_std(wall),
                "throughput_rps": _mean_std(throughput),
                "latency_min_ms": _mean_std(lat_min),
                "latency_p50_ms": _mean_std(p50),
                "latency_p95_ms": _mean_std(p95),
                "latency_p99_ms": _mean_std(p99),
                "latency_max_ms": _mean_std(lat_max),
                "success_count": _mean_std(success),
                "timeout_count": _mean_std(timeout),
                "error_count": _mean_std(error),
                "success_rate_pct": _mean_std([100 * s / t if t else 0.0 for s, t in zip(success, total_requests)]),
                "timeout_rate_pct": _mean_std([100 * s / t if t else 0.0 for s, t in zip(timeout, total_requests)]),
                "error_rate_pct": _mean_std([100 * s / t if t else 0.0 for s, t in zip(error, total_requests)]),
                "health_min_ms": _mean_std(health_min),
                "health_p50_ms": _mean_std(health_p50),
                "health_p99_ms": _mean_std(health_p99),
                "peak_cpu_percent_per_core": _mean_std(cpu_per_core),
                "health_p95_ms": _mean_std(health_p95),
                "health_max_ms": _mean_std(health_max),
                "peak_cpu_percent_sum": _mean_std(cpu),
                "peak_rss_mb": _mean_std(rss),
                "peak_thread_count": _mean_std(threads),
                "peak_process_count": _mean_std(procs),
            }
    return table


def _fmt(m: tuple[float, float], nd: int = 1) -> str:
    return f"{m[0]:.{nd}f} +/- {m[1]:.{nd}f}"


def _print_summary_table(table: dict) -> None:
    header = (f"{'variant':8} {'load':>6} {'wall_s':>16} {'thpt_rps':>14} {'p50_ms':>16} "
              f"{'p95_ms':>16} {'p99_ms':>16} {'max_ms':>16} {'health_p95_ms':>18} "
              f"{'succ/timeout/err':>18} {'peak_cpu%':>14} {'peak_rss_mb':>16} {'peak_thr/proc':>14}")
    print(header)
    print("-" * len(header))
    for variant in VARIANT_IDS:
        if variant not in table:
            continue
        for load, m in table[variant].items():
            succ = f"{m['success_count'][0]:.0f}/{m['timeout_count'][0]:.0f}/{m['error_count'][0]:.0f}"
            tp = f"{m['peak_thread_count'][0]:.0f}/{m['peak_process_count'][0]:.0f}"
            print(f"{variant:8} {load:>6} {_fmt(m['wall_clock_s'],2):>16} {_fmt(m['throughput_rps'],2):>14} "
                  f"{_fmt(m['latency_p50_ms']):>16} {_fmt(m['latency_p95_ms']):>16} "
                  f"{_fmt(m['latency_p99_ms']):>16} {_fmt(m['latency_max_ms']):>16} "
                  f"{_fmt(m['health_p95_ms']):>18} {succ:>18} {_fmt(m['peak_cpu_percent_sum']):>14} "
                  f"{_fmt(m['peak_rss_mb']):>16} {tp:>14}")


def _plot_metric_vs_concurrency(table: dict, metric_keys: list[str], labels: list[str], title: str,
                                 ylabel: str, out_path: Path, logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in VARIANT_IDS:
        if variant not in table:
            continue
        loads = sorted(table[variant].keys())
        for metric_key, label, style in zip(metric_keys, labels, ["-o", "--s", ":^"]):
            means = [table[variant][l][metric_key][0] for l in loads]
            stds = [table[variant][l][metric_key][1] for l in loads]
            series_label = f"{VARIANT_LABELS[variant]} {label}".strip()
            ax.errorbar(loads, means, yerr=stds, fmt=style, color=VARIANT_COLORS[variant],
                        alpha=1.0 if len(metric_keys) == 1 else (1.0 if label == labels[0] else 0.55),
                        label=series_label, capsize=3)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Concurrency (concurrent requests)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cmd_aggregate(args: argparse.Namespace) -> None:
    table = _build_summary_table()
    _print_summary_table(table)

    out = RUNS_DIR / "summary.json"
    out.write_text(json.dumps(table, indent=2, default=str))
    print(f"\nsummary written to {out}")

    plots_dir = RUNS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    _plot_metric_vs_concurrency(
        table, ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"], ["p50", "p95", "p99"],
        "Request latency (upload_pdf) vs concurrency", "Latency (ms, log scale)",
        plots_dir / "latency_vs_concurrency.png", logy=True,
    )
    _plot_metric_vs_concurrency(
        table, ["throughput_rps"], [""],
        "Throughput vs concurrency", "Requests / second",
        plots_dir / "throughput_vs_concurrency.png",
    )
    _plot_metric_vs_concurrency(
        table, ["health_p95_ms"], ["/health p95"],
        "Event-loop responsiveness (/health latency) vs concurrency", "Health-check latency (ms, log scale)",
        plots_dir / "health_latency_vs_concurrency.png", logy=True,
    )
    print(f"plots written to {plots_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run one variant's FastAPI app under uvicorn")
    p_serve.add_argument("--variant", required=True, choices=VARIANT_IDS)
    p_serve.add_argument("--port", type=int, default=None, help="defaults to VARIANT_PORTS[variant]")
    p_serve.add_argument("--host", default="127.0.0.1")

    p_worker = sub.add_parser("celery-worker", help="run variant c's Celery worker, in-process")
    p_worker.add_argument("--concurrency", type=int, default=CELERY_CONCURRENCY)
    p_worker.add_argument("--loglevel", default="info")

    p_drive = sub.add_parser("drive", help="one load-test trial against an already-running server")
    p_drive.add_argument("--variant", required=True, choices=VARIANT_IDS)
    p_drive.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8001")
    p_drive.add_argument("--health-url", default=None, help="defaults to --base-url if omitted")
    p_drive.add_argument("--pdf", required=True)
    p_drive.add_argument("--concurrency", type=int, required=True)
    p_drive.add_argument("--timeout", type=float, default=120.0, help="per-request end-to-end timeout (s)")
    p_drive.add_argument("--health-interval-ms", type=float, default=100.0)
    p_drive.add_argument("--resource-interval-ms", type=float, default=200.0)
    p_drive.add_argument("--poll-interval-ms", type=float, default=100.0, help="variant c status poll interval")
    p_drive.add_argument("--monitor-pids", default="", help="comma-separated root PIDs to track resource usage")
    p_drive.add_argument("--out", required=True, help="raw JSON output path")

    p_run = sub.add_parser("run", help="orchestrate the full variant x load x trial matrix")
    p_run.add_argument("variants", nargs="*", choices=VARIANT_IDS, default=None,
                        help="which variants to run (default: all three, in order)")

    sub.add_parser("gil-profile", help="GIL-release micro-benchmark for the parsing hot path")
    sub.add_parser("aggregate", help="aggregate runs/ into runs/summary.json + runs/plots/*.png")

    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    handlers = {
        "serve": cmd_serve,
        "celery-worker": cmd_celery_worker,
        "drive": cmd_drive,
        "run": cmd_run,
        "gil-profile": cmd_gil_profile,
        "aggregate": cmd_aggregate,
    }
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

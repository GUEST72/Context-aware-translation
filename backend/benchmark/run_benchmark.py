#!/usr/bin/env python3
"""Orchestrates the full variant x load-level x trial benchmark matrix.

For each variant (a: async inline, b: threadpool, c: celery) this:
  1. starts a fresh server (and, for c, a fresh Celery worker) on its own port
  2. waits for /health to come up
  3. for each load level, runs one discarded warm-up then N real trials via
     load_driver.py, saving each trial's raw JSON under runs/<variant>/<load>/
  4. tears the server (and worker) down before moving to the next variant

Run from anywhere; paths are resolved relative to this file.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
APP_DIR = BACKEND_DIR / "app"
VENV_PY = BACKEND_DIR / "venv" / "bin" / "python"
PDF_PATH = BENCH_DIR / "fixtures" / "bench_book.pdf"
RUNS_DIR = BENCH_DIR / "runs"

VARIANTS = {
    "a": {"module": "bench_variant_a:app", "port": 8001, "needs_celery": False},
    "b": {"module": "bench_variant_b:app", "port": 8002, "needs_celery": False},
    "c": {"module": "bench_variant_c:app", "port": 8003, "needs_celery": True},
}

LOAD_LEVELS = [10, 100, 1000]
TRIALS_PER_LEVEL = 3
WARMUP_RUNS = 1
REQUEST_TIMEOUT_S = 180.0
CELERY_CONCURRENCY = os.cpu_count() or 4
REDIS_ENV = {
    "CELERY_BROKER_URL": "redis://127.0.0.1:6379/0",
    "CELERY_RESULT_BACKEND": "redis://127.0.0.1:6379/1",
}


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
    cfg = VARIANTS[variant]
    env = os.environ.copy()
    if cfg["needs_celery"]:
        env.update(REDIS_ENV)
    log(f"starting server for variant {variant} on port {cfg['port']}")
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", cfg["module"], "--port", str(cfg["port"]), "--host", "127.0.0.1"],
        cwd=str(APP_DIR), env=env,
        stdout=open(RUNS_DIR / f"server_{variant}.log", "w"), stderr=subprocess.STDOUT,
    )
    wait_healthy(f"http://127.0.0.1:{cfg['port']}")
    log(f"server for variant {variant} healthy (pid={proc.pid})")
    return proc


def start_celery_worker() -> subprocess.Popen:
    env = os.environ.copy()
    env.update(REDIS_ENV)
    log(f"starting celery worker (concurrency={CELERY_CONCURRENCY})")
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "celery", "-A", "celery_config.celery_app", "worker",
         "--loglevel=info", f"--concurrency={CELERY_CONCURRENCY}", "-P", "prefork"],
        cwd=str(APP_DIR), env=env,
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
    for d in (APP_DIR / "bench_uploads", APP_DIR / "bench_outputs"):
        if d.exists():
            for f in d.iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass


def run_trial(variant: str, concurrency: int, monitor_pids: list[int], out_path: Path) -> dict:
    cfg = VARIANTS[variant]
    base_url = f"http://127.0.0.1:{cfg['port']}"
    cmd = [
        str(VENV_PY), str(BENCH_DIR / "load_driver.py"),
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
    cfg = VARIANTS[variant]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    clean_bench_dirs()

    celery_proc = start_celery_worker() if cfg["needs_celery"] else None
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


def main() -> int:
    variants = sys.argv[1:] or list(VARIANTS.keys())
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

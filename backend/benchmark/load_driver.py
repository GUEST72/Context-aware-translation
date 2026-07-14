#!/usr/bin/env python3
"""Load driver for the async-vs-threadpool-vs-celery upload_pdf benchmark.

Fires `--concurrency` concurrent POST /upload_pdf requests against a single
running server variant, measures end-to-end latency per request (for the
celery variant this includes polling /upload_pdf/status/{task_id} until the
worker finishes -- not just enqueue time), probes GET /health every
~100ms throughout the run to quantify event-loop responsiveness, and samples
CPU/RAM/thread-or-process count of the given PIDs (and their descendants)
throughout the run.

One process of this script = one trial. Warm-up vs. real trials, and
restarting the server between variants, is handled by run_benchmark.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx
import psutil


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


async def run(args: argparse.Namespace) -> dict[str, Any]:
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

    raw = {
        "summary": summary,
        "results": results,
        "health_samples": health_samples,
        "resource_samples": resource_samples,
    }
    return raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["a", "b", "c"])
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8001")
    p.add_argument("--health-url", default=None, help="defaults to --base-url if omitted")
    p.add_argument("--pdf", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--timeout", type=float, default=120.0, help="per-request end-to-end timeout (s)")
    p.add_argument("--health-interval-ms", type=float, default=100.0)
    p.add_argument("--resource-interval-ms", type=float, default=200.0)
    p.add_argument("--poll-interval-ms", type=float, default=100.0, help="variant c status poll interval")
    p.add_argument("--monitor-pids", default="", help="comma-separated root PIDs to track resource usage")
    p.add_argument("--out", required=True, help="raw JSON output path")
    args = p.parse_args()
    if not args.health_url:
        args.health_url = args.base_url
    return args


def main() -> int:
    args = parse_args()
    raw = asyncio.run(run(args))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    s = raw["summary"]
    print(f"variant={s['variant']} concurrency={s['concurrency']} "
          f"wall={s['wall_clock_s']:.2f}s throughput={s['throughput_rps']:.2f}rps "
          f"success={s['success_count']} timeout={s['timeout_count']} error={s['error_count']} "
          f"p50={s['latency_ms']['p50']:.0f}ms p95={s['latency_ms']['p95']:.0f}ms "
          f"health_p95={s['health_probe']['latency_ms']['p95']:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

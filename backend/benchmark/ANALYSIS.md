# Benchmark: async inline vs. threadpool vs. Celery for `/upload_pdf`

Full raw results: `runs/*/*/trial_*.json` (and `warmup_*.json`, discarded from all
statistics). Aggregated table: `runs/summary.json`. Plots: `runs/plots/`. Predictions
written before the full-load results came in: `PREDICTIONS.md`.

## Setup

- **Machine:** 12 logical cores, 14GB RAM (~8GB "available" once reclaimable cache is
  counted). This caps variant C's real parallelism at 12 concurrent processes.
- **Test PDF:** a 12-page, 2.83MB excerpt (pages 11-22) of
  `computer-networking-a-top-down-approach-8th-edition.pdf`, built with the same parser
  code path (`backend/benchmark/fixtures/bench_book.pdf`). The full 771-page/25MB book
  was rejected for the load matrix: a single parse takes ~4.2s and holds the GIL almost
  the entire time, so 1000-concurrent trials on it would serialize to 70+ minutes each for
  variant A, and buffering 1000 concurrent 25MB uploads would need ~25GB of RAM against
  ~8GB available. The 12-page excerpt parses in ~150-160ms and keeps 1000-concurrent
  buffer memory around ~2.8GB, which is what made a same-day, same-machine 10/100/1000
  matrix possible at all. This is a deliberate deviation from the original "same PDF"
  instruction, documented here for reproducibility.
- **Load levels:** 10 / 100 / 1000 concurrent requests, 1 discarded warm-up + 3 real
  trials per (variant, load) cell, server restarted fresh between variants
  (`benchmark.py run`). 3 trials instead of the requested 3-5, to keep total runtime
  practical given variant A/B's 1000-concurrency trials take minutes each.
- **Threadpool ceiling (variant B):** Starlette/AnyIO default worker-thread limiter, 40.
- **Celery (variant C):** `--concurrency=12` (`= os.cpu_count()`), prefork pool, Redis
  6379 broker+backend (`redis:7-alpine` via Docker, no host package changes).
- **Per-request timeout:** 180s end-to-end (for C, that's enqueue + poll-until-SUCCESS).
- **Driver:** custom `asyncio` + `httpx` script (`benchmark.py drive`), not Locust/k6, per your
  choice. It fires all N requests concurrently via `asyncio.gather`, polls
  `/upload_pdf/status/{task_id}` every 100ms for variant C, probes `/health` every ~100ms
  throughout the run on a separate client, and samples CPU%/RSS/thread-or-process count of
  the server (+ Celery worker tree for C) every 200ms via `psutil`.
- **Correctness fixes made before benchmarking** (required — the original code would have
  produced meaningless/corrupted results under concurrency):
  - `backend/app/parser/classifier.py` had leftover debug script code at module scope that
    hard-opened an absolute path from a different machine on import — this crashed every
    import of the parser, including the existing Celery worker. Removed.
  - Each benchmark variant writes both the uploaded file and the parsed JSON output to a
    UUID-named path per request (`unique_paths()` in `benchmark.py`), never touching the real app's shared
    `BOOK_PATH`/`BOOK_DATA` global.

## One important correction to the brief's own hypothesis

The brief's caveat was: *"pymupdf may release the GIL during native parsing... variant B
might outperform a naive CPU-bound expectation."* I measured this directly
(`benchmark.py gil-profile`) before running the load matrix:

- Of the ~150-160ms to parse one request, ~98% is inside `get_spans_from_page()`'s call to
  `page.get_text("dict")` — the actual MuPDF C call — and only ~1.7% is pure-Python
  grouping/classification.
- Despite that, running 8 parses across 8 threads was **not faster** than running them
  sequentially (0.90x "speedup", i.e. slightly slower once thread overhead is included) —
  an empirically measured **~0% GIL-released share**.

So in this pymupdf build, `get_text("dict")` does **not** release the GIL for this
workload. The prediction this implies (written in `PREDICTIONS.md` before the load
matrix ran) was that variant B would get **no throughput benefit** from threading, only
whatever event-loop-responsiveness benefit comes from moving work off the event-loop
thread. The load-matrix results confirm the first half and refute the second half (see
below) — B doesn't even reliably deliver the responsiveness benefit at high load.

## Results (mean ± stddev over 3 trials; warm-up discarded)

#### Table 1 — total wall-clock time & throughput

| variant | load | total wall-clock (s) | throughput (req/s) |
|---|---:|---:|---:|
| A (async) | 10 | 1.82 ± 0.04 | 5.49 ± 0.13 |
| A (async) | 100 | 21.37 ± 1.10 | 4.69 ± 0.24 |
| A (async) | 1000 | 234.44 ± 90.10 | 4.64 ± 1.46 |
| B (threadpool) | 10 | 1.75 ± 0.04 | 5.72 ± 0.12 |
| B (threadpool) | 100 | 20.81 ± 0.85 | 4.81 ± 0.20 |
| B (threadpool) | 1000 | 339.13 ± 7.90 | **2.95 ± 0.07** |
| C (celery) | 10 | 0.48 ± 0.01 | 20.89 ± 0.52 |
| C (celery) | 100 | 3.39 ± 0.03 | 29.47 ± 0.25 |
| C (celery) | 1000 | 35.25 ± 0.99 | **28.38 ± 0.81** |

#### Table 2 — `upload_pdf` end-to-end latency distribution (ms)

For C this is enqueue + poll-until-done, not just enqueue time.

| variant | load | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| A (async) | 10 | 236 ± 12 | 909 ± 19 | 1749 ± 42 | 1749 ± 42 | 1749 ± 42 |
| A (async) | 100 | 836 ± 59 | 10,812 ± 507 | 20,261 ± 1042 | 21,073 ± 1089 | 21,278 ± 1104 |
| A (async) | 1000 | 41,999 ± 72,744 | 73,527 ± 127,352 | 103,212 ± 178,769 | 105,719 ± 183,110 | 106,500 ± 184,464 |
| B (threadpool) | 10 | 1116 ± 502 | 1625 ± 103 | 1678 ± 33 | 1678 ± 33 | 1678 ± 33 |
| B (threadpool) | 100 | 899 ± 168 | 12,089 ± 770 | 20,304 ± 430 | 20,683 ± 848 | 20,715 ± 847 |
| B (threadpool) | 1000 | 121,337 ± 22,876 | **223,590 ± 34,228** | 289,147 ± 26,087 | 311,461 ± 21,232 | 315,171 ± 22,284 |
| C (celery) | 10 | 306 ± 41 | 359 ± 12 | 389 ± 13 | 389 ± 13 | 389 ± 13 |
| C (celery) | 100 | 1178 ± 189 | 2850 ± 78 | 3265 ± 28 | 3288 ± 19 | 3290 ± 18 |
| C (celery) | 1000 | 13,482 ± 6798 | **31,893 ± 1198** | 34,767 ± 953 | 35,051 ± 985 | 35,123 ± 983 |

The huge stddev on A's own numbers at load=1000 (min: 42,000 ± 72,744ms) is itself a
finding: A doesn't just get slow, it gets *unpredictable* — which request lands where in
the serialized queue is essentially arbitrary, so successful-request latency swings wildly
trial to trial depending on scheduling luck.

#### Table 3 — success / timeout / error rate (per load-level N requests, 180s per-request timeout)

| variant | load | success | timeout | error | success % | timeout % | error % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (async) | 10 | 10 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| A (async) | 100 | 100 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| A (async) | 1000 | 142 | 858 | 0 | 14.2 ± 24.6 | **85.8 ± 24.6** | 0.0 |
| B (threadpool) | 10 | 10 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| B (threadpool) | 100 | 100 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| B (threadpool) | 1000 | 513 | 487 | 0 | 51.3 ± 11.3 | **48.7 ± 11.3** | 0.0 |
| C (celery) | 10 | 10 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| C (celery) | 100 | 100 | 0 | 0 | 100.0 | 0.0 | 0.0 |
| C (celery) | 1000 | 997 | 0 | 3 | 99.7 ± 0.3 | 0.0 | 0.3 ± 0.3 |

#### Table 4 — `/health` probe latency (ms): the event-loop-freeze quantified

Probed every ~100ms from a separate client for the duration of each run.

| variant | load | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| A (async) | 10 | 44.1 ± 0.7 | 44.1 ± 0.7 | 1632.4 ± 42.2 | 1632.4 ± 42.2 | 1632.4 ± 42.2 |
| A (async) | 100 | 11.2 ± 1.5 | 320.6 ± 145.0 | 5580.6 ± 928.5 | 5580.6 ± 928.5 | 5580.6 ± 928.5 |
| A (async) | 1000 | 3.3 ± 4.3 | 11.9 ± 18.8 | 3620.7 ± 3555.4 | 4404.6 ± 4762.7 | 4888.5 ± 5542.8 |
| B (threadpool) | 10 | 49.7 ± 8.5 | 159.6 ± 42.2 | 405.0 ± 81.3 | 405.0 ± 81.3 | 405.0 ± 81.3 |
| B (threadpool) | 100 | 5.4 ± 2.3 | 497.6 ± 260.9 | 3221.6 ± 1082.2 | 3867.1 ± 399.5 | 3867.1 ± 399.5 |
| B (threadpool) | 1000 | 3.5 ± 1.6 | 450.6 ± 416.3 | **7693.9 ± 974.0** | 9464.2 ± 626.4 | 9658.6 ± 328.4 |
| C (celery) | 10 | 2.2 ± 0.7 | 5.4 ± 2.8 | 67.7 ± 0.2 | 67.7 ± 0.2 | 67.7 ± 0.2 |
| C (celery) | 100 | 8.2 ± 1.8 | 124.0 ± 20.6 | 302.6 ± 19.5 | 302.6 ± 19.5 | 302.6 ± 19.5 |
| C (celery) | 1000 | 5.4 ± 2.0 | 94.0 ± 44.7 | **1185.6 ± 731.3** | 6764.2 ± 2131.1 | 6764.2 ± 2131.1 |

Note the gap between **min** and **p95/max** for A and B: this is the freeze made visible.
A `/health` probe that happens to land in a gap between blocking parses returns in a few
ms (A@10: min=44ms); one that lands *during* a parse queues behind it and comes back
seconds later (A@10: p95=1632ms — a >35x spread at only 10 concurrent uploads). C's min
and p95 stay within a much narrower band at every load level (10ms vs 68-1186ms) because
the FastAPI process is never the one doing the blocking work.

#### Table 5 — resource usage (peak observed during the run)

CPU% is summed across the server process (+ Celery worker tree for C); "per core" divides
that by this machine's 12 logical cores as a normalized utilization estimate.

| variant | load | peak CPU % (summed) | peak CPU % (per core) | peak RAM (MB) | peak thread count | peak process count |
|---|---:|---:|---:|---:|---:|---:|
| A (async) | 10 | 124 ± 2 | 10.4 ± 0.2 | 192 ± 23 | 11 ± 0 | 1 |
| A (async) | 100 | 174 ± 7 | 14.5 ± 0.6 | 1137 ± 268 | 41 ± 0 | 1 |
| A (async) | 1000 | 216 ± 97 | 18.0 ± 8.1 | 5937 ± 729 | 38 ± 6 | 1 |
| B (threadpool) | 10 | 124 ± 7 | 10.3 ± 0.6 | 255 ± 22 | 11 ± 0 | 1 |
| B (threadpool) | 100 | 172 ± 3 | 14.4 ± 0.2 | 1456 ± 297 | 48 ± 12 | 1 |
| B (threadpool) | 1000 | 288 ± 21 | 24.0 ± 1.7 | 6392 ± 520 | 49 ± 14 | 1 |
| C (celery) | 10 | 701 ± 45 | 58.5 ± 3.8 | 1269 ± 26 | 33 ± 1 | 14 |
| C (celery) | 100 | **1090 ± 12** | **90.9 ± 1.0** | 2250 ± 254 | 75 ± 0 | 14 |
| C (celery) | 1000 | **1086 ± 11** | **90.5 ± 0.9** | 7467 ± 3175 | 54 ± 0 | 14 |

A/B never exceed ~24% of the machine's total CPU capacity even under load — direct
confirmation the GIL, not core count, is their ceiling. C reaches ~90% of all 12 cores
(process count 14 = 12 prefork workers + 1 celery master + 1 FastAPI process). Peak RAM
for all three grows with concurrency mostly because the endpoint buffers the whole upload
in memory (`await file.read()` / `file.file.read()`) before writing to disk — that's a
memory-scaling factor identical across variants, not a parsing artifact; C's higher
baseline RAM (1269MB even at load=10) is the fixed cost of 12 worker processes each
importing pymupdf.

## Analysis

**A (async inline) — event-loop freeze, exactly as predicted.** `async def upload_pdf`
runs `pymupdf.open()`/`export_to_json()` inline with no `await` inside — a genuinely
GIL-bound, uninterruptible call from the event loop's point of view. Requests fully
serialize: wall-clock time and p50 latency scale roughly linearly with concurrency (909ms
→ 10.8s → 73.5s median as load goes 10 → 100 → 1000), and at 1000 concurrent, **858 of
1000 requests (86%) time out** against the 180s budget — the classic cascading-failure
signature of an unbounded FIFO queue behind a single serialized worker. `/health` latency
degrades in lockstep at low/medium load (1.6s p95 already at only 10 concurrent uploads
in flight, since every `/health` coroutine has to wait its turn on the same
single-threaded, single-core execution). At 1000 concurrent, `/health` p95 actually reads
*lower* (3.6s) than at 100 (5.6s) with very high variance (±3.6s) — an artifact, not a
recovery: so many main requests time out before completing a parse that the total amount
of blocking work done during the sampling window is lower than at load=100, where nearly
everything still completed. Peak CPU stays at ~124-216% (roughly 1-2 cores' worth) even
under load — direct confirmation that the GIL, not available cores, is the ceiling.

**B (threadpool) — offloading without a throughput payoff, and it drags /health down
too.** Declaring `def upload_pdf` (no `async`) makes Starlette run it via
`anyio.to_thread.run_sync`, off the event-loop thread. But because the GIL-profiling
measurement above shows ~0% GIL release inside the parse, moving the work to a thread
doesn't create real parallelism — it just adds thread-scheduling and GIL-contention
overhead on top of the same serialized work. That's visible directly: **B's throughput at
load=1000 (2.95 req/s) is *worse* than A's (4.64 req/s)**, and B's p50 latency (224s) is 3x
A's (73.5s). The threadpool does reduce the *timeout rate* somewhat (487/1000 vs A's
858/1000) — plausibly because work gets distributed across ~40 concurrently-running
threads instead of one strict FIFO event-loop queue, so no single very-late request waits
behind literally all 999 others — but total completed work goes down, not up. Most
strikingly, **B's `/health` responsiveness is not reliably better than A's, and is worse
at load=1000** (7.7s p95 vs A's 3.6s, though A's number there is the timeout-inflated
artifact described above; at load=100, a cleaner comparison, B is 3.2s vs A's 5.6s — better
but still far from "responsive"). The mechanism: even though the parsing coroutine itself
never blocks the event loop with an `await`, the event-loop *thread* and the ~40 worker
*threads* are all OS threads inside one CPython process sharing one GIL. When 40 threads
are constantly doing GIL-bound work, the event-loop thread has to wait its turn for GIL
time slices (CPython's switch interval, default 5ms) just like any other thread, so
`/health` — itself a coroutine that needs to actually execute Python bytecode to run — gets
delayed too. Offloading to a thread avoids explicit `await`-blocking, but it does not
exempt the event loop from GIL contention when the offloaded work is itself GIL-bound.
Peak thread count caps at ~48-49 (the ~40-thread AnyIO limiter plus a handful of
housekeeping threads), confirming the predicted ~40-thread ceiling — but since throughput
was never limited by thread *count* here (it was limited by the GIL), that ceiling turned
out not to be the binding constraint the brief expected.

**C (Celery) — true multicore parallelism, the clear winner.** Each of the 12 worker
processes has its own interpreter and its own GIL, so parsing genuinely runs in parallel
across cores. Peak CPU usage (summed across processes) reaches ~1086-1090% — essentially
saturating all 12 cores — versus A/B's ~124-288%, a direct, measured confirmation of real
multiprocess parallelism where A/B only had the appearance of concurrency. Wall-clock time
at load=1000 is 35s vs. A's 234s and B's 339s (~7-10x faster), throughput holds at ~28-29
req/s essentially flat from load=100 to load=1000 (the system is already core-saturated at
100, so 1000 just queues more evenly behind the same 12 workers rather than degrading), and
there are **zero timeouts** at any load level. The FastAPI process itself only ever
enqueues and polls, so `/health` stays far more responsive than A/B (1.2s p95 at
load=1000, vs. 3.6-7.7s) — though it's not free: `/health` still degrades from a 68ms
baseline at load=10 to 1.2s at load=1000, because the single FastAPI process has to
compete for CPU scheduling time against 12 fully-saturated worker processes on a
12-core machine, and because polling 1000 in-flight status checks plus handling 1000
concurrent upload connections is itself real work for that one process. The small error
count at load=1000 (3-6 of 1000, `RemoteProtocolError`/`ReadError`) is a connection-level
artifact of a single-process uvicorn server accepting/reading 1000 simultaneous HTTP
connections — not a parsing failure or a Celery task failure — and would very likely
shrink with multiple uvicorn workers in front of Celery (see recommendation below).

## Verdict

**Async was the wrong tool for this endpoint, and the margin is severe, not marginal.**
At 1000 concurrent requests, the async-inline implementation fails outright (86% timeout
rate, 234s mean wall-clock with 90s of trial-to-trial stddev — i.e. also unpredictable),
and naively "fixing" it by switching to a plain `def` (threadpool offload) makes
*throughput* measurably worse (2.95 vs 4.64 req/s) while still leaving `/health`
unresponsive under load, because the underlying bottleneck was never really about
sync-vs-async scheduling — it was the GIL, and no amount of thread-based offloading
inside one process escapes that for GIL-bound work. Only true multiprocess parallelism
(Celery/RQ/multiprocessing — anything with one interpreter per core) actually buys
throughput: ~6x A's throughput and ~10x A's wall-clock time at load=1000, zero timeouts at
any tested load level, and the only implementation where `/health` stays meaningfully
responsive under load. **Recommendation for production:** keep the Celery-based
architecture already in `main.py`, size `--concurrency` to the deployment host's core
count (this machine: 12; re-tune per host), and additionally run the FastAPI layer itself
behind multiple uvicorn workers (e.g. `--workers 2-4`) since even variant C's single
API process became a secondary bottleneck (rising `/health` latency, a handful of
connection-level errors) at 1000 simultaneous HTTP connections purely from
accept/read load, separate from parsing.

# upload_pdf benchmark: async inline vs. threadpool vs. Celery

See `ANALYSIS.md` for the full write-up (setup, results table, per-variant analysis,
verdict). `PREDICTIONS.md` has the predictions written before the load matrix ran.

## Layout

- `fixtures/bench_book.pdf` — 12-page benchmark PDF (see `ANALYSIS.md` for why it's not
  the full book).
- `../app/bench_common.py` — shared UUID-path helpers + `parse_pdf()`, used by all three
  variants so concurrent requests never collide and never touch the real app's shared
  `BOOK_PATH`/global state.
- `../app/bench_variant_a.py` — `async def upload_pdf`, parsing inline on the event loop.
- `../app/bench_variant_b.py` — plain `def upload_pdf`, offloaded to the AnyIO threadpool.
- `../app/bench_variant_c.py` — enqueues to the existing Celery task (`../app/tasks.py`);
  exposes `/upload_pdf/status/{task_id}` for polling.
- `load_driver.py` — one process = one trial. Fires N concurrent requests, measures
  end-to-end latency (for C: enqueue + poll until done), probes `/health` every ~100ms
  throughout, samples CPU/RSS/thread-or-process count of given PIDs every 200ms.
- `run_benchmark.py` — orchestrates the full variant x load x trial matrix: starts/stops
  each server (and Celery worker for C) fresh, runs 1 warm-up (discarded) + 3 trials per
  (variant, load) cell, writes raw JSON to `runs/<variant>/<load>/`.
- `gil_profile.py` — standalone micro-benchmark: splits parse time between the MuPDF C
  call and pure-Python grouping/classification, and measures actual GIL-released share via
  sequential-vs-threaded scaling.
- `aggregate.py` — reads `runs/**/trial_*.json`, prints the mean±stddev table, writes
  `runs/summary.json` and `runs/plots/*.png`.

## Reproducing

Requires Docker (for Redis) and the backend venv (`pip install -r backend/requirements.txt`
plus `celery redis psutil matplotlib` — already added to the venv used here).

```bash
docker run -d --name bench-redis -p 6379:6379 redis:7-alpine   # if not already running
cd backend/benchmark
../venv/bin/python gil_profile.py           # ~15s
../venv/bin/python run_benchmark.py a b c   # ~45-60 min; runs all 3 variants
../venv/bin/python aggregate.py             # prints table, writes runs/summary.json + plots
```

Run a single variant with `run_benchmark.py a` (or `b`, `c`). Edit `LOAD_LEVELS`,
`TRIALS_PER_LEVEL`, `WARMUP_RUNS`, or `CELERY_CONCURRENCY` at the top of
`run_benchmark.py` to change the matrix. `runs/*.log` and `runs/*.pid` are gitignored
(server/worker stdout, regenerable); the raw per-trial JSON under `runs/<variant>/<load>/`,
`runs/summary.json`, and `runs/plots/*.png` are committed.

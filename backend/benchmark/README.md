# upload_pdf benchmark: async inline vs. threadpool vs. Celery

See `ANALYSIS.md` for the full write-up (setup, results table, per-variant analysis,
verdict). `PREDICTIONS.md` has the predictions written before the load matrix ran.

All benchmark code lives in one file, `benchmark.py`, with subcommands for each piece:
the three variant FastAPI apps (`app_a`/`app_b`/`app_c`), the Celery worker used by
variant c, the load driver, the matrix orchestrator, the GIL-split micro-benchmark, and
the results aggregator/plotter. Run `python benchmark.py --help` (or
`python benchmark.py <command> --help`) for the full CLI.

## Layout

- `fixtures/bench_book.pdf` — 12-page benchmark PDF (see `ANALYSIS.md` for why it's not
  the full book).
- `benchmark.py` — everything:
  - `unique_paths()` / `parse_pdf()` — shared UUID-path helpers used by all three
    variants so concurrent requests never collide and never touch the real app's shared
    `BOOK_PATH`/global state.
  - `app_a` (`serve --variant a`) — `async def upload_pdf`, parsing inline on the event loop.
  - `app_b` (`serve --variant b`) — plain `def upload_pdf`, offloaded to the AnyIO threadpool.
  - `app_c` (`serve --variant c`) — enqueues to the existing Celery task
    (`../app/tasks.py`); exposes `/upload_pdf/status/{task_id}` for polling.
    `celery-worker` runs that task's Celery worker in-process.
  - `drive` — one process = one trial. Fires N concurrent requests, measures end-to-end
    latency (for C: enqueue + poll until done), probes `/health` every ~100ms throughout,
    samples CPU/RSS/thread-or-process count of given PIDs every 200ms.
  - `run` — orchestrates the full variant x load x trial matrix: starts/stops each server
    (and Celery worker for C) fresh via subprocesses of `benchmark.py serve`/
    `celery-worker`, runs 1 warm-up (discarded) + 3 trials per (variant, load) cell via
    `benchmark.py drive`, writes raw JSON to `runs/<variant>/<load>/`.
  - `gil-profile` — standalone micro-benchmark: splits parse time between the MuPDF C
    call and pure-Python grouping/classification, and measures actual GIL-released share
    via sequential-vs-threaded scaling.
  - `aggregate` — reads `runs/**/trial_*.json`, prints the mean±stddev table, writes
    `runs/summary.json` and `runs/plots/*.png`.

## Reproducing

Requires Docker (for Redis) and the backend venv (`pip install -r backend/requirements.txt`
plus `celery redis psutil matplotlib` — already added to the venv used here).

```bash
docker run -d --name bench-redis -p 6379:6379 redis:7-alpine   # if not already running
cd backend/benchmark
../venv/bin/python benchmark.py gil-profile     # ~15s
../venv/bin/python benchmark.py run a b c       # ~45-60 min; runs all 3 variants
../venv/bin/python benchmark.py aggregate       # prints table, writes runs/summary.json + plots
```

Run a single variant with `benchmark.py run a` (or `b`, `c`). Edit `LOAD_LEVELS`,
`TRIALS_PER_LEVEL`, `WARMUP_RUNS`, or `CELERY_CONCURRENCY` near the top of `benchmark.py`
to change the matrix. To drive a server you started manually:

```bash
../venv/bin/python benchmark.py serve --variant a --port 8001 &
../venv/bin/python benchmark.py drive --variant a --base-url http://127.0.0.1:8001 \
  --pdf fixtures/bench_book.pdf --concurrency 100 --out /tmp/trial.json
```

`runs/*.log` and `runs/*.pid` are gitignored (server/worker stdout, regenerable); the raw
per-trial JSON under `runs/<variant>/<load>/`, `runs/summary.json`, and `runs/plots/*.png`
are committed.

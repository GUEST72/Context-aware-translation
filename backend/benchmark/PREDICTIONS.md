# Predictions (written before the full benchmark matrix finished)

Written after: fixing the parser import bug, confirming the app runs, and running the
GIL-split micro-benchmark (`gil_profile.py`) and a few low-concurrency smoke trials.
Written before: seeing variant B's or C's results at load=100/1000, and before seeing any
aggregated table.

## Known going in

- Single parse of the 12-page benchmark PDF: ~150-160ms, of which ~98% is inside
  `get_spans_from_page`'s `page.get_text("dict")` MuPDF call and ~1.7% is pure-Python
  grouping/classification.
- Threaded-vs-sequential micro-benchmark measured **~0% GIL release** for that MuPDF call in
  this pymupdf build (0.90x "speedup" with 8 threads, i.e. threading is a bit slower than
  sequential once overhead is included). This already contradicts the brief's caveat that
  pymupdf might release the GIL during parsing -- empirically, on this workload, it does not.
- Variant A smoke data at load=10 already shows /health p95 at ~1.6-1.8s, i.e. the event loop
  is visibly starved even at trivial concurrency.

## Predictions

- **Variant A (async inline):** since parsing is ~100% GIL-bound and runs on the event-loop
  thread, requests will fully serialize. Expect wall-clock time to scale ~linearly with
  concurrency (~150ms x N), /health latency to degrade in lockstep with load (already visible
  at N=10), and likely timeouts to start appearing at N=1000 if the 180s per-request timeout
  is tight relative to queueing depth (1000 x 150ms ~= 150s serialized, so it should just
  barely avoid timeouts, but variance/GIL overhead could push some requests over).
- **Variant B (threadpool):** given the ~0% measured GIL release, I predict B's *throughput*
  will be statistically indistinguishable from A's -- the threadpool does not buy real
  parallelism here because there's no GIL-free work to overlap. The only difference I expect
  is /health latency: since parsing runs off the event-loop thread, /health should stay
  responsive (low ms) even as upload_pdf throughput matches A. At load=1000, requests queue
  behind Starlette's default ~40-thread limiter, but since the work is GIL-bound anyway this
  queueing shouldn't change aggregate throughput -- it would only matter if the GIL were not
  the bottleneck.
- **Variant C (celery):** true multi-process parallelism, one worker process per core
  (concurrency=12 on this machine). Expect wall-clock time to scale ~linearly with
  concurrency/12 rather than concurrency, i.e. roughly an order of magnitude better wall time
  than A/B at load=100 and load=1000, and /health to stay responsive throughout since the
  FastAPI process itself never blocks on parsing (only enqueues and polls).

## What would surprise me / falsify this

- If B's throughput meaningfully beats A's at load=100/1000, that would mean the GIL-release
  micro-benchmark wasn't representative of behavior under real concurrent HTTP load (e.g. I/O
  around file read/write releasing the GIL more than expected at higher N).
- If C doesn't scale close to core-count parallelism, that would point to Celery/Redis
  overhead or task-queue contention dominating over actual parse time at this PDF size.
- If A doesn't show timeouts or severe health-probe degradation at load=1000, that would
  suggest per-request overhead is much lower than the ~150ms single-parse figure once
  concurrent I/O is involved (unlikely given the GIL-bound profile, but possible artifact of
  the driver or server).

# Context-Aware Translation — Design Document

## 1. Purpose

Context-Aware Translation is a system that translates a user-selected passage from a
book **without losing the surrounding context**. Instead of translating an isolated
paragraph, the system extracts relevant context and terminology *from the book itself*
and feeds that context into the translation model, so the output is faithful to how the
term or passage is used in that specific book.

**Target domain:** technical textbooks, where terminology and definitions introduced in
one place are reused (and assumed) elsewhere, and where a context-blind translation
frequently gets domain terms wrong.

## 2. Status legend

Throughout this document, each capability is tagged with its current state:

- `[DONE / develop]` — implemented and present on the `develop` branch.
- `[DONE / off-branch]` — implemented, but living on a separate branch not yet merged.
- `[TODO]` — designed or desired, not yet implemented.
- `[OPEN]` — an unresolved design question.

## 3. High-level architecture

The system has three functional areas:

1. **Authentication** — sign up / log in, backed by a user store. `[TODO]`
2. **Book upload & parsing** — receive a PDF, deduplicate it, and convert it to a
   structured JSON representation via the in-house parser. Partially `[DONE]`.
3. **Translation** — given a user's selected text, locate it in the book, assemble
   context around it, and translate via a model fallback chain. Partially `[DONE]`.

Two of these areas (parsing and translation) involve heavy work and must support many
concurrent users, so both are designed around a task-queue / worker model rather than
running inline in the request handler.

> **Assumption:** stack is FastAPI (backend) + Celery + Redis (queue/workers). The
> persistent stores (user DB, book metadata DB, model-state store) are not yet chosen;
> this document proposes PostgreSQL for relational metadata and object storage for the
> PDF/JSON blobs. See §6 and the open questions.

---

## 4. Authentication `[TODO]`

Not implemented and not present on `develop`.

### Requirements

- A user can either use an existing account or create a new one.
- A persistent user store holding, at minimum: `user_id`, `username`, `password_hash`.

### Sign up

- Available only to users without an existing account.
- The client submits a form containing the fields required to create an account
  (e.g. `username`, `password`).
- The backend performs basic validation, then inserts the new user **only if** the
  username is not already taken and the input is valid.

### Log in

- An existing user submits `username` + `password`.
- The backend verifies the credentials against the stored hash.

### Work to add

- Two API endpoints: `POST /signup` and `POST /login`.
- A user database (see §6).

> **Note:** passwords must be stored as hashes (argon2 or bcrypt), never plaintext, and
> session/token handling should use a vetted library rather than a hand-rolled scheme.

---

## 5. Book upload & parsing

### 5.1 Upload + parse pipeline `[DONE / develop]`

The user uploads a PDF via an HTTP `POST` request. The backend receives the file and runs
the **in-house PDF parser pipeline** (a project-specific library already built). The
pipeline produces a **JSON representation of the book** whose structure mirrors the book:
it extracts paragraphs, sections, and subsections.

### 5.2 Content-based deduplication `[TODO]`

Parsing is expensive, so the same book must never be parsed twice. Each book gets an
**ID derived from its content** so that:

1. The ID is generated from the **PDF itself**, not from the resulting JSON.
2. Two uploads receive the **same ID if and only if they are the same book.**

The filename cannot be the ID — the same book is uploaded by different users under
different filenames.

**Flow:** on upload, compute the book ID → check whether that ID already exists → if it
exists, skip parsing and reuse the stored JSON → if not, run the parser and store the
result under the new ID.

> **Original proposal:** hash the text extracted from the first 10–15 pages.
>
> **Recommended revision:** hashing extracted *text* from the front matter is fragile —
> extractors, OCR, whitespace, and encoding differences can make the same book hash
> differently, and front matter (title/copyright/edition pages) is exactly where editions
> diverge. Prefer one of:
> - **SHA-256 of the raw PDF bytes** if "same book" means "same file." Trivial and exact;
>   will not dedupe a re-exported copy.
> - **SHA-256 of the normalized full extracted text** (lowercased, whitespace collapsed)
>   if "same book" means "same content."
>
> Start with raw-byte SHA-256; revisit fuzzy matching only if real duplicate-content
> misses are observed. `[OPEN]` — confirm which definition of "same" you want.

### 5.3 Concurrency for parsing `[DONE / off-branch]`

> Implemented on `handling-multi-user-uploud-pdf-end-point-feature`, **not yet merged into
> `develop` and not pushed to remote.**

**Problem:** parsing is heavy in time and resources. Running the pipeline inline inside the
endpoint handler **blocks** other upload requests — with 4–5 concurrent users, user 5
waits for users 1–4 to finish before starting. This serializes a CPU-bound workload.

**Solution:** treat each parse as a separate worker task to achieve real parallelism, but
**cap the number of concurrent parses** (unbounded processes would exhaust RAM/CPU). This
is the **worker-pool pattern**, implemented with **Celery + Redis**.

**Validation:** load test with 10 concurrent requests reduced average latency from
**44,853 ms → 660 ms (≈67× improvement)** with a **0% failure rate** after introducing
async queuing and a bounded worker pool.

---

## 6. Translation

### 6.1 Request → Search → Context → Translate `[DONE / develop, single-user]`

The user selects text on a page in the frontend (currently limited to one page at a time)
and submits it for translation.

**Current endpoint input:** the selected `text` + the `page_number`. (No book ID yet —
the pipeline was built and tested for a single book / single user.)

The handler then calls two functions in sequence: **Search**, then **Context handling**.

#### Search

Inputs: the user's selected `text`, the `page_number`, and a Python object representing the
book's JSON (paragraphs / sections / subsections).

Search uses a simple matching function to find **which paragraph(s)** the selection came
from on that page. Two cases:

- **Exact match** — the selection is one paragraph or part of one paragraph → Search
  returns a single paragraph.
- **Partial match** — the selection spans multiple paragraphs → Search returns more than
  one paragraph.

#### Context handling

Returns two things — the **context text** and the **target text** (the user's selection):

- **Exact match:** context text = the previous and next paragraphs.
- **Partial match:** target text = the full paragraphs the selection spans.

> **Concern raised:** Search and Context feel coupled.
>
> **Assessment (confirmed against the code):** the contract is *already* clean. Today
> `search_for_text` returns a structured dict `{page_index, para_indexs, match_type}` and
> `get_context` consumes exactly that object — Search *locates*, Context *assembles*. This
> is natural cohesion, not a flaw. Only two small fixes needed:
> - **Consistency:** `search_for_text` opens the JSON **from disk on every call**, while
>   `get_context` takes an already-loaded object. Make both take the loaded object (see
>   §6.5 — this *is* the "per-request JSON load" problem).
> - **Polish:** `match_type` could be a real enum and `para_indexs` a typed list, but this
>   is cosmetic; the design is sound. Remove the debug `print` statements in `sentence_match`.

### 6.2 Translation model fallback `[DONE / develop, partial — has bugs]`

The context text + target text are composed into a translation prompt and sent through a
**fallback chain**, implemented in `ContextualTranslator`:

1. **LLM providers**, in order: HuggingFace router (3 models) → Groq → Gemini (3 models) →
   GitHub Models (gpt-4o-mini). Each prompts for Arabic output wrapped in `<tr>...</tr>`.
2. **Plain machine-translation fallback:** Google Translate (free endpoint) → MyMemory.
   These translate the *context paragraph*, then `_robust_extract` recovers the target
   span via Jaccard sentence overlap.
3. **Local fallback:** an in-process `ContextAwareTranslator` model.

Output is post-processed by `_clean_output` (strips `<think>` blocks, extracts `<tr>`
content, removes diacritics and non-Arabic characters).

> **Known bugs (confirmed in code) — this will not work in production as-is:**
>
> 1. **Timeouts never trip the cooldown.** `_trigger_cooldown` fires only on transient HTTP
>    codes (429, 5xx). On an actual timeout/exception the `except` branch only logs and
>    returns `None` — so a hanging provider is retried on **every** request, each paying the
>    full 15 s `timeout` before falling through. This is the main latency leak.
> 2. **State is per-instance and recreated per request.** `translate_function` calls
>    `ContextualTranslator().translate(...)`, building a **new instance every call**, so
>    `_cooldowns` and `_cache` are discarded each time — the cache never hits and cooldowns
>    never persist. Across multiple Celery worker processes the in-memory state isn't shared
>    regardless.
> 3. **Cooldown granularity is wrong.** It is keyed per *provider* (`"hf"`), but
>    `_translate_hf` loops over 3 models; one transient failure cools down all of HF and
>    `_is_available("hf")` then short-circuits the remaining models.
> 4. **Dead code:** `_disabled_providers` is declared and checked but never populated.
> 5. **Blocking pacing:** `_enforce_pacing` uses `time.sleep`, which blocks; incompatible
>    with the async direction in §6.6.
>
> **Required design — a circuit breaker with shared state:** move the breaker state **and**
> the translation cache into **Redis**, shared across all workers, keyed **per provider (or
> per model)**. Trip the breaker on timeout/exception *and* transient status. Three states:
> - **closed** — healthy, route requests here.
> - **open** — recently failing, skip without attempting (no 15 s penalty).
> - **half-open** — cooldown elapsed, allow one trial; success → closed, failure → open.
>
> A reusable Redis-backed implementation is provided in `circuit_breaker.py`. See §6.2.1.

### 6.2.1 Mapping the breaker onto the existing code

- Replace `_is_available(provider)` → `breaker.allow(provider)`.
- Replace `_trigger_cooldown(provider)` → `breaker.record_failure(provider)`, and **call it
  in the `except` branch** of `_post` (the timeout path), not only on transient statuses.
- Add `breaker.record_success(provider)` after a 200 response.
- Make `ContextualTranslator` a **singleton / long-lived** object inside each worker (or
  hold the breaker + cache in Redis) so state survives across requests.
- Move `_cache` to Redis (`SETEX` with a TTL) so cache hits work across workers.

### 6.3 Choosing context scope `[OPEN]`

> **Question raised:** why not just use the whole page as context?
>
> **Assessment:** whole-page context is a fine cheap **baseline / fallback**, but it
> undercuts the project's thesis. The context a technical term needs often is *not* on the
> same page (a definition three chapters earlier, a glossary, notation set up in the
> intro). Cross-document context is precisely the differentiator. There is also a cost
> angle: more context = more tokens, more latency, and sometimes *worse* output, because
> irrelevant text dilutes the model's attention. Keep page-as-context as a baseline; do not
> let it crowd out the retrieval work that is the point of the project.

### 6.4 Advanced context extraction `[TODO / partial research done]`

Beyond the simple prev/next-paragraph context, the system will add more advanced context
extraction:

- terminology & definition extraction (critical for textbooks),
- summarization,
- agents that extract context from the book intelligently (must **not** naively scan the
  whole book).

This area needs dedicated study and design.

> **Decision:** the advanced extraction research (ATE) lives in a **separate repository**.
> Methods from research papers — and new ideas — will be prototyped there. It is out of
> scope for the core system repo. This is the right call: it protects the core system from
> research churn.

### 6.5 Passing the book ID into translation `[TODO]`

The translation request must include the **book ID** so the system knows which book to
search and extract context from. The book metadata record will store the **path to the
JSON**, so Search can load (deserialize JSON → Python object) the right book.

> **Concern raised:** deserializing the full JSON on **every** translation request is slow
> and wasteful. **Confirmed in code:** `search_for_text` does `open(book_Jason)` +
> `json.load` on every call. Options, in increasing order of effort:
> - **Per-worker LRU cache** keyed by `book_id` — cheap; each worker caches independently.
> - **Shared cache (Redis)** holding the pre-parsed structure in a fast format
>   (pickle/msgpack), so any worker reuses it.
> - **Stop loading the whole book.** Index paragraphs once at upload time into a queryable
>   store (Postgres or a search/vector index) keyed by `book_id` + page + paragraph; a
>   translation request then fetches only the few paragraphs it needs. This is also the
>   substrate the advanced extraction in §6.4 will need, so it is not throwaway work.
>
> The "keep a session/thread alive while the user works on a book" idea works but makes
> workers **stateful**, which complicates horizontal scaling (sticky routing). Prefer a
> shared cache or an index over session affinity. `[OPEN]`

### 6.6 Concurrency for translation `[TODO]`

The Search → Context → Translate pipeline is **not** yet wrapped in the worker pattern.
There may be thousands of translation requests per second, so concurrency handling is
required here too.

> **Important distinction — do not copy the parser's design verbatim:**
> - **Parsing is CPU-bound** → process-level parallelism with a bounded worker pool is
>   correct (this is why the §5.3 load test improved so much).
> - **Translation is I/O-bound** — almost all time is spent *waiting on the LLM API*. For
>   I/O-bound work, **asyncio** lets one process handle thousands of concurrent in-flight
>   requests cheaply; spawning processes wastes memory on workers that are merely sleeping
>   on network calls.
>
> At "thousands per second," the real bottleneck is the **LLM provider's rate limits**, not
> your workers. What translation needs is **async concurrency + rate limiting / backpressure
> toward the model**, not raw process parallelism. A task queue may still help here for
> retries and observability, but for different reasons than parsing. `[OPEN]` — confirm
> direction.

---

## 7. Proposed data model `[TODO]`

> Currently implicit. Making it explicit is a prerequisite for clean deployment.

- **Object storage** (S3 / MinIO) for large blobs: the original PDFs and the parsed JSON.
- **PostgreSQL** for relational metadata:
  - `users` — `user_id`, `username`, `password_hash`, …
  - `books` — `book_id` (content hash), `storage_path_pdf`, `storage_path_json`,
    `parse_status`, timestamps.
  - `model_state` — per-model circuit-breaker state, failure counts, last-failure time,
    cooldown (may live in Redis instead, for hot-path access shared across workers).

---

## 8. Concurrency model — summary

| Workload    | Nature    | Right tool                                   |
|-------------|-----------|----------------------------------------------|
| Parsing     | CPU-bound | Bounded process worker pool (Celery + Redis) |
| Translation | I/O-bound | Async concurrency + rate limiting/backpressure |

Match the concurrency model to the workload — processes for CPU work, async for I/O work.

---

## 9. Git repository — current state & cleanup

### 9.1 Current branches (remote)

- `feature/phase1-context-variants` — early term/definition extraction trial; forked from
  `main`, unrelated to current pipeline.
- `feature/phase2-bm25-lightweight` — same situation.
- `feature/phase2-lazy-embedding` — same situation.
- `api-calls-feature` — translation/upload API-call handling; **merged into `develop`**.
- `model` — model handling; **merged into `develop`**.
- `api_and_model` — API branch merged into model, then into `develop`.
- `Terminology-Memory-Extraction` — `develop` + regex-based term/definition extraction;
  no effect on the pipeline.
- `feature/retrieve_relevant_chunks` — simple RAG approach for book info; forked from
  `main`, unrelated to current pipeline.
- `parsing-feature` — full parser pipeline; **merged into `develop`**.
- `frontend-enhancement` — full frontend + API calls to backend; **merged into `develop`**.
- `feature/context-extraction` — same situation as `feature/retrieve_relevant_chunks`.
- `Bilingual-Sentence-Alignment` — same situation as `feature/context-extraction`.

### 9.2 Local-only / unpushed work

- `handling-multi-user-uploud-pdf-end-point-feature` — the Celery/Redis parsing work
  (§5.3). **Not merged into `develop`, not pushed.** This is valuable and exists only
  locally — it is the highest-priority git action.
- `Automatic_Glossary` — a non-regex term/definition extraction method with promising
  results. **Mistake:** the code was committed into `develop`'s `terminology/` folder
  instead of onto this branch.

### 9.3 Cleanup plan

**Step 0 — verify before deleting.** `git branch --merged develop` lists branches fully
contained in `develop`; anything there is safe to delete with zero loss.

**Bucket A — merged into `develop` → delete.**
`api-calls-feature`, `model`, `api_and_model`, `parsing-feature`, `frontend-enhancement`.
`git branch -d <name>` locally and `git push origin --delete <name>` on the remote.

**Bucket B — research spikes off `main`, unrelated → archive as tags, then delete.**
`feature/phase1-context-variants`, `feature/phase2-bm25-lightweight`,
`feature/phase2-lazy-embedding`, `feature/retrieve_relevant_chunks`,
`feature/context-extraction`, `Bilingual-Sentence-Alignment`, `Terminology-Memory-Extraction`.
For each: `git tag archive/<name> <name>` then delete the branch. Commits stay recoverable
via the tag; the branch list gets clean. Most of this likely belongs in the ATE repo.

**Bucket C — unmerged value → merge first, then delete.**
`handling-multi-user-uploud-pdf-end-point-feature`: merge into `develop` and **push** before
touching anything else, because it currently exists only on one machine.

**The `Automatic_Glossary` mistake.** If the glossary code is good (it is), the simplest fix
is to accept it where it landed (in `develop`) and delete/tag the `Automatic_Glossary`
branch. Only do cherry-pick surgery if its presence in `develop` actually causes a problem.
It may ultimately belong in the ATE repo.

### 9.4 Workflow going forward

The intended habit — *branch off `develop`, build one feature, merge when green, delete the
branch* — is the standard feature-branch workflow and a good default. Refinements:

- Keep branches **short-lived and single-purpose** (long-lived branches are what produced
  the current sprawl).
- Consider **squash-merge** so each feature lands as one clean commit.
- If this is solo / tiny-team, the `develop` layer may be unnecessary ceremony —
  trunk-based (`main` + short feature branches) is simpler. Keep `develop` only if a real
  staging line separate from `main` is needed. `[OPEN]`

---

## 10. Deployment-readiness gaps `[TODO]`

- **Explicit persistence layer** (§7) — object storage + Postgres, chosen deliberately.
- **Containerization** — Docker + a `docker-compose` for local dev (API + Redis + workers
  + Postgres). This is what makes "deploys cleanly" real, and it naturally expresses the
  CPU-vs-I/O worker split as separate services.
- **Observability** — structured logging + basic metrics (request latency, queue depth,
  circuit-breaker state, LLM error rate). Cannot debug thousands of req/s blind.
- **Auth hardening** — argon2/bcrypt hashing, vetted token/session library.

---

## 11. Task list

### Milestone 0 — protect existing work (do first)
- [ ] Merge `handling-multi-user-uploud-pdf-end-point-feature` into `develop`.
- [ ] Push `develop` to remote so the Celery/Redis parsing work is no longer local-only.

### Milestone 1 — git cleanup
- [ ] Run `git branch --merged develop` and confirm Bucket A.
- [ ] Delete Bucket A branches (local + remote).
- [ ] Tag-and-delete Bucket B research branches (`archive/<name>`).
- [ ] Decide `Automatic_Glossary`: accept-in-develop + delete branch, or move to ATE repo.
- [ ] Decide `develop`-vs-trunk-based workflow; document the chosen branching rule.

### Milestone 2 — persistence foundation
- [ ] Choose and stand up Postgres (`users`, `books`, optionally `model_state`).
- [ ] Choose and stand up object storage (S3 or MinIO) for PDFs + JSON.
- [ ] Write `docker-compose` for local dev (API, Redis, workers, Postgres, object store).

### Milestone 3 — book ID & dedup
- [ ] Decide the "same book" definition (raw-byte vs normalized-text hash). `[OPEN]`
- [ ] Implement book-ID computation.
- [ ] On upload: compute ID → look up in `books` → skip parse if present, else parse + store.

### Milestone 4 — translation correctness & robustness
- [ ] Make `book_id` a required field on the translation request.
- [ ] Refactor Search to return a structured match result (paragraph refs + `match_type`).
- [ ] Refactor Context to consume only the match result.
- [ ] Implement the per-model circuit breaker on Redis (closed / open / half-open).
- [ ] Replace the broken timeout-with-no-recovery logic with the circuit breaker.

### Milestone 5 — book loading performance
- [ ] Decide loading strategy: per-worker LRU cache vs shared Redis cache vs paragraph
      index. `[OPEN]`
- [ ] Implement the chosen strategy so full-JSON deserialization is not per-request.

### Milestone 6 — translation concurrency
- [ ] Implement translation as async concurrency (not a copy of the CPU worker pool).
- [ ] Add rate limiting / backpressure toward the LLM provider.
- [ ] (Optional) keep a task queue for retries/observability only.

### Milestone 7 — authentication
- [ ] `POST /signup` with validation + uniqueness check + password hashing.
- [ ] `POST /login` with credential verification.
- [ ] Session/token handling via a vetted library.

### Milestone 8 — operability
- [ ] Structured logging.
- [ ] Metrics: latency, queue depth, circuit-breaker state, LLM error rate.

---

## Appendix — resolved from provided code, and what's still open

The parser JSON schema, the Search/Context functions, and the translation logic have now
been reviewed. Resolved:

1. **JSON schema** — confirmed: `{"pages": [{"page": int, "paragraphs": [{"chapter",
   "section", "subsection", "type", "paragraph"}]}]}`. Search keys on `page` and indexes
   into `paragraphs`.
2. **Search/match function** — reviewed. Already returns a structured result; the only fix
   is to stop reading the file inside it (take the loaded object) and drop debug prints.
3. **Model fallback** — reviewed; bugs catalogued in §6.2. Providers/models are now known:
   HF (DeepSeek-R1, Llama-3.3-70B, Qwen2.5-7B) → Groq (llama-3.1-8b-instant) → Gemini
   (2.5-flash, 2.0-flash, flash-latest) → GitHub (gpt-4o-mini) → Google/MyMemory → local.
   Per-request timeout is 15 s; cooldown 65 s; min interval 0.5 s.

Still needed to finish the design:

4. **Stack — confirmed.** Frontend: React 19 + TypeScript (Vite), Tailwind, pdfjs/react-pdf.
   Backend: FastAPI + Uvicorn, PyMuPDF (parsing), PyTorch / Transformers /
   sentence-transformers / spaCy / NLTK, requests. **No database declared yet** — this is a
   gap to close (see §7 and `INFRASTRUCTURE_AND_SCALING.md`).
5. **Scale target — clarified.** No fixed req/s number; the goal is a design where *scale is
   a cost problem, not an engineering problem* (a stateless, horizontally scalable app
   tier). This sets the direction more than a number would — see
   `INFRASTRUCTURE_AND_SCALING.md` §1–§4.
6. **Team — not solo.** Most of the system is built by the author except the model/translation
   layer (which is also the buggiest part — §6.2). Because it's a team, **keep `develop`**
   and the feature-branch workflow (§9.4); the `develop`-vs-trunk question is resolved toward
   keeping `develop`.
7. **"Same book" definition.** Still open — decides Milestone 3.

See the companion **`INFRASTRUCTURE_AND_SCALING.md`** for the state model (per-request vs
per-process vs shared), the statelessness principle behind the scale target, the target
five-piece architecture, and the database operational knowledge / learning roadmap.
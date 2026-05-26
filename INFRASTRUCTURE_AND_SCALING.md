# Infrastructure & Scaling — Concepts and Learning Roadmap

> Companion to `DESIGN.md`. That document describes *what the system is*. This one
> describes *how to make it scale cleanly* and *what operational knowledge is needed to
> get there*. No code — this is the reference for understanding and decisions; an
> implementation tool can produce the code once the decisions here are made.

## 1. North star

**Goal:** reach a design where handling more load means adding more identical machines and
paying more money — **not** redesigning. In other words: *scale becomes a cost problem, not
an engineering problem.*

The technical name for this is a **horizontally scalable, stateless application tier**, and
it follows from one rule:

> **Application processes must hold no state that a request depends on.**

When that holds, any worker can serve any request, so scaling is "run more copies behind a
load balancer." When it breaks — when a request must reach a *specific* worker because that
worker holds the user's session or a cached book — scaling becomes engineering again
(sticky routing, cache coordination, etc.).

## 2. The three kinds of state

Most scaling mistakes are putting state in the wrong category.

| Kind            | Lifetime                    | Lives where                          | Examples in this system                                  |
|-----------------|-----------------------------|--------------------------------------|----------------------------------------------------------|
| **Per-request** | one request                 | local variables                      | the user's selected text, page number                    |
| **Per-process** | worker lifetime, reused     | worker memory (built once at start)  | loaded PyTorch/Transformers model, HTTP session, DB/Redis connection pool |
| **Shared**      | system lifetime, all workers| external service                     | cache, circuit-breaker state, user sessions, parsed-book cache, the queue |

**Clarifying the "singleton" point (the per-process column):**
- The bug today is `ContextualTranslator()` being constructed **per request**, which
  rebuilds expensive per-process objects every time (worst case, reloading an ML model).
- A "per-process singleton" means: build those objects **once when the worker starts**, and
  reuse them across every request that worker handles. One per worker — not one in the whole
  system.

**Clarifying Redis (the shared column):**
- Redis is **not** an object in your app's memory. It is a **separate server process** your
  app connects to over a socket — like a database.
- Because it's external, **all** workers connect to the **same** Redis. That is what makes
  cache / breaker state shared and persistent across requests and across processes.
- The cache therefore needs **no** Python singleton: any worker reads/writes the same Redis.
  The per-process singleton only matters for the **connection pool** to Redis and for the
  loaded models.

## 3. What breaks statelessness here (avoid)

- **Per-worker in-memory cache of a book** that requests rely on → pins work to a worker.
  *Scalable version:* cache the parsed book in **Redis** so every worker benefits.
- **Keeping a session/thread alive per book on a worker** (an idea raised in the notes) →
  requires sticky routing. Prefer shared cache or a paragraph index.
- **In-memory translation cache / cooldowns** (current code) → lost per request and not
  shared across workers. Move to Redis.

## 4. The target architecture (five standard pieces)

1. **Stateless FastAPI workers** — interchangeable, disposable, behind a load balancer.
2. **PostgreSQL** — relational metadata (users, books, model/breaker state if not in Redis).
3. **Redis** — cache, circuit-breaker state, and the Celery task queue.
4. **Object storage (S3 / MinIO)** — the large blobs: original PDFs and parsed JSON.
   Databases are bad at large blobs; keep them out of Postgres.
5. **Celery workers** — heavy CPU jobs (parsing). Translation is I/O-bound → async +
   rate limiting (see `DESIGN.md` §6.6), not the same process pool.

Queues decouple tiers: the API (producer) and workers (consumers) scale independently and
the queue absorbs bursts. The eventual bottlenecks are the **database** and the **LLM
provider rate limits** — design so those can be scaled later without a rewrite; do **not**
pre-build them now.

## 5. Database operational knowledge (the identified gap)

Understanding schema/queries is not the same as running a database as infrastructure.
For each item below: **understand the concept → make a decision → then implement.**

1. **Managed vs self-hosted.** Self-hosting means owning backups, patching, failover, and
   replication. Managed services (RDS, Cloud SQL, Neon, Supabase) do that for you.
   *Decision:* default to **managed** early; revisit only if cost forces it.

2. **Connection pooling (the classic scaling trap).** DB connections are expensive and
   capped (Postgres ~100 by default). `50 processes × 20 connections = 1000` → the DB
   falls over. Each process reuses a small **pool** (a per-process singleton); with many
   processes, add an external pooler (**PgBouncer**) in front. *Budget connection count
   deliberately; it must not grow freely with traffic.*

3. **Migrations (toy → real project).** Never hand-edit a production schema. Schema changes
   are **versioned migration files** (Alembic), in git, applied identically across
   dev/staging/prod. Highest-leverage habit to adopt from this project.

4. **Secrets & network isolation.** Credentials live in env vars / a secrets manager, never
   in code or git. The DB is **not** on the public internet — private network, app-only
   access. App DB user gets **least privilege**. These three habits are most of practical
   DB security.

5. **Environment separation.** Separate DB + credentials for dev / staging / prod. Never
   test against production data.

6. **Scaling reads first.** Reads dominate here (same books translated repeatedly). Add
   **read replicas** (reads to replicas, writes to primary) and cache hot reads in Redis.

7. **Scaling writes (later, hard).** Partitioning/sharding exists but is far off. Know it;
   don't build it. Pre-sharding an unloaded DB is premature engineering.

8. **Tested backups.** Automated backups **plus a restore you've actually verified**. An
   untested backup is not a backup.

## 6. Tasks (learn → decide → implement; no code here)

### Learn (build the tacit knowledge before delegating implementation)
- [ ] Connection pooling: why it exists, how pool size relates to DB connection limits, what
      PgBouncer adds when process count is high.
- [ ] Migrations with Alembic: the versioned-schema workflow across environments.
- [ ] Secrets management + private networking for a database.
- [ ] Read replicas and the read/write split; what is safe to serve from a replica.
- [ ] Object storage basics (S3/MinIO): why blobs go here, not in Postgres.

### Decide
- [ ] Managed vs self-hosted Postgres (recommend: managed).
- [ ] Object storage: cloud S3 vs self-hosted MinIO.
- [ ] "Same book" definition for the dedup hash (raw-byte vs normalized-text) —
      see `DESIGN.md` Milestone 3.
- [ ] Cache location for the parsed book: Redis shared cache vs paragraph index in Postgres
      (recommend: shared, to preserve statelessness) — see `DESIGN.md` §6.5.
- [ ] Whether "thousands of translation req/s" is a real near-term target (drives how much
      async/rate-limit work to do now vs later).

### Implement (the scalable foundation)
- [ ] Make FastAPI workers stateless: no per-request state a future request depends on.
- [ ] Make `ContextualTranslator`, the HTTP session, the model load, and the Redis/DB pools
      **per-process singletons** (built once at worker startup, reused).
- [ ] Move the translation cache and circuit-breaker state into **Redis** (shared).
- [ ] Stand up Postgres (managed) with a connection pool; wire Alembic migrations.
- [ ] Stand up object storage; store PDFs + parsed JSON there, paths in Postgres.
- [ ] `docker-compose` for local dev: API + Redis + workers + Postgres + object store.
- [ ] Basic observability: request latency, queue depth, breaker state, DB connection count,
      LLM error rate.

## 7. Guardrail against the opposite mistake

With strong fundamentals, the risk is **over-engineering** — building replication and
sharding before there are users. The north star in §1 is the guard: build the *property*
(statelessness) cheaply now; defer the *expensive parts* (replicas, sharding, PgBouncer at
scale) until cost actually forces them. "Designed so it *can* scale" beats "scaled
prematurely."
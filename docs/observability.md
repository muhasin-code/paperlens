# Langfuse Observability

This document describes the observability added in Milestone 4.1: the self-hosted Langfuse stack, the Python client wrapper, what a real PaperLens request records, and the in-process metrics that remain available when Langfuse is off.

It describes the code as it runs. Where an instrumentation point exists on a class but is not on the default request path, that is called out. The milestone plan's trace tree (`retrieval.semantic`, `retrieval.bm25`, `reranker`, `generation` as children of every `rag.query`) is the target shape, not the tree produced by `POST /query` today.

---

## What this milestone records

PaperLens already returns latency fields on `QueryResponse` (`retrieval_time_ms`, `generation_time_ms`, `total_time_ms`). Those numbers die with the HTTP response. Milestone 4.1 adds two stores:

1. **Langfuse traces**, for one request at a time. A trace is the thing you open when a single answer is wrong, slow, or refused.
2. **An in-process `MetricsCollector`**, for aggregates across the life of the API process: P50/P95 latency per component, a confidence histogram, citation coverage, and a token counter. Read it at `GET /observability/metrics`. It is not durable. A process restart clears it.

Langfuse is self-hosted. Nothing is sent to Langfuse Cloud unless `LANGFUSE_HOST` is changed to a cloud URL.

The pipeline does not wait on Langfuse to answer the user. Export is asynchronous. A down Langfuse server, a missing key, or `LANGFUSE_ENABLED=false` must not fail `/query`.

---

## Request path, and where a trace is born

```
POST /query
  │
  ├─ mode = "structured"
  │     QueryRouter.classify()          (no span; in-process only)
  │     ExtractionInference.extract()   → observation "extraction"
  │
  └─ mode = "conversational" or "auto"  (the default)
        RAGService.query()
          trace "rag.query"                          ← root observation
            HybridRetriever.search()
              span "retrieval.hybrid"                ← child, metadata only
            LLM provider generate()                  ← not its own observation
          scores: latency_p50, citation_coverage, confidence
          flush()
```

`RouterService.route_query()` and `CrossEncoderReranker.rerank()` are instrumented, but the live route in `src/paperlens/api/routes.py` does not call them.

- `routes.py` branches on `request.mode` itself. `RouterService` is never imported by the route module, so a normal `/query` does not emit `router.decision`.
- The conversational path uses hybrid RRF only. The cross-encoder was left out of `RAGService.query()` after the Milestone 2.5 benchmark. A `reranker` span appears only if some other caller invokes `CrossEncoderReranker.rerank()`.

`POST /query/stream` updates the in-process retrieval latency and does not open a Langfuse trace.

---

## Self-hosted stack

Langfuse v4 does not store traces inside the web process. The web container accepts a batch, writes the payload to object storage, and enqueues a job. The worker reads that job and writes rows into ClickHouse. The UI reads ClickHouse. If the worker is not running, the SDK can report success and the UI still shows no data. This was the failure mode during Milestone 4.1 end-to-end testing.

```
PaperLens (uvicorn on the host)
    │  OTLP / ingestion  (HTTP, batched, then flush)
    ▼
langfuse-web :3000
    │  event blob                 │  job id
    ▼                             ▼
minio (bucket "langfuse")       redis (BullMQ)
    │                             │
    └──────────► langfuse-worker ─┘
                      │
                      ▼
                  clickhouse   ◄──── langfuse-web (UI queries)
                  postgres     (users, projects, API keys)
```

### Services

Defined in `docker-compose.yml`. Images are the v4 line (`docker.langfuse.com/langfuse/langfuse:4` and `langfuse-worker:4`).

| Service | Image | Published port | Memory cap | Role |
|---|---|---|---|---|
| `langfuse-web` | `langfuse/langfuse:4` | `3000` | 1536 MB | UI and ingestion API |
| `langfuse-worker` | `langfuse/langfuse-worker:4` | none (3030 stays inside the network) | 512 MB | Queue consumer. Required for traces to appear. |
| `postgres` | `postgres:17` | none | 512 MB | Users, projects, API keys |
| `clickhouse` | `clickhouse/clickhouse-server:25.12` | none | 1024 MB | Trace and observation storage |
| `redis` | `redis:7-alpine` | none | 256 MB | BullMQ queue. `maxmemory 200mb`, policy `allkeys-lru`. |
| `minio` | `cgr.dev/chainguard/minio` | `9090` → container `9000` | 256 MB | S3-compatible store for event payloads |

With the worker running, the cap sum is about 4 GB. The development machine this was sized for has about 8 GB. The embedding model (`BAAI/bge-large-en-v1.5` on CPU) plus the API process sits in what remains. ClickHouse also has an internal cap, `CLICKHOUSE_MAX_MEMORY_USAGE=700000000`.

Postgres, ClickHouse, and Redis are not published to the host. Only the UI (`localhost:3000`) and MinIO (`localhost:9090`) are.

The MinIO bucket is the directory created before the server starts (`mkdir -p /data/langfuse`). There is no separate bucket-init container. If that directory is missing, event upload fails and the worker has nothing to ingest.

### Starting and stopping

```bash
make langfuse-up      # web, worker, postgres, clickhouse, redis, minio
make langfuse-logs    # follow web and worker
make langfuse-down    # docker compose down (named volumes are kept)
```

`langfuse-up` must start `langfuse-worker`. A compose file that hides the worker behind a profile, or a Make target that only starts `langfuse-web`, reproduces the empty UI.

First boot of an empty Postgres volume can create an organization from the `LANGFUSE_INIT_*` variables in `docker-compose.yml`:

| Variable | Default in compose |
|---|---|
| `LANGFUSE_INIT_ORG_NAME` | `paperlens` |
| `LANGFUSE_INIT_PROJECT_NAME` | `paperlens` |
| `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` | `${LANGFUSE_PUBLIC_KEY}` from `.env` |
| `LANGFUSE_INIT_PROJECT_SECRET_KEY` | `${LANGFUSE_SECRET_KEY}` from `.env` |
| `LANGFUSE_INIT_USER_EMAIL` | `admin@paperlens.local` |
| `LANGFUSE_INIT_USER_PASSWORD` | `paperlens` |

Those init variables run only when the database has no organization yet. Creating the project by hand in the UI does not rewrite keys that were later pasted into `.env`, and a later change to `.env` does not rotate keys already stored in Postgres. The public key shown under Project Settings → API Keys must be the `LANGFUSE_PUBLIC_KEY` the API process is using.

UI: `http://localhost:3000`.

Health, from the host:

```bash
curl -sf http://localhost:3000/api/public/health
```

The container healthcheck hits the same path from inside the web container.

---

## Configuration

`src/paperlens/settings.py` loads these from `.env`. The SDK is not given the process environment by pydantic-settings. The wrapper passes the values into the `Langfuse()` constructor itself, so a key that exists only in `.env` is still visible to the client.

| Setting | Environment variable | Default | Meaning |
|---|---|---|---|
| `langfuse_enabled` | `LANGFUSE_ENABLED` | `true` | `false` skips client construction. Requests still run. |
| `langfuse_public_key` | `LANGFUSE_PUBLIC_KEY` | empty | Project public key (`pk-lf-...`). Empty disables tracing. |
| `langfuse_secret_key` | `LANGFUSE_SECRET_KEY` | empty | Project secret key (`sk-lf-...`). Empty disables tracing. |
| `langfuse_host` | `LANGFUSE_HOST` | `http://localhost:3000` | Passed to the SDK as `base_url`. |

`host=` on the SDK constructor is a deprecated alias of `base_url`. The wrapper uses `base_url`.

After construction the wrapper calls `auth_check()`. Failure is logged at ERROR and does **not** turn the client into a no-op and does **not** abort API startup. A Langfuse that is still migrating, or a key that is wrong, should be obvious in the log without taking the research API down. A wrong key means traces will not show up until the key is fixed and the API is restarted.

Expected startup lines when the key is accepted:

```text
INFO:paperlens.observability:Langfuse auth_check ok (host=http://localhost:3000)
INFO:paperlens.observability:Langfuse client initialized (host=http://localhost:3000)
INFO:paperlens:Langfuse tracing enabled (host=http://localhost:3000)
```

`auth_check failed` means the host answered and rejected the key. Fix the key pair. Do not look at the worker yet.

Disable tracing without stopping Docker:

```bash
# .env
LANGFUSE_ENABLED=false
```

Restart uvicorn. pydantic-settings reads `.env` at process start.

---

## Client wrapper

`src/paperlens/observability/langfuse_client.py` is the only module that imports `langfuse`. The rest of PaperLens talks to `LangfuseClient`.

One client per process, built on first use:

```python
from src.paperlens.observability.langfuse_client import get_langfuse_client

client = get_langfuse_client()
```

`main.py` builds it during FastAPI lifespan and stores it on `app.state.observer`. `RAGService` receives that instance from `get_rag_service()`. On shutdown the lifespan calls `observer.shutdown()` (flush, then SDK shutdown) and `reset_langfuse_client()`.

### Degradation rules

| Condition | Behaviour |
|---|---|
| `langfuse_enabled` is false | `_init_client` returns `None`. `enabled` is false. |
| Public or secret key is empty | Same. A warning is logged. The SDK is not constructed. |
| `import langfuse` fails | Warning, `enabled` is false. |
| Constructor or `auth_check` raises | Constructor failure disables tracing. `auth_check` failure does not. |
| `trace` / `span` / `generation` / `score` / `flush` while disabled | No-op. Context managers yield a `_NoOpSpan` with `update()` and `end()` that do nothing. |

`NoOpClient` is the public stand-in used by tests. It has the same four operations and `enabled = False`.

### Operations

| Method | SDK call | What it is |
|---|---|---|
| `trace(name, ...)` | `start_as_current_observation(as_type="span")` with a new `TraceContext` | Root observation. `user_id`, `session_id`, and `tags` are copied into metadata. The SDK does not take them as top-level arguments. |
| `span(name, ...)` | `start_as_current_observation(as_type="span")` | Child of the current observation, if one is active in this task. |
| `generation(name, model, ...)` | `start_as_current_observation(as_type="generation")` | Child marked as a model call. Carries `model` and `usage_details`. |
| `score(name, value, ...)` | `create_score` | Numeric or categorical score on the current trace. |
| `latency_score(ms, p)` | `score(name=f"latency_{p}", ...)` | Convenience. Callers pass `"p50"`. See the naming warning below. |
| `flush()` | SDK `flush()` | Blocks until the current batch is handed to the API. Failures log at WARNING. |
| `shutdown()` | `flush()` then SDK `shutdown()` | Lifespan exit only. |

`RAGService.query()` calls `flush()` in a `finally` block, after the `with trace` block has closed the root observation. Flushing inside the `with` would export a span that is still open. The SDK also flushes on a timer (default interval 5 seconds, or 512 spans). The explicit flush is what makes a single `/query` show up without waiting, and without depending on the background thread surviving a reload.

Ingestion is still asynchronous on the server side. `flush()` means the web container accepted the batch. The worker still has to write ClickHouse. The UI often lags the HTTP response by 15–30 seconds, longer if the worker is busy or the machine is swapping.

### Context propagation

The SDK uses OpenTelemetry contextvars. A `span()` or `generation()` opened while a `trace()` is active on the same asyncio task becomes a child. That is why `HybridRetriever.search()` can call `observer.span("retrieval.hybrid")` with no trace id of its own and still land under `rag.query`: `search()` runs inside the `with observer.trace(...)` block, before any `await` that leaves the task.

Context does not cross `asyncio.gather`, `asyncio.create_task`, or a thread pool unless it is copied explicitly. PaperLens does not do that copy. A span opened on a worker thread will start its own trace.

`uvicorn --reload` runs the app in a child process. The client is constructed in that child's lifespan, which is the correct place. Constructing it in the reloader parent and inheriting it across the fork is not.

---

## Trace anatomy

Names below are the `name=` strings passed to the wrapper. In the Langfuse v4 UI the traces table is an observations table filtered to root observations. Clear that filter only when you want every child row on its own.

### Conversational `POST /query`

Root observation `rag.query`.

| Field | Value |
|---|---|
| Input | The user query string |
| Metadata | `query_length` |
| Output | Set at the end of every return path: `answer`, `mode`, `confidence`, `citation_count`. Empty retrieval and refusal set this too. Before that fix the output stayed null because it was only passed at span start. |

Child, when retrieval returns:

`retrieval.hybrid` — a span opened **after** semantic search, BM25, and RRF have finished. The Langfuse duration of this observation is the time spent returning from the `with` block, which is close to zero. The real times are metadata, measured with `time.perf_counter` around each stage:

| Metadata key | Meaning |
|---|---|
| `retrieval.semantic_ms` | Dense search, including query embedding |
| `retrieval.bm25_ms` | BM25 search |
| `retrieval.rrf_ms` | Sort and cut of the fused list |
| `retrieval.total_ms` | Wall time of `search()` |
| `hybrid.semantic.results` | Candidates from Chroma |
| `hybrid.bm25.results` | Candidates from BM25 |
| `hybrid.rrf.fused_count` | Chunks returned to the RAG service |
| `rrf_k` | RRF constant used for this call |

There is no separate `retrieval.semantic` or `retrieval.bm25` observation. Both latencies are fields on this one span.

The LLM call is not a `generation` observation. Model name, finish reason, and a truncated raw completion are written to the API log (`paperlens.llm` and `paperlens.api`), not to Langfuse. Token totals in the metrics collector are `len(text.split())` estimates, not `usage` from Groq.

### Scores on `rag.query`

Written at the end of the trace, including the empty-retrieval and exact-refusal paths.

| Score name | Value | What it actually is |
|---|---|---|
| `latency_p50` | `total_time_ms` of **this** request | Not a percentile. The name is historical. The P50 across requests is on `GET /observability/metrics`, under `latency_p50_ms["rag.total"]`. |
| `citation_coverage` | `1.0` if the response has at least one citation, else `0.0` | Per trace, not the running ratio. The running ratio is `citation_coverage` on the metrics snapshot. |
| `confidence` | `0.0` on refusal or empty retrieval, otherwise the mean of the top three retrieval scores clamped to `[0, 1]` | A retrieval heuristic. It is not a calibrated probability and not a Ragas score. |

An exact match on the prompt's `refusal_message` (`I cannot answer from the provided sources.`) is a refusal: citations empty, both scores zero. A substring match is not. An answer that merely discusses refusal is returned as an answer.

### Structured `POST /query` (`mode=structured`)

No `rag.query` root. `ExtractionInference.extract()` emits a root `generation` named `extraction` (nothing else is active, so the generation is the root).

| Field | Value |
|---|---|
| Model | `settings.extraction_adapter_repo` |
| Input | The query text (this route passes the query, not a retrieved chunk) |
| Output | `ExtractionOutput` as a dict |
| Metadata | `extraction.mode` (`hf_api`, `local_placeholder`, or `kaggle`), `extraction.latency_ms` |

`track_latency("extraction", ...)` and `track_tokens(...)` run whether or not Langfuse is enabled. The same generation is opened again, with near-zero duration, by `RouterService._handle_structured` if that service is used. On the live route it is not, so you get one `extraction` observation, not two.

### Instrumented, but not on the default route

**`router.decision`** (`RouterService.route_query`)

Metadata: `router.mode`, `router.confidence`, `router.matched_keywords` (first 5 only), `router.total_keywords`. Score `router_mode` is `1.0` for structured and `0.0` for conversational, with `data_type=CATEGORICAL`. The span is opened after `classify()` returns, so its duration is the handler time, not the classifier time. Classifier time is `track_latency("router.decision_time", ...)`.

**`reranker`** (`CrossEncoderReranker.rerank`)

Metadata: `reranker.input_count`, `reranker.output_count`, `reranker.latency_ms`, `reranker.model`. Also `track_latency("reranker", ...)`. Not called from `RAGService.query()`.

### Decorators

`src/paperlens/observability/instrumentation.py` defines `@instrument_retrieval`, `@instrument_reranker`, `@instrument_generation`, and `@instrument_extraction`. The production call sites do not use them. They open the span after the wrapped function returns, same pattern as the manual instrumentation, and they are covered by `tests/test_observability/test_instrumentation.py` as callable wrappers. Do not assume a decorator is in the request path because the helper exists.

---

## In-process metrics

`src/paperlens/observability/metrics.py`. One `MetricsCollector` per process (`get_metrics()`). A lock guards the deques. The buffers are bounded so a long-running API cannot grow them without limit.

| Method | Stores |
|---|---|
| `track_latency(component, ms)` | Up to 1000 samples per component name (`deque(maxlen=1000)`). |
| `track_confidence(score)` | Histogram. Bucket key is `round(score, 2)`. `None` is ignored. |
| `track_citation(has_citation)` | Increments `query_count`. Increments the cited counter when `has_citation` is true. |
| `track_tokens(model, n)` | Running sum per model string. `n` is a word-count estimate, not a tokenizer count. |

Components written by the current call sites:

| Component | Written from |
|---|---|
| `retrieval` | `RAGService.query` and `query_stream` |
| `generation` | `RAGService.query` only (not the stream path) |
| `rag.total` | `RAGService.query` only |
| `extraction` | `ExtractionInference.extract` |
| `reranker` | `CrossEncoderReranker.rerank` |
| `router.decision_time` | `RouterService.route_query` |

P50 is `sorted_vals[n // 2]` (the high median). P95 is `sorted_vals[int(n * 0.95)]` once the component has at least 20 samples; below that, P95 is the max. Both are rounded to two decimal places.

Citation coverage on the snapshot is `cited_queries / total_queries` over the whole process, not over the latency window. A refusal and an empty retrieval both call `track_citation(False)` and therefore lower the ratio. An exception does too.

### `GET /observability/metrics`

No authentication. It is a development endpoint. Do not expose it on a shared network without a gate.

```bash
curl -s http://localhost:8000/observability/metrics | jq .
```

The body is the `Snapshot` fields:

```json
{
  "latency_p50_ms": {"retrieval": 240.12, "generation": 1600.55, "rag.total": 1900.01},
  "latency_p95_ms": {"retrieval": 9800.0, "generation": 3200.0, "rag.total": 13000.0},
  "latency_samples": 12,
  "confidence_histogram": {"0.0": 2, "0.42": 1},
  "citation_coverage": 0.3333,
  "token_usage": {"openai/gpt-oss-120b": 840},
  "query_count": 3
}
```

`Snapshot` is a dataclass. The handler must serialize it with `dataclasses.asdict` (or an equivalent). Calling `.model_dump()` on it raises `AttributeError` and the route returns 500. If that is what you see, that is the cause. An empty process returns zeros and empty maps, which is also what you see with `LANGFUSE_ENABLED=false` after a restart: the collector does not care whether Langfuse is on, but it only fills up as requests arrive.

`uvicorn --reload` starts a new process on code changes. The snapshot goes back to zero. Langfuse does not.

---

## Verifying a trace end to end

1. `make langfuse-up`. `docker compose ps` must list `langfuse-worker` as running, not as absent.
2. `curl -sf http://localhost:3000/api/public/health`.
3. Start the API (`make run-backend`). Confirm `Langfuse auth_check ok` in the log. `initialized` without `auth_check ok` is not enough.
4. Send one query:

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What learning rate schedule is proposed for transformers?", "top_k": 5}'
```

5. Wait about 20 seconds. In the UI, open the project whose API keys match `.env`. Set the time range to the last hour. Filter to root observations if the table is showing every span.

You should see one root named `rag.query`, with a child `retrieval.hybrid` when retrieval returned chunks. The output block should contain the answer text, not null. Scores should be attached to that trace.

The API log should also contain:

```text
Retrieved N chunks: ['<chunk_id> | <title>', ...]
LLM raw response (<n> chars): ...
Groq finish_reason=stop content_chars=<n> model=openai/gpt-oss-120b
```

Those lines are the way to separate "the model refused" from "the validator rewrote the answer" from "retrieval returned the wrong papers". Langfuse does not currently store the raw completion or the chunk titles. The log does.

---

## Debugging from a trace

Use this order. Skipping to the prompt is how a retrieval bug gets misread as a model bug.

1. **No row in the UI.** Check `auth_check` in the API log, then `docker compose ps` for the worker, then `make langfuse-logs`. A 401 on `/api/public/ingestion` or `/api/public/otel` is a key mismatch. A 200 or 207 from the web container with an idle worker is the queue-not-consumed failure. Confirm you are in the project that owns the key, and that the time filter includes "now".
2. **`rag.query` exists, `retrieval.hybrid` does not.** Retrieval returned no chunks. The service returns the refusal without calling `search()`'s span (the span is inside `search()`, and `search()` returned an empty list before the span, or was not entered). Check Chroma and the BM25 index, not the prompt.
3. **`retrieval.hybrid` exists, but the titles in the API log are off topic.** The refusal or the weak answer is a retrieval result. The metadata counts (`hybrid.semantic.results`, `hybrid.bm25.results`) tell you which index was empty. A large `retrieval.semantic_ms` on the first query after startup is the embedding model loading a batch, not a Langfuse stall.
4. **Chunks look relevant, output is the refusal sentence.** The model followed the prompt. Compare the Langfuse output with the `LLM raw response` log line. If they match, the model really emitted the refusal. If the log has a real answer and the API returned the refusal, validation rewrote it (see `docs/citation-enforcement.md`).
5. **Output is a real answer, `citation_coverage` is 0.** The model text had no `[chunk_id]` markers, so the response was still returned (after retries) but the citation list may be the retrieved set rather than markers parsed from the text. Read `citation_count` on the trace output.
6. **Latency looks huge on the span and small in metadata, or the reverse.** Trust metadata (`retrieval.total_ms`) and `QueryResponse` for retrieval. Trust `total_time_ms` on the response, and the `latency_p50` score, for end-to-end. Do not trust the Langfuse duration of `retrieval.hybrid`.

`generation_time_ms > 0` together with the refusal string means the model was called. It does not mean retrieval failed. `generation_time_ms == 0` and empty citations means retrieval returned nothing and the model was not called.

---

## Dashboard views

Langfuse does not ship a PaperLens dashboard. The scores above are enough to build three views. In the UI these are custom score widgets or filters on the observations table, not a separate metrics product.

### Latency

- Score `latency_p50`, numeric, unit milliseconds. Plot the mean or a percentile **of this score** over time. Remember each point is one request's total time. The name is not already a P50.
- For a real P50/P95 of the process, use `GET /observability/metrics` (`latency_p50_ms`, `latency_p95_ms`) while the API process is still up. That view will not survive a restart and is not in Langfuse.
- Break down one slow trace with `retrieval.semantic_ms` and `retrieval.bm25_ms` on the child span. Semantic time dominates on a cold embedding cache. Generation time is not a Langfuse field; it is `generation_time_ms` on the HTTP response and `latency_p50_ms["generation"]` in the snapshot.

### Citation coverage

- Score `citation_coverage` on `rag.query`. The mean of this score over a window is the fraction of traced queries that returned at least one citation.
- The same definition, for the current process only, is `citation_coverage` on `/observability/metrics`.
- Refusals and empty retrievals are zeros. They belong in the denominator. Filtering them out will report 100% coverage on the queries that happened to cite.

### Confidence

- Score `confidence`, numeric, range 0 to 1.
- The in-process histogram (`confidence_histogram`) is bucketed to 0.01. It is the right view for "are scores piled on 0 because everything is refusing?" A Langfuse histogram of the score answers the same question across restarts.

### Router, when `RouterService` is actually called

- Score `router_mode`: categorical, `1` structured, `0` conversational. The mean is the fraction of routed queries that took the extraction path.
- This score is absent on the default `/query` path.

---

## Local development workflow

```bash
make langfuse-up
make run-backend
make test-observability    # tests/test_observability only
make langfuse-logs
make langfuse-down
```

`make test-observability` does not need Docker. The tests construct a disabled client or a no-op and never call the network:

| File | Covers |
|---|---|
| `tests/test_observability/test_langfuse_client.py` | `NoOpClient`; disabled flag; missing keys; `score()` does not raise |
| `tests/test_observability/test_instrumentation.py` | Decorators and context managers are callable and preserve the wrapped name |
| `tests/test_observability/test_metrics.py` | P50/P95, histogram buckets, citation ratio |

They do not cover `auth_check`, the Docker stack, or a real export. Those are the manual steps in "Verifying a trace end to end".

`make test` remains the full suite. Observability must not change `QueryResponse` for the existing API tests: a disabled client returns the same answer, citations, and timings as no instrumentation at all.

---

## Source files

| File | Role |
|---|---|
| `docker-compose.yml` | Langfuse v4 stack and memory caps |
| `Makefile` | `langfuse-up`, `langfuse-down`, `langfuse-logs`, `test-observability` |
| `src/paperlens/settings.py` | `langfuse_enabled`, keys, host |
| `.env.example` | Placeholders. Keys stay empty in the example. |
| `src/paperlens/observability/langfuse_client.py` | SDK wrapper, no-op, `auth_check`, flush |
| `src/paperlens/observability/instrumentation.py` | Decorators and `PipelineObserver`. Not used by the live route. |
| `src/paperlens/observability/metrics.py` | Process-local aggregates |
| `src/paperlens/observability/__init__.py` | Re-exports |
| `src/paperlens/main.py` | Construct the client at startup, `shutdown()` on exit |
| `src/paperlens/api/routes.py` | Passes `app.state.observer` into `RAGService`. `GET /observability/metrics`. |
| `src/paperlens/api/rag.py` | Root trace `rag.query`, scores, `flush()` |
| `src/paperlens/retrieval/hybrid.py` | Child span `retrieval.hybrid` |
| `src/paperlens/retrieval/reranker.py` | Span `reranker` (not on the default path) |
| `src/paperlens/extraction/inference.py` | Generation `extraction` |
| `src/paperlens/api/router_service.py` | Span `router.decision` (not on the default path) |
| `requirements.txt` | `langfuse>=2.25.0`. The installed distribution is the OpenTelemetry SDK (v3/v4 API: `start_as_current_observation`, `TraceContext`). Do not downgrade to the v2 `trace()` / `span()` client. The wrapper will not run against it. |

---

## Troubleshooting

| What you see | Cause | What to do |
|---|---|---|
| UI has no traces, API log says `Langfuse client initialized` | Web accepted nothing yet, or the worker never started. `initialized` only means the constructor returned. | Confirm `auth_check ok`. `docker compose ps` must show `langfuse-worker`. Restart the stack with `make langfuse-up` after the worker is part of that target. |
| `auth_check failed` | Keys in `.env` are not keys of this Langfuse database. Common after a UI signup on a volume that was already initialized, or after recreating the volume without recreating `.env`. | In the UI, open the project, copy the API keys into `.env`, restart uvicorn. Init env vars will not repair an existing database. |
| Web log shows `401` on ingestion | Same key mismatch. The SDK does not fail the HTTP request to PaperLens when export fails. | Fix keys. `flush()` logs a warning when the export itself raises. A 401 from the exporter is often only visible in `make langfuse-logs` or with SDK debug. |
| Web log shows `200`/`207`, UI still empty after a minute | Worker down, worker crashing, or MinIO bucket missing so the job has no payload. | `make langfuse-logs` and read the worker. Confirm the `langfuse` bucket exists in MinIO (`localhost:9090`). |
| Traces appear only after the API process exits | `flush()` is not called on the request path, so the batch waits for process shutdown. | `RAGService.query` must `flush()` in `finally`, outside the `with trace` block. |
| Span duration for retrieval is ~0 ms | The span is opened after the work, on purpose of the current code, and closed immediately. | Read `retrieval.total_ms` and the stage fields. Do not "fix" this by wrapping the whole function unless you also want the Langfuse duration to match the wall clock. |
| `latency_p50` in the UI is a single large number | It is this request's `total_time_ms`, not a percentile. | Use the metrics endpoint for P50/P95. |
| `GET /observability/metrics` returns 500 | `Snapshot` is a dataclass and the handler calls `.model_dump()`. | Serialize with `dataclasses.asdict`. |
| Metrics endpoint is all zeros after you know you sent queries | The process restarted (reload, crash) or you are looking at a different worker process. | Send one more query to this process. Langfuse is the durable copy. |
| First query's `retrieval.semantic_ms` is several seconds, later queries are a few hundred milliseconds | Cold embedding model or cold Chroma, not a tracing bug. | Compare a second query before calling retrieval slow. |
| `LANGFUSE_ENABLED=false` but traces still appear | The running process was started before the edit. | Restart uvicorn. |
| Tracing enabled and `/query` is much slower | Export is supposed to be off the response path. A multi-second stall usually means `flush()` is blocked on an unreachable host (connect timeout). | Check `LANGFUSE_HOST`. From the host, `localhost:3000` is correct. From a container it is not. |
| Child span is its own root | It was created outside the `with trace` block, or on another task. | Open the span in the same task, inside the root `with`. |

---

## Known limitations

- The conversational path has no `generation` observation. Model, finish reason, and the raw completion are logs.
- `retrieval.hybrid` does not contain the chunk ids or titles. Those are logged, not traced.
- `POST /query/stream` opens `rag.query_stream`, scores `ttft_ms`, and flushes in `finally`. It still has no `generation` observation. TTFT percentiles are `latency_p50_ms["stream.ttft"]` on the metrics endpoint.
- `RouterService` and the reranker are instrumented and unused by `routes.py`.
- Score name `latency_p50` is not a percentile.
- Token counts are whitespace splits.
- In-process metrics are per process and unbounded in time only in the sense that ratios cover every request since start; latency samples are capped at 1000 per component.
- OpenTelemetry context does not cross tasks. Do not move retrieval into `asyncio.gather` and expect the child span to stay attached.
- `flush()` does not mean the row is visible in the UI yet. The worker is still in the path.

---

*Document created: 22-10-2026*

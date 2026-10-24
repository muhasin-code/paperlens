Here are the exact code changes for **Step 5**:

---

### 1. Create `docs/streaming.md` (new file)

```markdown
# PaperLens — SSE Streaming Protocol

This document describes the Server-Sent Events (SSE) streaming protocol used by `POST /query/stream`, the latency metrics it emits, how the React frontend consumes it, and where the numbers land in Langfuse and the metrics endpoint.

---

## What this endpoint is

`POST /query/stream` returns `text/event-stream` with token-by-token responses for conversational queries. The blocking endpoint `POST /query` remains unchanged and continues to return a complete `QueryResponse` JSON object. The React UI calls `/query/stream`.

---

## Request

Same body as `/query`:

```json
{
  "query": "string (required, 1–2000 chars)",
  "top_k": "integer (optional, 1–20, default: settings.retrieval_top_k)",
  "model": "string (optional, overrides default model)",
  "mode": "string (optional, 'auto' | 'conversational' | 'structured', default: 'auto')"
}
```

- `mode=structured` returns a single error event (see below).
- `mode=conversational` or omitted/`auto` streams normally.
- The query router (`RouterService`) does **not** run on this route; the route branches directly on `mode`.

---

## Wire format

- Media type: `text/event-stream; charset=utf-8`
- Headers: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`
- Each event is a JSON object prefixed with `data: ` and terminated by a blank line (`\n\n`).

```
data: {"citations": [...], "retrieval_time_ms": 195.3}

data: {"token": "Recent"}

data: {"token": " papers"}

data: {"generation_time_ms": 2847.3, "total_time_ms": 3042.1, "confidence": 0.7245, "model": "phi4-mini", "ttft_ms": 350.12, "chunk_count": 27, "chunks_per_sec": 8.9, "done": true}

```

---

## Events

### 1. Citations event (always first)

```json
{
  "citations": [
    {
      "chunk_id": "2401.00001v1_chunk_0001",
      "arxiv_id": "2401.00001v1",
      "title": "Test Paper",
      "authors": ["Alice"],
      "section_label": "method",
      "page_start": 3,
      "page_end": 4,
      "score": 0.85,
      "rank": 1
    }
  ],
  "retrieval_time_ms": 195.3
}
```

### 2. Token events (zero or more)

```json
{"token": "Recent"}
{"token": " papers"}
```

### 3. Final event (always last, `done: true`)

```json
{
  "generation_time_ms": 2847.3,
  "total_time_ms": 3042.1,
  "confidence": 0.7245,
  "model": "phi4-mini",
  "ttft_ms": 350.12,
  "chunk_count": 27,
  "chunks_per_sec": 8.9,
  "done": true
}
```

### 4. Refusal variant (when completion equals `refusal_message` exactly)

```json
{
  "generation_time_ms": 120.5,
  "total_time_ms": 315.8,
  "confidence": 0.0,
  "model": "phi4-mini",
  "ttft_ms": 45.2,
  "chunk_count": 1,
  "chunks_per_sec": 8.3,
  "refused": true,
  "citations": [],
  "done": true
}
```

> The initial citations event still contains the retrieval citations. The final event's `"citations": []` tells the client to clear them.

### 5. Empty retrieval (no chunks found)

```json
{
  "answer": "I cannot answer from the provided sources.",
  "citations": [],
  "confidence": 0.0,
  "retrieval_time_ms": 180.2,
  "generation_time_ms": 0,
  "total_time_ms": 180.2,
  "done": true
}
```
No token events, no `stream()` call.

### 6. Structured mode error

```json
{
  "error": "Streaming not available for structured extraction",
  "done": true
}
```
Single event, valid JSON, parsable by `JSON.parse`.

---

## Latency definitions

| Field | Definition |
|-------|------------|
| **TTFT** (`ttft_ms`) | Milliseconds from the start of the successful `stream()` call to the first non-empty chunk yielded to the client. Retrieval is **not** included. A failed primary attempt that yielded nothing is **not** included. |
| **Chunk** | One string yielded by `LLMProvider.stream()` (an Ollama `response` piece or a Groq `delta.content`). **Not** a tokenizer token. |
| **`chunk_count`** | Number of non-empty chunks forwarded to the client. |
| **`chunks_per_sec`** | `chunk_count / ((generation_time_ms - ttft_ms) / 1000)`. Zero when denominator ≤ 0. This is a **yield rate**, not a tokenizer token rate. |
| **`generation_time_ms`** | From the start of the successful `stream()` call to the last chunk. Includes TTFT. |
| **`total_time_ms`** | From the start of `query_stream()` to the final event. Includes retrieval. |

There is **no** `total_tokens` field. The project has no tokenizer on the streaming path.

---

## Fallback behaviour

- Primary exception **before any chunk** → falls back to `llm_fallback_model`.
- Primary exception **after at least one chunk** → exception propagates, stream ends, **no fallback** (the client would otherwise see two answers glued together).
- The client should treat a missing `done: true` event as a failure and **not** retry automatically (a retry would re-run retrieval and the model).

---

## Frontend integration

The React app (`frontend/src/App.tsx`) uses `fetch` + `ReadableStream` (not `EventSource`, which only supports GET). It does **not** implement reconnection — a dropped POST is a failed request.

### Parsing loop (simplified)

```typescript
const reader = response.body?.getReader();
const decoder = new TextDecoder("utf-8");
let partialLine = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const textChunk = decoder.decode(value, { stream: true });
  const lines = (partialLine + textChunk).split("\n");
  partialLine = lines.pop() || "";

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data: ")) continue;
    const jsonStr = trimmed.substring(6).trim();
    if (jsonStr === "[DONE]") break;

    const parsed = JSON.parse(jsonStr);

    if (parsed.token !== undefined) {
      setAnswer(prev => prev + parsed.token);
    }
    if (Array.isArray(parsed.citations)) {
      setCitations(parsed.citations); // replace, not append
    }
    if (typeof parsed.ttft_ms === "number") setTtft(parsed.ttft_ms);
    if (typeof parsed.chunks_per_sec === "number") setChunksPerSec(parsed.chunks_per_sec);
    if (typeof parsed.total_time_ms === "number") setLatency(parsed.total_time_ms);
    if (typeof parsed.error === "string") setError(parsed.error);
  }
}
```

### Displayed metrics

- **Latency badge**: `total_time_ms` from final event (not `performance.now()`).
- **TTFT badge**: `ttft_ms` from final event.
- **Chunks/sec badge**: `chunks_per_sec` from final event (shown during streaming).

---

## Langfuse tracing

### Trace name
`rag.query_stream`

### Spans
- Child `retrieval.hybrid` (emitted by `HybridRetriever.search` when retrieval returns chunks).

### Scores on the trace
- `latency_p50` = this request's `total_time_ms` (historical name, not a percentile).
- `citation_coverage` = 1.0 if cited, 0.0 on refusal or empty retrieval.
- `confidence` = heuristic (mean of top-3 retrieval scores, 0 on refusal).
- `ttft_ms` = measured TTFT (only when a first chunk existed).

### No `generation` span
By design — the streaming path records TTFT and chunk counts as trace scores and in the final SSE event. The blocking path `rag.query` does have a `generation` span.

### Flush
`observer.flush()` runs in `finally`, **outside** the `with trace` block (flushing inside exports an open span).

---

## Metrics endpoint

`GET /observability/metrics` includes TTFT under the component key `stream.ttft`:

```json
{
  "latency_p50_ms": {
    "retrieval": 195.3,
    "generation": 2847.3,
    "rag.total": 3042.1,
    "stream.ttft": 350.12
  },
  "latency_p95_ms": {
    "retrieval": 420.1,
    "generation": 4100.0,
    "rag.total": 4520.0,
    "stream.ttft": 680.5
  }
}
```

Same percentile rules as other components: P50 = `sorted[n//2]`, P95 = `sorted[int(n*0.95)]` once ≥ 20 samples, else max. Process-local, reset on process restart.

---

## Source files

| File | Role |
|------|------|
| `src/paperlens/api/rag.py` | `query_stream()` implementation, TTFT measurement, trace |
| `src/paperlens/api/routes.py` | `/query/stream` endpoint, structured error, headers |
| `src/paperlens/llm/ollama_provider.py` | `stream()` with TTFT logging |
| `src/paperlens/llm/openai_compat_provider.py` | `stream()` matching `generate()` options |
| `src/paperlens/observability/metrics.py` | `track_latency("stream.ttft", ...)` |
| `frontend/src/App.tsx` | SSE consumer, metric display |
| `tests/test_api/test_query_stream.py` | Streaming tests |

---

## Troubleshooting

| What you see | Cause | What to do |
|---|---|---|
| Whole answer appears at once | Proxy/buffering between client and uvicorn | `curl -N` against `:8000` is the control. If curl streams and browser doesn't, the extra hop is buffering. UI calls `:8000` directly. |
| `JSON.parse` fails on structured error | Old single-quoted body still deployed | Restart uvicorn. Body is now `json.dumps`. |
| TTFT ≈ retrieval time + generation | TTFT measured from `retrieval_start` | Must start at `stream_start` of the attempt that yielded. |
| Two answers in one stream | Fallback ran after chunks yielded | Bug. Test `test_primary_yields_then_raises_no_fallback` prevents it. |
| Latency badge > `total_time_ms` in payload | `performance.now()` fallback still in `handleSubmit` | Remove it (see frontend changes in Step 3). |
| No `rag.query_stream` in Langfuse | Worker down, keys wrong, `flush` not in `finally` | Same checks as `docs/observability.md`. `initialized` is not enough. |
| Vite won't start | Port 3000 is Langfuse | Port is 5173 (see `vite.config.ts`). |
| Stream and `/query` disagree | One path uses `build_prompt()` or `num_predict` 256 | Both now go through `_user_prompt` + `_llm_options`. |

---

*Document created: 23-10-2026*
```

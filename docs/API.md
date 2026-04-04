# Legal Intelligence Engine — API Reference

Production RAG service for the **Mexican Penal Code (CDMX)**.
Base URL (Railway): `https://your-app.railway.app`
Interactive docs (Swagger UI): `https://your-app.railway.app/docs`
ReDoc: `https://your-app.railway.app/redoc`

---

## Authentication

Most endpoints are public. The `/v1/evaluate` endpoint requires an
`X-Admin-Key` header whose value must match the `ADMIN_API_KEY` environment
variable configured on the server.

---

## Rate Limiting

Per-IP fixed-window rate limit (default: **20 requests / 60 seconds**).
Configurable via the `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_S` env vars.

When the limit is exceeded the server returns **429** with a `Retry-After`
header indicating how many seconds to wait.

---

## Endpoints

### POST /v1/query — Blocking RAG query

Runs the full pipeline (query expansion → dense search → reranking →
generation) and returns a complete `QueryResponse`.

```bash
curl -X POST https://your-app.railway.app/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es la pena por robo?", "top_k": 5}'
```

With optional metadata filters:

```bash
curl -X POST https://your-app.railway.app/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuál es la pena por homicidio doloso?",
    "filters": {"chapter": "Título Segundo"},
    "top_k": 10
  }'
```

**Request body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Legal question in Spanish (5–1000 chars) |
| `filters` | object \| null | `null` | Pinecone metadata filter (see examples above) |
| `top_k` | integer | `5` | Chunks returned after reranking (1–20) |
| `stream` | boolean | `false` | Hint to the client to use the streaming endpoint instead |

**Response — 200 OK**

```json
{
  "answer": "El robo simple se sanciona con prisión de tres meses a tres años conforme al Artículo 220 del Código Penal del CDMX.",
  "citations": [
    {
      "article_number": "220",
      "source_text": "Artículo 220. Al que con ánimo de dominio y sin consentimiento...",
      "confidence": 0.89
    }
  ],
  "retrieval": {
    "chunks_retrieved": 52,
    "chunks_after_rerank": 5,
    "avg_similarity_score": 0.76,
    "retrieval_time_ms": 9120.4,
    "expanded_queries": [
      "pena robo código penal CDMX",
      "sanción robo simple México",
      "años prisión robo"
    ]
  },
  "generation_time_ms": 1850.2,
  "total_time_ms": 10970.6,
  "model": "claude-haiku-4-5-20251001",
  "prompt_version": "v1.v1",
  "hallucination_flags": []
}
```

**Error responses**

| Status | `error` value | Cause |
|--------|--------------|-------|
| 422 | `Validation error` | Question too short/long, `top_k` out of 1–20 range |
| 429 | `Rate limit exceeded` | Per-IP limit hit; check `Retry-After` header |
| 500 | `Pipeline error` | Pinecone or Anthropic API failure |

---

### POST /v1/query/stream — Streaming SSE query

Same pipeline as `/v1/query` but streams the answer as **Server-Sent Events**.
Three event types are emitted in order:

| Event | Payload | When |
|-------|---------|------|
| `retrieval` | `RetrievalMetadata` JSON | After expand→search→rerank completes |
| `token` | `{"token": "<text>"}` | For every output token from Claude |
| `complete` | `QueryResponse` JSON | After the last token; includes full citations |
| `error` | `{"error": "...", "request_id": "..."}` | On unhandled pipeline exception |

```bash
curl -X POST https://your-app.railway.app/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es la pena por robo?"}' \
  --no-buffer
```

**Example SSE output**

```
event: retrieval
data: {"chunks_retrieved":52,"chunks_after_rerank":5,"avg_similarity_score":0.76,"retrieval_time_ms":9120.4,"expanded_queries":["pena robo CDMX",...]}

event: token
data: {"token":"El"}

event: token
data: {"token":" robo"}

...

event: complete
data: {"answer":"El robo simple se sanciona...","citations":[...],"hallucination_flags":[],...}
```

**JavaScript client example** (fetch + ReadableStream):

```javascript
const resp = await fetch('/v1/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: '¿Cuál es la pena por robo?' }),
});

const reader = resp.body.getReader();
const decoder = new TextDecoder();

for await (const chunk of readLines(reader)) {
  if (chunk.startsWith('event: token')) {
    const data = JSON.parse(chunk.replace(/^data: /, ''));
    process.stdout.write(data.token);
  }
}
```

---

### GET /v1/health — Service health probe

Returns connectivity status for Pinecone and the Anthropic API, the current
vector count, and process uptime.

```bash
curl https://your-app.railway.app/v1/health
```

**Response — 200 OK**

```json
{
  "status": "healthy",
  "pinecone_connected": true,
  "anthropic_connected": true,
  "index_vector_count": 12847,
  "uptime_seconds": 3661.2,
  "version": "0.1.0"
}
```

`status` is one of `healthy` (both services up), `degraded` (one service
down), or `unhealthy` (both down).  The HTTP status code is always **200**
regardless of `status` — callers should inspect the JSON field.

---

### GET /v1/metrics — Prometheus metrics

Returns Prometheus text exposition format (`text/plain; version=0.0.4`).

```bash
curl https://your-app.railway.app/v1/metrics
```

**Example output**

```
# HELP legal_query_total Total RAG query requests
# TYPE legal_query_total counter
legal_query_total{status="success"} 142.0
legal_query_total{status="error"} 3.0
legal_query_total{status="rate_limited"} 7.0

# HELP legal_query_duration_seconds End-to-end query wall time
# TYPE legal_query_duration_seconds histogram
legal_query_duration_seconds_bucket{endpoint="query",le="5.0"} 12.0
...

# HELP legal_active_requests Requests currently being processed
# TYPE legal_active_requests gauge
legal_active_requests 2.0
```

---

### POST /v1/evaluate — Trigger evaluation run (admin)

Starts a RAGAS + citation-accuracy evaluation in a background worker.
Returns **202 Accepted** immediately; results are written to
`results/eval_{run_id}.json` when the job completes.

Requires `X-Admin-Key` header.

```bash
curl -X POST https://your-app.railway.app/v1/evaluate \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-admin-key" \
  -d '{
    "test_set_path": "data/evaluation/golden_set.json",
    "category": "homicidio"
  }'
```

Run a single question by ID:

```bash
curl -X POST https://your-app.railway.app/v1/evaluate \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-admin-key" \
  -d '{
    "test_set_path": "data/evaluation/golden_set.json",
    "question_id": "q-042"
  }'
```

**Response — 202 Accepted**

```json
{
  "job_id": "a1b2c3d4",
  "status": "started",
  "message": "Evaluation job a1b2c3d4 is running in the background. Results will be saved to results/eval_{run_id}.json. Monitor progress via server logs (filter: job_id=a1b2c3d4).",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error responses**

| Status | Cause |
|--------|-------|
| 403 | `X-Admin-Key` value does not match `ADMIN_API_KEY` |
| 503 | `ADMIN_API_KEY` is not configured on this instance |

---

## Common Error Envelope

All 4xx and 5xx responses share this shape:

```json
{
  "error": "Rate limit exceeded",
  "detail": "Retry after 42s (20 req/60s per IP)",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

The `request_id` matches the `X-Request-ID` response header and appears in
all server log lines for that request — use it to correlate client errors
with server-side traces.

---

## Request Tracing

Every request/response pair carries an `X-Request-ID` header (UUID4).
Pass your own `X-Request-ID` on the way in to propagate a correlation ID from
an upstream service:

```bash
curl -X POST https://your-app.railway.app/v1/query \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-trace-id-001" \
  -d '{"question": "¿Qué es el fraude?"}'
```

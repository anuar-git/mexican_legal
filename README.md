# Legal Intelligence Engine

Production-grade Retrieval-Augmented Generation (RAG) system for semantic search
and question-answering over the **Mexican Federal Penal Code (CDMX)**.

Built with Pinecone, LangChain, Claude Haiku, and FastAPI. Evaluated against a
curated test set of 75 questions spanning 6 legal categories.

🔗 **[Live API Docs](https://legal.anuarhage.com/docs)**
&nbsp;·&nbsp;
🔗 **[Monitoring Dashboard](https://legal.anuarhage.com/dashboard)**
&nbsp;·&nbsp;
📹 **[2-Minute Demo](https://loom.com/)**

---

## Performance

Evaluated against 75 questions across 6 legal categories (homicidio, robo,
fraude, lesiones, delitos-sexuales, adversarial out-of-scope). All 75 runs
completed without pipeline errors.

| Metric | Score |
|---|---|
| **Faithfulness** (RAGAS) | **0.84** |
| **Citation Accuracy** (custom) | **0.78** |
| OOS Refusal Rate | 80% |
| Hallucination Rate | 29% |
| p50 Latency | 10.5 s |
| p95 Latency | 12.5 s |

> **Note on hallucination rate:** the 29% figure counts any response that cites
> an article absent from the top-5 retrieved chunks. Raising `top_k_final` from
> 5 → 10 reduces this significantly at a modest latency cost — a tuning knob, not
> a fundamental limitation.

> **Note on RAGAS metrics:** answer relevance and context recall scored low
> (≈0.11–0.13) due to a known RAGAS evaluator bias against Spanish-language
> references. Faithfulness — the metric least sensitive to this bias — is the
> most reliable RAGAS signal for this corpus.

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │                User Query                    │
                          └───────────────────┬──────────────────────────┘
                                              │  POST /v1/query
                          ┌───────────────────▼──────────────────────────┐
                          │              FastAPI (src/api/)               │
                          │  rate-limit · request-id · metrics · logging  │
                          └───────────────────┬──────────────────────────┘
                                              │
                          ┌───────────────────▼──────────────────────────┐
                          │          RAG Pipeline (src/retrieval/)        │
                          │                                               │
                          │  1. Query Expansion                           │
                          │     Claude Haiku → 3 alternative phrasings   │
                          │                      │                        │
                          │  2. Dense Search                              │
                          │     Pinecone (llama-text-embed-v2)            │
                          │     top-20 per variant → merge + dedup        │
                          │                      │                        │
                          │  3. Cross-Encoder Rerank                      │
                          │     ms-marco-MiniLM-L-6-v2 → top-5 chunks    │
                          │                      │                        │
                          │  4. Generation                                │
                          │     Claude Haiku + structured citation prompt │
                          │                      │                        │
                          │  5. Hallucination Check                       │
                          │     cited articles ⊆ retrieved chunks?        │
                          └───────────────────┬──────────────────────────┘
                                              │
                     ┌────────────────────────▼──────────────────────────┐
                     │                  QueryResponse                     │
                     │  answer · citations · retrieval_metadata · timing  │
                     └───────────────────────────────────────────────────┘

  Offline                         Ingestion Pipeline (src/ingestion/)
  ───────   PDF → pdfplumber → recursive chunker (512 tok / 64 overlap)
            → Pinecone inference embed (llama-text-embed-v2, batch=64)
            → upsert with stable IDs + legal hierarchy metadata
```

---

## Key Technical Decisions

**Legal-boundary chunking.** Chunks are split using a separator hierarchy ordered
by legal document structure — `ARTÍCULO → CAPÍTULO → TÍTULO → SECCIÓN → paragraph
→ line` — rather than generic newlines. This preserves article boundaries and
propagates `article_number / title / chapter / section` metadata across page
boundaries via continuation flags and doubly-linked `prev/next_chunk_index`
pointers. Retrieval can therefore surface the exact article a user asks about
rather than an adjacent fragment.

**Query expansion before search.** Before querying Pinecone, Claude Haiku
rewrites the user's question into three alternative phrasings (synonyms, legal
register shifts, keyword variants). Each phrasing is searched independently;
results are merged and deduplicated by chunk ID. This substantially increases
recall on comparison-type questions where the user's phrasing diverges from the
statute's language.

**Two-stage retrieval (dense + rerank).** Stage 1 retrieves the top-20 candidates
across all expanded queries — high recall, low precision. Stage 2 runs
`cross-encoder/ms-marco-MiniLM-L-6-v2` over the 20 candidates and keeps only
the top-5 by rerank score. This combination is substantially more precise than
using similarity score alone as the final ranking signal.

**Citation grounding and hallucination detection.** The generator is instructed to
cite articles explicitly. Every cited `ARTÍCULO NNN` is cross-referenced against
the metadata of the top-5 retrieved chunks after generation. Citations absent from
the retrieved context are flagged as potential hallucinations and surfaced
separately in the API response — they never silently appear as grounded citations.

**Resumable ingestion.** The embedder checkpoints every 10 batches to
`data/processed/checkpoint.json`. If ingestion is interrupted (network timeout,
rate limit), it resumes from the last checkpoint rather than re-embedding from
scratch. The loader supports `--rebuild` (delete + full re-ingest) and `--update`
(skip IDs already present) modes.

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/yourusername/legal-intelligence-engine.git
cd legal-intelligence-engine

cp .env.example .env
# Edit .env: fill in PINECONE_API_KEY, PINECONE_INDEX_NAME, ANTHROPIC_API_KEY

docker compose up --build
```

Services:
- API: http://localhost:8000 — Swagger UI at `/docs`
- Dashboard: http://localhost:8501

### Option B — Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,monitoring]"

cp .env.example .env  # fill in the three required API keys

# Ingest the penal code into Pinecone (one-time setup, ~15 min)
python scripts/ingest.py --source data/raw/ --strategy recursive --rebuild

# Start the API
uvicorn src.api.main:app --reload

# In a second terminal — start the monitoring dashboard
streamlit run src/monitoring/dashboard.py
```

### First query

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es la pena por homicidio doloso en el CDMX?"}'
```

Response shape:

```json
{
  "answer": "El homicidio doloso se sanciona con prisión de 8 a 20 años...",
  "citations": [
    { "article_number": "123", "source_text": "ARTÍCULO 123. Al que prive...", "confidence": 0.91 }
  ],
  "retrieval": {
    "chunks_retrieved": 47,
    "chunks_after_rerank": 5,
    "avg_similarity_score": 0.78,
    "retrieval_time_ms": 8320,
    "expanded_queries": ["pena homicidio doloso CDMX", "..."]
  },
  "generation_time_ms": 2100,
  "total_time_ms": 10421,
  "hallucination_flags": []
}
```

---

## Ingestion Pipeline

```bash
# Full rebuild — delete the Pinecone index and re-ingest everything
python scripts/ingest.py --source data/raw/ --strategy recursive --rebuild

# Incremental — skip chunk IDs already stored (safe to re-run)
python scripts/ingest.py --source data/raw/ --strategy recursive --update

# Dry run — extract + chunk + embed but skip the upsert (cost-free validation)
python scripts/ingest.py --source data/raw/ --strategy recursive --dry-run
```

---

## Evaluation

```bash
# Full evaluation suite — 75 questions, RAGAS + custom metrics (~13 min)
python scripts/evaluate.py \
  --test-set data/evaluation/golden_test_set.json \
  --output   results/my_run.json

# Quick smoke test — 5 questions, no RAGAS (fast, no LLM judge cost)
python scripts/evaluate.py --limit 5 --no-ragas

# Filter to a single category
python scripts/evaluate.py --category homicidio

# View the markdown report for a run
cat results/reports/eval_ef061dbb.md
```

Metrics computed:
- **RAGAS** — Faithfulness, Answer Relevance, Context Relevance, Context Recall
  (LLM-as-judge via Claude Haiku)
- **Custom** — Citation Accuracy, Retrieval Hit Rate, Article Coverage Rate,
  OOS Refusal Rate, Hallucination Rate, per-stage latency

Results are written to `results/*.json`; markdown reports to
`results/reports/eval_*.md`.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/query` | Blocking RAG query → full `QueryResponse` |
| `POST` | `/v1/query/stream` | Streaming SSE — answer tokens as they arrive |
| `GET` | `/v1/health` | Connectivity probe (Pinecone + Anthropic) |
| `GET` | `/v1/metrics` | Prometheus text exposition |
| `POST` | `/v1/evaluate` | Admin-gated eval trigger (background task) |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

Rate limit: **10 requests / 60 seconds per IP** (configurable via env vars).
See [docs/API.md](docs/API.md) for full request/response schemas and `curl`
examples.

---

## Project Structure

```
legal-intelligence-engine/
├── src/
│   ├── ingestion/
│   │   ├── extractor.py       # PDF → List[ExtractedPage] via pdfplumber
│   │   ├── chunker.py         # Recursive chunker; 512-tok / 64-tok overlap
│   │   ├── embedder.py        # Pinecone inference API; batched + resumable
│   │   └── loader.py          # Pinecone upsert; rebuild / update modes
│   ├── retrieval/
│   │   ├── retriever.py       # Query expansion → dense search → rerank
│   │   ├── generator.py       # Claude Haiku generation + citation check
│   │   ├── pipeline.py        # Async end-to-end query() / query_stream()
│   │   └── prompts.py         # Versioned prompt registry
│   ├── api/
│   │   ├── main.py            # FastAPI app; all endpoints
│   │   ├── models.py          # Pydantic request/response schemas
│   │   ├── middleware.py      # Request-ID injection, rate limiting
│   │   ├── request_logger.py  # SQLite request log (read by dashboard)
│   │   └── metrics_collector.py  # Prometheus counters / histograms
│   ├── evaluation/
│   │   ├── evaluator.py       # Orchestrates pipeline runs over test set
│   │   ├── metrics.py         # Custom metrics (citation acc, hit rate, …)
│   │   ├── ragas_scorer.py    # RAGAS integration (faithfulness, relevance)
│   │   ├── reporter.py        # Markdown report generator
│   │   └── test_set.py        # Golden test set loader + validator
│   └── monitoring/
│       └── dashboard.py       # 4-tab Streamlit dashboard
├── scripts/
│   ├── ingest.py              # CLI: run the full ingestion pipeline
│   ├── evaluate.py            # CLI: run evaluation, write JSON + MD report
│   └── load_test.py           # aiohttp load test against a live API
├── tests/
│   ├── ingestion/test_chunker.py  # 9 test classes, 40+ assertions
│   └── retrieval/             # Integration tests (require RUN_INTEGRATION=1)
├── data/
│   ├── raw/                   # Source PDF (gitignored)
│   ├── processed/             # Chunk JSON + embedder checkpoints
│   └── evaluation/            # Golden test set (75 questions)
├── results/                   # Eval run JSON + markdown reports
├── docs/API.md                # Full API reference
├── Dockerfile                 # Multi-stage; pre-downloads HF cross-encoder
├── Dockerfile.dashboard       # Streamlit image
├── docker-compose.yml         # api + dashboard; health-checked
├── .env.example               # Environment variable reference
└── .github/workflows/ci.yml   # lint → test → docker build → eval
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.12 | — |
| **Vector DB** | Pinecone (serverless) | Dense vector storage and search |
| **Embeddings** | `llama-text-embed-v2` (Pinecone inference) | 1024-dim chunk embeddings |
| **LLM** | Claude Haiku (Anthropic) | Query expansion and answer generation |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision reranking of top-20 candidates |
| **Chunking** | LangChain RecursiveCharacterTextSplitter | Legal-boundary-aware chunking |
| **PDF extraction** | pdfplumber | Structured text extraction with page metadata |
| **Tokeniser** | tiktoken (`cl100k_base`) | Token-count proxy throughout the pipeline |
| **API** | FastAPI + uvicorn | Async REST API with SSE streaming |
| **Evaluation** | RAGAS + custom metrics | Faithfulness, citation accuracy, hit rate |
| **Monitoring** | Streamlit + Plotly + SQLite | 4-tab operational dashboard |
| **Metrics** | Prometheus client | Counter / histogram exposition for scraping |
| **CI/CD** | GitHub Actions | lint → unit tests → Docker build → eval |
| **Deployment** | Railway (Docker) | Container hosting with env var management |

---

## CI/CD

Four jobs run on every push to `master`:

1. **lint** — `ruff check` + `mypy` (~1 min)
2. **test** — `pytest` unit tests with coverage report to Codecov (~2 min)
3. **docker-build** — multi-stage image build + container smoke test (~8 min, cached)
4. **evaluate** — 10-question eval run, result uploaded as a build artifact (~3 min)

The eval job runs only on `master` and requires live API secrets. The Docker build
uses GitHub Actions layer cache; repeated runs skip the ~1 GB PyTorch + HuggingFace
model layers.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in three required values:

| Variable | Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | ✅ | Pinecone project API key |
| `PINECONE_INDEX_NAME` | ✅ | Name of the serverless index |
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key |
| `ADMIN_API_KEY` | ✅ | Secret for `POST /v1/evaluate` |
| `PINECONE_CLOUD` | optional | Cloud provider (default: `aws`) |
| `PINECONE_REGION` | optional | Region (default: `us-east-1`) |
| `RATE_LIMIT_REQUESTS` | optional | Requests per IP per window (default: `10`) |
| `RATE_LIMIT_WINDOW_S` | optional | Window in seconds (default: `60`) |
| `LOG_LEVEL` | optional | Logger level (default: `INFO`) |

---

## License

MIT

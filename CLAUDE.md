# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Production RAG system for the Mexican Penal Code (CDMX). Stack: Pinecone, LangChain, Claude Haiku, FastAPI, Railway deployment. Source document: `data/raw/codigo_penal_cdmx_31225.pdf`.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"

# Development
uvicorn src.api.main:app --reload
streamlit run src/monitoring/dashboard.py

# Ingestion pipeline (mutually exclusive modes)
python scripts/ingest.py --source data/raw/ --strategy recursive --rebuild   # Delete index, full re-ingest
python scripts/ingest.py --source data/raw/ --strategy recursive --update    # Incremental (skip existing IDs)
python scripts/ingest.py --source data/raw/ --strategy recursive --dry-run   # Extract+chunk+embed, skip upsert

# Evaluation
python scripts/evaluate.py             # Run RAGAS + citation accuracy eval

# Quality
pytest tests/ -v                                               # unit tests only (integration skipped)
RUN_INTEGRATION=1 pytest tests/retrieval/test_pipeline.py -v  # integration tests (hits live APIs)
pytest tests/path/to/test_file.py::test_name -v
ruff check src/
mypy src/
```

## Implementation Status

Fully implemented:
- `src/ingestion/` — entire 4-step ingestion pipeline
- `src/retrieval/` — query expansion, dense search, reranking, generation, end-to-end pipeline
- `src/api/` — FastAPI endpoints (`/v1/query`, `/v1/query/stream`, `/v1/health`, `/metrics`, `/v1/evaluate`), Pydantic models, rate-limiting middleware, SQLite request logger, Prometheus metrics collector
- `src/evaluation/` — RAGAS scorer, custom metrics (citation accuracy, hit rate, hallucination, OOS refusal, coverage), evaluator runner, reporter
- `src/monitoring/dashboard.py` — 4-tab Streamlit dashboard (Operations Overview, Retrieval Quality, Evaluation Dashboard, Live Query Tester)
- `scripts/ingest.py` — ingestion CLI
- `scripts/evaluate.py` — evaluation CLI with `--question-id`, `--no-ragas`, `--compare` flags
- `scripts/load_test.py` — async load tester via aiohttp
- `tests/ingestion/test_chunker.py` — 9 test classes covering the chunker
- `tests/retrieval/test_pipeline.py` — integration tests for full RAG flow (requires `RUN_INTEGRATION=1`)
- `tests/api/test_models.py` — 30+ unit tests for all Pydantic models
- `tests/evaluation/test_metrics.py` — 30+ unit tests for all custom metric functions

Stubbed (empty files, not yet implemented):
- `src/evaluation/test_set.py` — golden test set loader (questions loaded from `data/evaluation/golden_test_set.json` directly by the evaluator)
- `src/monitoring/collectors.py` — Prometheus metric collectors (metrics are collected via `src/api/metrics_collector.py` instead)
- `scripts/serve.py` — server start script (use `uvicorn src.api.main:app --reload` directly)

## Architecture

Linear 5-step pipeline:

1. **Ingestion** (`src/ingestion/`)
   - `extractor.py`: `extract_pages(pdf_path)` → `List[ExtractedPage]` via pdfplumber
   - `chunker.py`: `recursive_chunk(pages)` → `List[TextChunk]`; 512-token chunks with 64-token overlap; separators ordered by legal document structure (ARTÍCULO → CAPÍTULO → TÍTULO → SECCIÓN → paragraph → line); extracts and propagates article/title/chapter/section hierarchy across page boundaries; chunks carry `is_continuation` flag and doubly-linked `prev/next_chunk_index` pointers
   - `embedder.py`: `embed_chunks(chunks)` → `List[EmbeddedChunk]` using Pinecone inference API (`llama-text-embed-v2`); batches of 64, checkpoints every 10 batches to `data/processed/checkpoint.json` (resumable)
   - `loader.py`: `upsert_chunks(chunks, mode)` → upserts to Pinecone; stable chunk IDs as `{source_file_stem}-{chunk_index}`; `rebuild` mode deletes the index first, `update` mode skips already-stored IDs

2. **Retrieval** (`src/retrieval/`)
   - `retriever.py`: `Retriever.retrieve(query, filters)` → `RetrievalResult`; Stage 1: Claude Haiku expands query into 3 alternatives; Stage 2: Pinecone dense search top-20 per variant, merged + deduplicated by chunk ID; Stage 3: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks to top-5
   - `generator.py`: `Generator.generate(retrieval)` → `GenerationResult`; formats chunks with article/chapter headers; calls Claude Haiku with `SYSTEM_PROMPT`; extracts cited article numbers via regex; cross-references against retrieved chunks (`hallucination_flag=True` when article absent); `generate_stream()` yields tokens for SSE
   - `prompts.py`: single source of truth for all prompts; versioned as `SYSTEM_PROMPT_V1`, `QUERY_TEMPLATE_V1`, `EXPANSION_TEMPLATE_V1`; `PROMPT_REGISTRY` dict for eval tooling
   - `pipeline.py`: async `query(question, filters)` and `query_stream(question, filters)`; lazy singleton `Retriever` + `Generator`; sync internals offloaded via `asyncio.to_thread()`

3. **API** (`src/api/`)
   - `config.py`: `Settings` (pydantic-settings); reads all env vars from `.env`
   - `models.py`: all request/response Pydantic models (`QueryRequest`, `QueryResponse`, `CitationResponse`, `RetrievalMetadata`, `HealthResponse`, `EvalRequest`, `ErrorResponse`); `RetrievalResultProtocol` / `GenerationResultProtocol` structural stubs
   - `middleware.py`: `LoggingMiddleware` — per-request timing, structured logging, error capture; rate limiter (sliding window, configurable via `RATE_LIMIT_RPM`)
   - `request_logger.py`: SQLite-backed request log; schema: `(id, timestamp, question, answer, latency_ms, status, error)`; read by the monitoring dashboard
   - `metrics_collector.py`: Prometheus counters and histograms; exposed at `GET /metrics`
   - `main.py`: FastAPI app with lifespan startup probes; endpoints: `GET /v1/health`, `GET /metrics`, `POST /v1/query`, `POST /v1/query/stream` (SSE), `POST /v1/evaluate` (admin-only, background job)

4. **Evaluation** (`src/evaluation/`)
   - `metrics.py`: deterministic custom metrics — `citation_accuracy`, `retrieval_hit_rate`, `detect_hallucination`, `latency_breakdown`, `article_coverage_rate`, `oos_correctly_refused`; aggregation helpers `score_result`, `aggregate`, `stratified_breakdown`, `worst_performers`, `best_performers`
   - `evaluator.py`: `Evaluator.run(questions)` → `EvalRun`; runs each question through the live pipeline; stores `EvalResult` per question with citations, retrieved chunks, latency split
   - `ragas_scorer.py`: wraps RAGAS library; uses Claude Haiku as LLM judge; scores faithfulness, answer_relevance, context_relevance, context_recall
   - `reporter.py`: renders `EvalRun` → markdown report + `results/eval_{run_id}.json`

5. **Monitoring** (`src/monitoring/`)
   - `dashboard.py`: 4-tab Streamlit app; reads SQLite request log and `results/eval_*.json` files; Tab 1: KPI row + latency charts; Tab 2: similarity trends + hallucination rate; Tab 3: eval run selector + failure explorer; Tab 4: live query tester against running API

## Baseline Performance

Evaluation run `ef061dbb` — 75 questions across 15 categories and 4 difficulty levels:

| Metric | Score |
|---|---|
| Faithfulness (RAGAS) | 0.84 |
| Citation Accuracy (custom) | 0.78 |
| OOS Refusal Rate | 80% (8/10) |
| Hallucination Rate | 29% (22/75) |
| Retrieval Hit Rate | 0.15 (inflated by article number mismatch — see tuning notes) |
| p50 Latency | 10.5 s |
| p95 Latency | 12.5 s |

Results stored in `results/eval_ef061dbb.json`. Use this as the comparison baseline when making retrieval or generation changes.

## Known Warnings & Tuning Notes

- **`bert.embeddings.position_ids UNEXPECTED`** — appears on cross-encoder load; harmless, safe to ignore.
- **CUDA driver too old** — cross-encoder runs on CPU; retrieval is slower (~8s). Not a blocker, just a hardware limitation.
- **Citation grounding** — to reduce hallucination flags, raise `top_k_final` from 5 to 8–10 in `Retriever`. Baseline hallucination rate is 29% at top_k=5; raising to 10 reduces it at ~0.5–1s latency cost.
- **Article number mismatch** — golden test set (`data/evaluation/golden_test_set.json`) uses Federal Penal Code article numbers (302, 307); indexed document is CDMX code (articles 123, 125). This inflates the miss rate metric; `retrieval_hit_rate` of 0.15 is partly an evaluation artefact, not purely a retrieval failure.

## Architecture Decisions

- **Chunking**: Both recursive and semantic strategies must be selectable via `--strategy` flag (not hardcoded); semantic strategy is not yet implemented
- **Retrieval**: Multi-query fan-out (dense vector search only, no sparse/BM25) → rerank top-20 to top-5. The method is named `_hybrid_search` but refers to multi-query merging, not sparse+dense fusion. True hybrid (BM25) is a future improvement.
- **Generation**: Claude Haiku with structured citation prompt (article/chapter references)
- **Eval**: RAGAS metrics + custom citation accuracy
- **Tokens**: `cl100k_base` tokenizer (tiktoken) used as proxy for token counting throughout

## Style

- Type hints on all functions
- Docstrings on all modules and public functions
- Pydantic models for all data structures
- Async where possible in API layer

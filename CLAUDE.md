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
- `src/retrieval/` — query expansion, hybrid search, reranking, generation, end-to-end pipeline
- `tests/ingestion/test_chunker.py` — 9 test classes covering the chunker
- `tests/retrieval/test_pipeline.py` — integration tests for full RAG flow (requires `RUN_INTEGRATION=1`)

Stubbed (empty files, not yet implemented):
- `src/api/` — FastAPI endpoints, Pydantic models, middleware
- `src/evaluation/` — RAGAS evaluator, test set, metrics
- `src/monitoring/` — Streamlit dashboard, collectors
- `scripts/evaluate.py`, `scripts/serve.py`

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

3. **API** (`src/api/`) — FastAPI async endpoints, Pydantic models for all shapes

4. **Evaluation** (`src/evaluation/`) — RAGAS metrics + custom citation accuracy

5. **Monitoring** (`src/monitoring/`) — Streamlit dashboard + metrics collectors

## Known Warnings & Tuning Notes

- **`bert.embeddings.position_ids UNEXPECTED`** — appears on cross-encoder load; harmless, safe to ignore.
- **CUDA driver too old** — cross-encoder runs on CPU; retrieval is slower (~8s). Not a blocker, just a hardware limitation.
- **Citation grounding** — to reduce hallucination flags, raise `top_k_final` from 5 to 8–10 in `Retriever`, or lower the reranker score threshold. Defer to evaluation phase.

## Architecture Decisions

- **Chunking**: Both recursive and semantic strategies must be selectable via `--strategy` flag (not hardcoded); semantic strategy is not yet implemented
- **Retrieval**: Hybrid search (dense + sparse) → rerank top-20 to top-5
- **Generation**: Claude Haiku with structured citation prompt (article/chapter references)
- **Eval**: RAGAS metrics + custom citation accuracy
- **Tokens**: `cl100k_base` tokenizer (tiktoken) used as proxy for token counting throughout

## Style

- Type hints on all functions
- Docstrings on all modules and public functions
- Pydantic models for all data structures
- Async where possible in API layer

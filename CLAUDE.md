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
uvicorn src.api.main:app --reload       # API server
streamlit run src/monitoring/dashboard.py

# Pipeline
python scripts/ingest.py                # Extract → chunk → embed → upsert to Pinecone
python scripts/evaluate.py             # Run RAGAS + citation accuracy eval

# Quality
pytest tests/ -v                        # All tests
pytest tests/path/to/test_file.py::test_name -v   # Single test
ruff check src/                         # Lint
mypy src/                               # Type check
```

## Architecture

The pipeline flows linearly through four layers:

1. **Ingestion** (`src/ingestion/`) — PDF → text → chunks → embeddings → Pinecone
   - `extractor.py`: `extract_pages(pdf_path)` → `List[ExtractedPage]` using pdfplumber
   - `chunker.py`: Two selectable strategies — recursive character splitting and semantic chunking; 512-token chunks, 64-token overlap
   - `embedder.py`: Generates embeddings for chunks
   - `loader.py`: Upserts to Pinecone index

2. **Retrieval** (`src/retrieval/`) — Query → hybrid search → rerank → top-5 chunks
   - `retriever.py`: Hybrid search (dense + sparse), reranks top-20 down to top-5
   - `prompts.py`: Structured citation prompt templates for Claude Haiku

3. **API** (`src/api/`) — FastAPI with async endpoints; Pydantic models for all request/response shapes

4. **Evaluation** (`src/evaluation/`) — RAGAS framework + custom citation accuracy metric

5. **Monitoring** (`src/monitoring/`) — Streamlit dashboard + metrics collectors

## Architecture Decisions

- **Chunking**: Both recursive and semantic strategies must be selectable via config (not hardcoded)
- **Retrieval**: Hybrid search (dense + sparse) → rerank top-20 to top-5
- **Generation**: Claude Haiku with structured citation prompt (article/chapter references)
- **Eval**: RAGAS metrics + custom citation accuracy

## Style

- Type hints on all functions
- Docstrings on all modules and public functions
- Pydantic models for all data structures
- Async where possible in API layer

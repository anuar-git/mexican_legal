# Contributing to Legal Intelligence Engine

## Setup

```bash
git clone https://github.com/yourusername/legal-intelligence-engine.git
cd legal-intelligence-engine

cp .env.example .env
# Edit .env — fill in PINECONE_API_KEY, PINECONE_INDEX_NAME, ANTHROPIC_API_KEY

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test,eval,monitoring]"
```

## Development Commands

```bash
# Lint
ruff check src/

# Type check
mypy src/ --ignore-missing-imports --explicit-package-bases

# Unit tests (no live APIs)
pytest tests/ -v -m "not integration"

# Integration tests (requires live API keys + RUN_INTEGRATION=1)
RUN_INTEGRATION=1 pytest tests/retrieval/test_pipeline.py -v

# Coverage report
pytest tests/ -m "not integration" --cov=src --cov-report=term-missing

# Start the API locally
uvicorn src.api.main:app --reload

# Start the monitoring dashboard
streamlit run src/monitoring/dashboard.py
```

## Project Structure

The codebase is split into five independent packages — each is independently
testable and has no circular imports:

```
src/ingestion/   PDF → chunks → embeddings → Pinecone (offline)
src/retrieval/   expand → search → rerank → generate (hot path)
src/api/         FastAPI endpoints + middleware + request logging
src/evaluation/  golden test set runner + RAGAS + custom metrics
src/monitoring/  Streamlit dashboard
```

## Making Changes

### Ingestion (`src/ingestion/`)

After modifying chunking or embedding logic, re-ingest and re-evaluate:

```bash
python scripts/ingest.py --source data/raw/ --strategy recursive --rebuild
python scripts/evaluate.py --no-ragas --limit 10  # quick sanity check
```

### Retrieval or Generation (`src/retrieval/`)

Any change to `retriever.py`, `generator.py`, `prompts.py`, or `pipeline.py`
requires a full evaluation run before the PR can merge. Include the
`results/eval_{run_id}.json` artefact in the PR description along with a
`--compare` diff against the baseline:

```bash
python scripts/evaluate.py --output results/my_change.json
python scripts/evaluate.py --compare results/eval_baseline.json results/my_change.json
```

The baseline run is `results/eval_ef061dbb.json` (run date: 2026-03-29,
75 questions, Faithfulness 0.84, Citation Accuracy 0.78).

### Adding Test Questions

Add entries to `data/evaluation/golden_test_set.json`.  Each entry requires:

| Field | Notes |
|---|---|
| `id` | Sequential `qNNN` — no gaps |
| `question` | Spanish, natural phrasing a user would actually type |
| `expected_answer` | Verbatim excerpt from the statute or a concise synthesis |
| `expected_articles` | Article numbers the answer must cite — verified against the PDF |
| `category` | One of the existing legal categories (see the distribution table in `docs/EVALUATION.md`) |
| `difficulty` | `easy` / `medium` / `hard` / `adversarial` |
| `question_type` | `factual_lookup` / `comparison` / `multi_article` / `conditional` / `adversarial_oos` / `adversarial_tricky` |
| `notes` | What a correct answer must include — used as a human review checklist |

## Pull Requests

**Branch naming**

```
feature/description     new capability
fix/description         bug fix
eval/description        evaluation run or metric change
docs/description        documentation only
```

**PR checklist**

- [ ] `ruff check src/` passes with no errors
- [ ] `mypy src/ --ignore-missing-imports --explicit-package-bases` passes
- [ ] `pytest tests/ -m "not integration"` passes (90+ tests, no failures)
- [ ] If retrieval/generation changed: evaluation results attached and compared to baseline
- [ ] If new public functions added: Google-style docstring present
- [ ] If schema changed: `src/api/models.py` updated and `docs/API.md` reflects the change

**CI pipeline** runs automatically on every PR:

1. `ruff check` + `mypy` (~1 min)
2. `pytest` with coverage (~2 min)
3. Docker multi-stage build + container smoke test (~8 min, cached)
4. Evaluation on 10 questions against live APIs (master branch only)

All four jobs must pass before a PR can merge.

## Environment Variables

See `.env.example` for the full reference. The three variables needed for local
development with live APIs:

```
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=legal-intelligence
ANTHROPIC_API_KEY=...
```

Tests marked `not integration` run without any API keys.

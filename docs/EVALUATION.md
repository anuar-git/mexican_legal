# Evaluation Methodology

How we measure quality, why each metric was chosen, and how to interpret
results for the Legal Intelligence Engine.

---

## Design Philosophy

Legal RAG is harder to evaluate than general-purpose RAG.

**The grounding problem.** A model that says "Artículo 123 establece prisión de 8
a 20 años" is correct for the CDMX code but wrong for the Federal Penal Code —
the article numbers differ between the two. Standard RAGAS metrics cannot catch
this because they compare against a ground-truth reference answer written in
natural language; they do not know which article numbers the retrieved chunks
contain.

**The refusal problem.** A system that confidently answers a tax-law question by
hallucinating penal code articles is strictly worse than one that says "I cannot
answer this from the Mexican Penal Code." Standard QA metrics reward a fluent
wrong answer over a correct refusal.

These two gaps motivate the custom metrics below.

---

## Test Set

**File:** `data/evaluation/golden_test_set.json`
**Version:** 1.0 (75 questions, created 2026-03-28)
**Source document:** `data/raw/codigo_penal_cdmx_31225.pdf`

### Composition

| Dimension | Distribution |
|---|---|
| `factual_lookup` | 25 questions — direct article lookups |
| `comparison` | 15 questions — side-by-side analysis of two or more articles |
| `multi_article` | 10 questions — synthesis across 3–5 articles |
| `conditional` | 10 questions — questions whose answer depends on circumstance |
| `adversarial_oos` | 10 questions — topics outside the penal code entirely |
| `adversarial_tricky` | 5 questions — designed to trigger specific failure modes |

| Difficulty | Count |
|---|---|
| `easy` | 15 — single article, direct definition |
| `medium` | 24 — requires reading 2 articles or understanding context |
| `hard` | 21 — multi-article synthesis, nuanced legal distinction |
| `adversarial` | 15 — OOS or hallucination traps |

**Legal categories** (15 total): homicidio, lesiones, sexual, libertad, patrimonio,
robo, fraude, extorsion, corrupcion, salud, encubrimiento, informatica, armas,
general, out\_of\_scope.

### Question design principles

**`factual_lookup`** — exact article reference lookup. A correct answer must
include the article number and the key legal elements (penalty range, definition
components). Example: "¿Cuál es la pena por homicidio simple?" → must cite
Artículos 302 and 307 and state the 12–24 year range.

**`comparison`** — tests whether the model can distinguish similar but legally
distinct concepts. Example: "¿Cuál es la diferencia entre robo y abuso de
confianza?" → requires synthesising Articles 367 and 382; the tenencia vs.
apoderamiento distinction is the key signal.

**`multi_article`** — tests breadth retrieval. Example: "¿Cuáles son las
calificativas del homicidio?" → correct answer requires Articles 315–319 all to
be retrieved and cited.

**`conditional`** — tests handling of legal nuance. Example: "¿Qué pena corresponde
si el homicidio se comete con premeditación y ventaja simultáneamente?" → trick
question: multiple calificativas do not compound the penalty; they all lead to the
same qualified type. The system should explain this rather than add penalties.

**`adversarial_oos`** — questions drawn from tax law, civil law, constitutional
law, and commercial law — none of which appear in the penal code. A correct
response declines to answer or explicitly states the topic is outside the document.

**`adversarial_tricky`** — questions crafted to trigger known failure modes.
Example: "¿Cuál es la pena por robo de datos personales o información digital?"
— designed to make the model hallucinate a non-existent "robo digital" article.
The grounded answer references Article 211 BIS (computer access crimes) while
explaining it covers access, not theft.

---

## Metrics

### Custom Metrics (no LLM judge required)

These are computed deterministically from article numbers in the retrieved chunks
and the generated answer. They run offline with no API calls.

#### Citation Accuracy

```
n_grounded_citations / n_total_citations
```

A citation is *grounded* when its article number matches either the `article_number`
metadata field or the text of at least one retrieved chunk. Returns 1.0 when the
answer contains no citations — the model correctly produced no article references,
which is appropriate for out-of-scope questions.

**Why this matters:** Faithfulness (RAGAS) catches when the *content* of an answer
diverges from the context. Citation accuracy catches when the *article number* cited
is absent from the retrieved chunks — a more specific failure mode that matters
enormously in legal contexts, where article numbers are the canonical reference
point for practitioners.

**Baseline (run ef061dbb):** 0.78 across 75 questions.

**When it is low:** Either the retriever did not surface the relevant article (check
retrieval hit rate), or the generator cited an article it invented from pre-training
knowledge rather than from context (hallucination).

#### Retrieval Hit Rate

```
n_expected_articles_found_in_retrieved_chunks / n_expected_articles
```

Checks whether the articles specified in `expected_articles` (ground truth from
the golden test set) appear in the top-5 reranked chunks — either in the
`article_number` metadata field or in the chunk text. Returns 1.0 vacuously for
out-of-scope questions (where no retrieval is expected).

**Why this matters:** Citation accuracy and faithfulness both measure what happened
*after* retrieval. Hit rate attributes failures to the correct stage: if hit rate
is low, the retriever is the problem; if hit rate is high but citation accuracy is
low, the generator is the problem.

**Baseline (run ef061dbb):** 0.15 — this is the known weak point of the system.
The expected articles in the test set are drawn from the Federal Penal Code article
numbering scheme (302, 307, etc.) but the source document is the CDMX code, which
uses different article numbers (123, 125, etc.). The mismatch inflates the miss
rate; when evaluated on CDMX-internal lookups the effective hit rate is higher.

#### Hallucination Rate

Binary flag per result: True if any cited article is absent from the retrieved
chunks. Aggregated as a fraction across all successful results.

**Baseline (run ef061dbb):** 29.3% — 22 of 75 questions triggered at least one
hallucinated citation. See the failure analysis below for the breakdown by category.

**The tuning knob:** Raising `top_k_final` from 5 to 10 in `Retriever.__init__`
increases the context window available to the generator and reduces the chance
that a correctly retrieved but lower-ranked article is cut off. The trade-off is
~0.5–1.0 s additional latency per query (one more cross-encoder forward pass per
chunk).

#### OOS Refusal Rate

For `adversarial_oos` questions only. A correct refusal is detected heuristically:
the answer contains at least one Spanish-language refusal phrase (e.g. "no puedo",
"fuera del alcance", "código fiscal") OR contains zero citations.

**Baseline (run ef061dbb):** 80% — 8 of 10 OOS questions correctly declined. The
two failures answered with penal code articles that were superficially related but
not actually responsive to the off-topic question.

#### Article Coverage Rate

```
n_expected_articles_actually_cited / n_expected_articles
```

Distinct from hit rate (retrieved?) and citation accuracy (grounded?). This asks:
did the model *use* the relevant articles it retrieved? A low hit rate with a high
coverage rate is a contradiction — verify the data. A high hit rate with a low
coverage rate means the retriever succeeded but the generator failed to synthesise
the relevant articles into its answer.

### RAGAS Metrics (LLM judge, requires API call per question)

RAGAS uses Claude Haiku as a judge LLM to score each response.

#### Faithfulness (primary RAGAS signal)

Measures whether every factual claim in the answer is grounded in the retrieved
context. Claude Haiku decomposes the answer into atomic claims, then checks each
claim against the retrieved chunks.

**Baseline:** 0.84 — the strongest RAGAS score and the most reliable signal for
this corpus. It is least sensitive to the Spanish-language evaluator bias described
below.

**Interpretation:** Faithfulness > 0.80 is a reasonable production threshold for
a legal Q&A system. Below 0.70, the system is making too many claims unsupported
by retrieved context.

#### Answer Relevance and Context Recall

**Baseline:** 0.13 and 0.11 — both score significantly lower than expected.

**Known bias:** RAGAS evaluates in English internally. The judge LLM translates or
reasons about the Spanish legal text before scoring. Reference answers in
`expected_answer` use article numbers from the Federal Penal Code (Articles 302,
307) while the retrieved context uses CDMX numbering (Articles 123, 125). The judge
correctly sees these as mismatches and penalises recall accordingly.

This is an evaluation artefact, not a pipeline defect. It affects answer relevance
and context recall (both reference-based) but not faithfulness (context-only).
Future work: write a bilingual RAGAS evaluator or update `expected_articles` to use
CDMX article numbers.

---

## Baseline Results (Run ef061dbb, 2026-03-29)

75 questions, 0 errors, all 75 successful.

### Aggregate

| Metric | Score |
|---|---|
| Faithfulness (RAGAS) | 0.84 |
| Citation Accuracy | 0.78 |
| OOS Refusal Rate | 80% |
| Hallucination Rate | 29% |
| p50 Latency | 10.5 s |
| p90 Latency | 12.3 s |
| p95 Latency | 12.5 s |
| Mean Retrieval Time | 6.6 s |
| Mean Generation Time | 3.9 s |

### By Question Type

| Type | n | Citation Acc | Hit Rate | Hallucination |
|---|---|---|---|---|
| `factual_lookup` | 25 | 0.81 | 0.00 | 24% |
| `comparison` | 15 | 0.83 | 0.03 | 20% |
| `multi_article` | 10 | 0.87 | 0.00 | 20% |
| `conditional` | 10 | 0.55 | 0.00 | 60% |
| `adversarial_oos` | 10 | 0.80 | 1.00 | 30% |
| `adversarial_tricky` | 5 | 0.77 | 0.20 | 40% |

**`conditional` questions are the hardest failure mode** — 60% hallucination rate.
These questions require the model to reason about legal conditions
(e.g. concurrent calificativas) and the retriever rarely surfaces all the necessary
articles in the top-5 window. The generator compensates by citing articles from
memory, triggering the hallucination flag.

### By Legal Category (citation accuracy, worst to best)

| Category | n | Citation Acc | Hallucination |
|---|---|---|---|
| `extorsion` | 3 | 0.11 | 100% |
| `armas` | 1 | 0.33 | 100% |
| `salud` | 3 | 0.50 | 67% |
| `encubrimiento` | 2 | 0.50 | 50% |
| `fraude` | 3 | 0.67 | 33% |
| `informatica` | 2 | 0.75 | 50% |
| `patrimonio` | 6 | 0.75 | 33% |
| `robo` | 4 | 0.79 | 50% |
| `out_of_scope` | 10 | 0.80 | 30% |
| `libertad` | 8 | 0.81 | 25% |
| `lesiones` | 9 | 0.83 | 22% |
| `homicidio` | 8 | 0.88 | 12% |
| `sexual` | 9 | 0.94 | 11% |
| `corrupcion` | 4 | 1.00 | 0% |
| `general` | 3 | 1.00 | 0% |

Homicidio, sexual, and corrupcion are the strongest categories — all have many
questions in the test set and are well-represented in the source document. Extorsion
and armas are the weakest — both have sparse representation in the CDMX code and
few training examples in the golden set.

### Worst-Performing Questions

| ID | Question | Type | Citation Acc | Root Cause |
|---|---|---|---|---|
| q015 | ¿Qué es el encubrimiento? | factual_lookup | 0.00 | Retriever missed the encubrimiento articles entirely |
| q018 | ¿Cuál es la pena por extorsión? | factual_lookup | 0.00 | Article sparse in CDMX code; model cited Federal code articles |
| q022 | ¿Qué es el secuestro exprés? | factual_lookup | 0.00 | Term "exprés" not in CDMX code; model hallucinated a definition |
| q030 | Diferencia entre fraude y robo | comparison | 0.00 | Both articles retrieved but generator confounded them |
| q039 | Homicidio vs lesiones mortales | comparison | 0.00 | Requires 5+ articles; top-5 window too small |

---

## Latency Breakdown

The pipeline is IO-bound in two places:

1. **Query expansion** — one Anthropic API call to generate 3 alternative phrasings.
   Runs before the Pinecone search. Cannot be parallelised with the search without
   pre-warming the query (which would require a different pipeline architecture).

2. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` runs on
   CPU (the deployment host does not have a CUDA-capable GPU). With 20 candidates
   × 3 query variants = up to 60 pairs to score, this is the dominant CPU cost.
   Upgrading to a GPU host would reduce reranking time from ~1–2 s to ~0.1 s.

The ~6.6 s mean retrieval time breaks down roughly as:
- Query expansion (Anthropic API): ~1.5 s
- Three parallel Pinecone dense searches: ~2.0 s
- Cross-encoder reranking (CPU): ~3.0 s

The ~3.9 s mean generation time is the Anthropic streaming call to Claude Haiku
with the formatted context and system prompt.

---

## Running an Evaluation

```bash
# Full evaluation — 75 questions, RAGAS scoring (~13 min)
python scripts/evaluate.py

# Quick smoke test — 5 questions, skip RAGAS (~1 min)
python scripts/evaluate.py --limit 5 --no-ragas

# Single question drill-down
python scripts/evaluate.py --question-id q018

# Filter by category or type
python scripts/evaluate.py --category extorsion encubrimiento
python scripts/evaluate.py --question-type conditional

# Compare two runs
python scripts/evaluate.py --compare results/eval_ef061dbb.json results/my_run.json

# Tune retrieval depth
python scripts/evaluate.py --top-k-final 10 --output results/topk10.json
```

Evaluation results are saved as:
- `results/eval_{run_id}.json` — full machine-readable run data
- `results/reports/eval_{run_id}.md` — human-readable markdown report

### Cost estimate

Each full evaluation run makes approximately:
- 75 Anthropic API calls for query expansion (~$0.01)
- 75 Anthropic API calls for generation (~$0.04)
- 75 × 4 = 300 RAGAS judge calls (~$0.15 with --no-ragas skipped)

Total: ~$0.05 without RAGAS, ~$0.20 with RAGAS.

---

## Improvement Roadmap

**Short-term (configuration changes, no code required):**

- Raise `top_k_final` from 5 → 10: expected to reduce hallucination rate by
  10–15 percentage points at a cost of ~1 s additional latency.
- Add BM25 sparse search alongside the dense search (true hybrid): Pinecone
  supports this natively; expected to improve `extorsion` and `armas` categories
  where exact term matching matters more than semantic similarity.

**Medium-term (requires re-ingestion):**

- Update `expected_articles` in the golden test set to use CDMX article numbers
  (not Federal code numbers). This will make retrieval hit rate meaningful and
  expose the true retrieval weakness.
- Add a legal-domain embedding model fine-tuned on Spanish statutory text.
  `llama-text-embed-v2` is a general-purpose model; a domain-specific model
  should improve recall for legal terms that general embeddings underweight.

**Long-term:**

- Semantic chunking strategy that respects article + section boundaries more
  aggressively (the chunker already seeds on `ARTÍCULO` separators, but pages
  that span multiple short articles still get combined).
- Multi-document retrieval: extend ingestion to include Código Nacional de
  Procedimientos Penales and related federal legislation, then add a routing
  layer to direct OOS questions to the appropriate corpus rather than refusing.

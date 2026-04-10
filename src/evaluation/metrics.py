"""Custom evaluation metrics for the Mexican Penal Code RAG system.

Implements domain-specific metrics that RAGAS does not cover.  Four core
metrics match the section specification; the module extends each with
richer internals and adds three additional metrics that demonstrate
understanding of the legal domain:

Core (spec):
    citation_accuracy        — cited articles grounded in retrieved context
    retrieval_hit_rate       — expected articles present in retrieved chunks
    detect_hallucination     — any citation absent from retrieved context
    latency_breakdown        — per-stage timing breakdown

Extended beyond spec:
    article_coverage_rate    — expected articles actually cited in the answer
                               (measures whether the model uses what it retrieved)
    oos_correctly_refused    — adversarial OOS questions answered with refusal
    score_result             — all custom metrics for one EvalResult
    score_all                — apply score_result to a list
    apply_to_run             — write scores back into EvalResult fields in an EvalRun
    aggregate                — mean/p95 statistics across all results
    stratified_breakdown     — metrics grouped by difficulty / question_type / category
    worst_performers         — N results with the lowest score on a given metric
    best_performers          — N results with the highest score on a given metric
"""

from __future__ import annotations

import re
import statistics

from pydantic import BaseModel

from src.evaluation.evaluator import EvalResult, EvalRun

# ---------------------------------------------------------------------------
# Article number normalisation
# ---------------------------------------------------------------------------

# Matches "Artículo 123", "artículo 13 BIS", "Art. 266 Bis", etc.
_ARTICLE_TEXT_RE = re.compile(
    r"[Aa]rt[ií]culo[s]?\s+([\d]+(?:\s+(?:BIS|TER|QUATER|QUINTUS))?)",
    re.IGNORECASE,
)

# Refusal phrases in Spanish — used by oos_correctly_refused()
_REFUSAL_RE = re.compile(
    r"no\s+(puedo|encuentro|está|está\s+regulad|regula|contempla|es\s+materia|"
    r"se\s+encuentra|tengo\s+información|aborda)|"
    r"fuera\s+del\s+(alcance|código)|"
    r"código\s+(fiscal|civil|de\s+procedimientos|de\s+comercio)|"
    r"ley\s+federal\s+de|"
    r"no\s+es\s+parte\s+del|"
    r"no\s+forma\s+parte|"
    r"esta\s+pregunta\s+no|"
    r"el\s+código\s+penal\s+no",
    re.IGNORECASE,
)


def _norm_article(s: str) -> str:
    """Normalise an article number to uppercase with collapsed whitespace.

    Examples:
        "13 bis"  → "13 BIS"
        "266BIS"  → "266 BIS"   (no space before BIS in some sources)
        " 302 "   → "302"
    """
    s = s.strip().upper()
    # Insert a space before BIS/TER/QUATER if missing: "266BIS" → "266 BIS"
    s = re.sub(r"(\d)(BIS|TER|QUATER|QUINTUS)", r"\1 \2", s)
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", s)


def _retrieved_article_set(retrieved_chunks: list[dict]) -> set[str]:
    """Build a normalised set of article numbers from retrieved chunk data.

    Checks two sources (in order of reliability):
    1. ``chunk["metadata"]["article_number"]`` — set during ingestion.
    2. Regex scan of ``chunk["text"]`` — catches continuation chunks whose
       metadata may not carry the article number.

    Args:
        retrieved_chunks: Serialised chunk dicts from ``EvalResult``.

    Returns:
        Set of normalised article number strings (e.g. {"302", "307", "13 BIS"}).
    """
    articles: set[str] = set()
    for chunk in retrieved_chunks:
        meta_art = chunk.get("metadata", {}).get("article_number")
        if meta_art:
            articles.add(_norm_article(str(meta_art)))
        for match in _ARTICLE_TEXT_RE.finditer(chunk.get("text", "")):
            articles.add(_norm_article(match.group(1)))
    return articles


# ---------------------------------------------------------------------------
# Core metrics (spec)
# ---------------------------------------------------------------------------


def citation_accuracy(eval_result: EvalResult) -> float:
    """What fraction of cited articles exist in the retrieved context?

    A citation is considered *grounded* when its normalised article number
    matches the metadata or text of at least one retrieved chunk.  Returns
    1.0 when the answer contains no citations — the model correctly produced
    no article references (appropriate for out-of-scope questions).

    Args:
        eval_result: A single EvalResult from ``Evaluator.run()``.

    Returns:
        Float in [0, 1].  1.0 = all citations grounded; 0.0 = all hallucinated.
    """
    if not eval_result.actual_citations:
        return 1.0

    retrieved = _retrieved_article_set(eval_result.retrieved_chunks)
    cited_norm = [_norm_article(c) for c in eval_result.actual_citations]
    n_valid = sum(1 for c in cited_norm if c in retrieved)
    return round(n_valid / len(cited_norm), 4)


def retrieval_hit_rate(eval_result: EvalResult) -> float:
    """What fraction of expected articles were found in the retrieved chunks?

    Measures retriever completeness for this question — did the dense
    search + reranking pipeline surface the relevant legal articles?
    Returns 1.0 vacuously when ``expected_articles`` is empty (adversarial /
    out-of-scope questions where no retrieval is expected).

    Args:
        eval_result: A single EvalResult from ``Evaluator.run()``.

    Returns:
        Float in [0, 1].
    """
    if not eval_result.expected_articles:
        return 1.0

    retrieved = _retrieved_article_set(eval_result.retrieved_chunks)
    expected_norm = [_norm_article(a) for a in eval_result.expected_articles]
    n_found = sum(1 for a in expected_norm if a in retrieved)
    return round(n_found / len(expected_norm), 4)


def detect_hallucination(eval_result: EvalResult) -> bool:
    """Does the answer cite any article NOT present in the retrieved context?

    A single ungrounded citation is enough to flip the flag.  When there are
    no citations the answer is considered hallucination-free (returns False).

    Args:
        eval_result: A single EvalResult from ``Evaluator.run()``.

    Returns:
        True if any cited article is absent from the retrieved chunks.
    """
    if not eval_result.actual_citations:
        return False

    retrieved = _retrieved_article_set(eval_result.retrieved_chunks)
    for citation in eval_result.actual_citations:
        if _norm_article(citation) not in retrieved:
            return True
    return False


def latency_breakdown(eval_result: EvalResult) -> dict[str, float]:
    """Break down where time is spent across pipeline stages.

    Computes the overhead attributed to query expansion and reranking as the
    residual between ``total_time_ms`` and the sum of ``retrieval_time_ms``
    and ``generation_time_ms``.

    Args:
        eval_result: A single EvalResult from ``Evaluator.run()``.

    Returns:
        Dict with absolute milliseconds and percentage share per stage.
        Keys: ``retrieval_ms``, ``generation_ms``, ``overhead_ms``,
        ``total_ms``, ``retrieval_pct``, ``generation_pct``, ``overhead_pct``.
    """
    total = eval_result.total_time_ms
    ret = eval_result.retrieval_time_ms
    gen = eval_result.generation_time_ms
    overhead = max(0.0, total - ret - gen)  # expansion + reranking residual

    if total == 0:
        return {
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "overhead_ms": 0.0,
            "total_ms": 0.0,
            "retrieval_pct": 0.0,
            "generation_pct": 0.0,
            "overhead_pct": 0.0,
        }

    return {
        "retrieval_ms": round(ret, 1),
        "generation_ms": round(gen, 1),
        "overhead_ms": round(overhead, 1),
        "total_ms": round(total, 1),
        "retrieval_pct": round(ret / total * 100, 1),
        "generation_pct": round(gen / total * 100, 1),
        "overhead_pct": round(overhead / total * 100, 1),
    }


# ---------------------------------------------------------------------------
# Extended domain metrics
# ---------------------------------------------------------------------------


def article_coverage_rate(eval_result: EvalResult) -> float:
    """What fraction of expected articles did the model actually cite?

    Distinct from ``retrieval_hit_rate`` (which checks whether articles were
    *retrieved*) and from ``citation_accuracy`` (which checks whether cited
    articles are *grounded*).  This metric asks: did the model use the
    retrieved context to cite the articles it was supposed to cite?

    Low retrieval hit rate → retriever problem.
    High retrieval hit rate + low article coverage → generation problem.
    Low citation accuracy → hallucination problem.

    Returns 1.0 vacuously when ``expected_articles`` is empty.

    Args:
        eval_result: A single EvalResult from ``Evaluator.run()``.

    Returns:
        Float in [0, 1].
    """
    if not eval_result.expected_articles:
        return 1.0

    expected_norm = {_norm_article(a) for a in eval_result.expected_articles}
    cited_norm = {_norm_article(c) for c in eval_result.actual_citations}
    covered = expected_norm & cited_norm
    return round(len(covered) / len(expected_norm), 4)


def oos_correctly_refused(eval_result: EvalResult) -> bool | None:
    """Did the model correctly refuse to answer an out-of-scope question?

    Only meaningful for ``adversarial_oos`` questions.  Returns ``None`` for
    all other question types so aggregate functions can filter it out.

    "Correct refusal" is detected heuristically: the answer must contain at
    least one Spanish-language refusal phrase (e.g. "no puedo", "código fiscal",
    "fuera del alcance") OR contain zero citations.  Both are weak signals;
    the conjunction of both (refusal phrase AND no citations) is a strong signal.

    Args:
        eval_result: A single EvalResult from ``Evaluator.run()``.

    Returns:
        True = correct refusal, False = incorrectly answered, None = N/A.
    """
    if eval_result.question_type != "adversarial_oos":
        return None
    if eval_result.error:
        return None

    answer = eval_result.actual_answer
    has_refusal_phrase = bool(_REFUSAL_RE.search(answer))
    has_no_citations = len(eval_result.actual_citations) == 0
    return has_refusal_phrase or has_no_citations


# ---------------------------------------------------------------------------
# Per-result scoring
# ---------------------------------------------------------------------------


class ResultMetrics(BaseModel):
    """All custom metric scores for a single EvalResult.

    Attributes:
        question_id:           Matches EvalResult.question_id.
        citation_accuracy:     Fraction of cited articles grounded in context.
        retrieval_hit_rate:    Fraction of expected articles retrieved.
        article_coverage_rate: Fraction of expected articles actually cited.
        hallucination_detected: True if any ungrounded citation found.
        oos_correctly_refused: Refusal signal for OOS questions; None otherwise.
        latency:               Per-stage timing dict.
        retrieved_article_set: Normalised article numbers in retrieved chunks.
        missing_articles:      Expected articles not found in retrieved chunks.
        hallucinated_articles: Cited articles not found in retrieved chunks.
    """

    question_id: str
    citation_accuracy: float
    retrieval_hit_rate: float
    article_coverage_rate: float
    hallucination_detected: bool
    oos_correctly_refused: bool | None
    latency: dict[str, float]
    retrieved_article_set: list[str]
    missing_articles: list[str]
    hallucinated_articles: list[str]


def score_result(eval_result: EvalResult) -> ResultMetrics:
    """Compute all custom metrics for a single EvalResult.

    Args:
        eval_result: A single EvalResult (error results are accepted; metrics
                     default to neutral values when ``error`` is not None).

    Returns:
        ResultMetrics with all fields populated.
    """
    if eval_result.error:
        return ResultMetrics(
            question_id=eval_result.question_id,
            citation_accuracy=0.0,
            retrieval_hit_rate=0.0,
            article_coverage_rate=0.0,
            hallucination_detected=False,
            oos_correctly_refused=None,
            latency=latency_breakdown(eval_result),
            retrieved_article_set=[],
            missing_articles=eval_result.expected_articles,
            hallucinated_articles=[],
        )

    retrieved = _retrieved_article_set(eval_result.retrieved_chunks)

    # Missing: expected articles absent from retrieved chunks
    expected_norm = [_norm_article(a) for a in eval_result.expected_articles]
    missing = [a for a in expected_norm if a not in retrieved]

    # Hallucinated: cited articles absent from retrieved chunks
    cited_norm = [_norm_article(c) for c in eval_result.actual_citations]
    hallucinated = [c for c in cited_norm if c not in retrieved]

    return ResultMetrics(
        question_id=eval_result.question_id,
        citation_accuracy=citation_accuracy(eval_result),
        retrieval_hit_rate=retrieval_hit_rate(eval_result),
        article_coverage_rate=article_coverage_rate(eval_result),
        hallucination_detected=detect_hallucination(eval_result),
        oos_correctly_refused=oos_correctly_refused(eval_result),
        latency=latency_breakdown(eval_result),
        retrieved_article_set=sorted(retrieved),
        missing_articles=missing,
        hallucinated_articles=hallucinated,
    )


def score_all(results: list[EvalResult]) -> list[ResultMetrics]:
    """Apply ``score_result`` to every EvalResult in a list.

    Args:
        results: Output of ``Evaluator.run().results``.

    Returns:
        List of ResultMetrics in the same order as ``results``.
    """
    return [score_result(r) for r in results]


def apply_to_run(eval_run: EvalRun) -> EvalRun:
    """Write custom metric scores back into EvalResult fields in an EvalRun.

    Useful when loading a saved EvalRun from disk and wanting to refresh the
    custom metric fields (e.g. after normalisation logic changes).  Mutates
    the EvalRun in-place and refreshes aggregate_scores.

    Args:
        eval_run: An EvalRun loaded from JSON or returned by Evaluator.run().

    Returns:
        The same EvalRun with updated custom metric fields and aggregates.
    """
    for result in eval_run.results:
        if result.error:
            continue
        result.citation_accuracy = citation_accuracy(result)
        result.retrieval_hit_rate = retrieval_hit_rate(result)
        result.hallucination_detected = detect_hallucination(result)
    eval_run.recalculate_aggregates()
    return eval_run


# ---------------------------------------------------------------------------
# Aggregate & stratified analysis
# ---------------------------------------------------------------------------


class AggregateMetrics(BaseModel):
    """Aggregated custom metric statistics across an entire EvalRun.

    Attributes:
        n_total:                  Total questions evaluated.
        n_successful:             Questions with no pipeline error.
        n_errors:                 Questions that raised a pipeline exception.
        mean_citation_accuracy:   Mean citation accuracy across successful results.
        mean_retrieval_hit_rate:  Mean retrieval hit rate.
        mean_article_coverage_rate: Mean article coverage rate.
        hallucination_rate:       Fraction of successful results with hallucination.
        n_hallucinated:           Count of results with hallucination_detected=True.
        oos_refusal_rate:         Fraction of OOS questions correctly refused.
        n_oos_questions:          Count of adversarial_oos questions.
        n_oos_refused:            Count correctly refused.
        mean_retrieval_ms:        Mean retrieval stage latency.
        mean_generation_ms:       Mean generation stage latency.
        mean_total_ms:            Mean end-to-end latency.
        p95_total_ms:             95th percentile end-to-end latency.
        mean_faithfulness:        Mean RAGAS faithfulness (None if not scored).
        mean_answer_relevance:    Mean RAGAS answer relevance.
        mean_context_relevance:   Mean RAGAS context relevance.
        mean_context_recall:      Mean RAGAS context recall.
    """

    n_total: int
    n_successful: int
    n_errors: int

    mean_citation_accuracy: float | None
    mean_retrieval_hit_rate: float | None
    mean_article_coverage_rate: float | None
    hallucination_rate: float | None
    n_hallucinated: int
    oos_refusal_rate: float | None
    n_oos_questions: int
    n_oos_refused: int

    mean_retrieval_ms: float | None
    mean_generation_ms: float | None
    mean_total_ms: float | None
    p95_total_ms: float | None

    mean_faithfulness: float | None
    mean_answer_relevance: float | None
    mean_context_relevance: float | None
    mean_context_recall: float | None


def _mean(values: list[float]) -> float | None:
    """Return the mean of a non-empty list, or None."""
    return round(statistics.mean(values), 4) if values else None


def _p95(values: list[float]) -> float | None:
    """Return the 95th percentile of a non-empty list, or None."""
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * 0.95) - 1)
    return round(sorted_vals[idx], 1)


def aggregate(results: list[EvalResult]) -> AggregateMetrics:
    """Compute aggregate statistics across all results.

    Computes custom metrics on-the-fly from EvalResult data, so it gives
    correct numbers even if ``citation_accuracy`` etc. fields on EvalResult
    are stale or None.  RAGAS fields are read directly from EvalResult and
    averaged over non-None values.

    Args:
        results: All EvalResult objects from an EvalRun.

    Returns:
        AggregateMetrics with means, percentiles, and counts.
    """
    ok = [r for r in results if r.error is None]
    n_total = len(results)
    n_ok = len(ok)

    # Custom metrics — recomputed from scratch
    cit_acc = [citation_accuracy(r) for r in ok]
    ret_hr = [retrieval_hit_rate(r) for r in ok]
    art_cov = [article_coverage_rate(r) for r in ok]
    halluc_flags = [detect_hallucination(r) for r in ok]
    n_hallucinated = sum(halluc_flags)

    # OOS refusal
    oos_results = [r for r in ok if r.question_type == "adversarial_oos"]
    oos_refused_flags = [oos_correctly_refused(r) for r in oos_results]
    n_oos_refused = sum(1 for f in oos_refused_flags if f is True)
    oos_rate = round(n_oos_refused / len(oos_results), 4) if oos_results else None

    # Latency
    ret_ms = [r.retrieval_time_ms for r in ok]
    gen_ms = [r.generation_time_ms for r in ok]
    total_ms = [r.total_time_ms for r in ok]

    # RAGAS fields (already on EvalResult, may be None)
    def _ragas_mean(field: str) -> float | None:
        vals = [getattr(r, field) for r in ok if getattr(r, field) is not None]
        return _mean(vals)

    return AggregateMetrics(
        n_total=n_total,
        n_successful=n_ok,
        n_errors=n_total - n_ok,
        mean_citation_accuracy=_mean(cit_acc),
        mean_retrieval_hit_rate=_mean(ret_hr),
        mean_article_coverage_rate=_mean(art_cov),
        hallucination_rate=round(n_hallucinated / n_ok, 4) if n_ok else None,
        n_hallucinated=n_hallucinated,
        oos_refusal_rate=oos_rate,
        n_oos_questions=len(oos_results),
        n_oos_refused=n_oos_refused,
        mean_retrieval_ms=_mean(ret_ms),
        mean_generation_ms=_mean(gen_ms),
        mean_total_ms=_mean(total_ms),
        p95_total_ms=_p95(total_ms),
        mean_faithfulness=_ragas_mean("faithfulness"),
        mean_answer_relevance=_ragas_mean("answer_relevance"),
        mean_context_relevance=_ragas_mean("context_relevance"),
        mean_context_recall=_ragas_mean("context_recall"),
    )


def _stratum_summary(results: list[EvalResult]) -> dict:
    """Compute a concise metric summary dict for a subset of results."""
    ok = [r for r in results if r.error is None]
    if not ok:
        return {"n": 0, "n_errors": len(results)}

    def _ragas_mean(field: str) -> float | None:
        vals = [getattr(r, field) for r in ok if getattr(r, field) is not None]
        return _mean(vals)

    halluc = [detect_hallucination(r) for r in ok]
    return {
        "n": len(ok),
        "n_errors": len(results) - len(ok),
        "citation_accuracy": _mean([citation_accuracy(r) for r in ok]),
        "retrieval_hit_rate": _mean([retrieval_hit_rate(r) for r in ok]),
        "article_coverage_rate": _mean([article_coverage_rate(r) for r in ok]),
        "hallucination_rate": round(sum(halluc) / len(ok), 4),
        "mean_total_ms": _mean([r.total_time_ms for r in ok]),
        "faithfulness": _ragas_mean("faithfulness"),
        "context_recall": _ragas_mean("context_recall"),
    }


def stratified_breakdown(results: list[EvalResult]) -> dict[str, dict]:
    """Group metric summaries by difficulty, question_type, and category.

    Each leaf dict contains the same keys as ``_stratum_summary``.  Groups
    with zero successful results still appear with ``{"n": 0, "n_errors": k}``.

    Args:
        results: All EvalResult objects from an EvalRun.

    Returns:
        Dict with three top-level keys::

            {
                "by_difficulty":     {"easy": {...}, "medium": {...}, ...},
                "by_question_type":  {"factual_lookup": {...}, ...},
                "by_category":       {"homicidio": {...}, ...},
            }
    """

    def _group(key: str) -> dict:
        groups: dict[str, list[EvalResult]] = {}
        for r in results:
            groups.setdefault(getattr(r, key), []).append(r)
        return {label: _stratum_summary(group) for label, group in sorted(groups.items())}

    return {
        "by_difficulty": _group("difficulty"),
        "by_question_type": _group("question_type"),
        "by_category": _group("category"),
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def worst_performers(
    results: list[EvalResult],
    metric: str = "citation_accuracy",
    n: int = 5,
) -> list[EvalResult]:
    """Return the N results with the lowest score for a given metric.

    Successful results only (errors are excluded).  The metric is
    re-computed on-the-fly for ``citation_accuracy``, ``retrieval_hit_rate``,
    and ``article_coverage_rate``; for other fields (RAGAS scores) the value
    stored on ``EvalResult`` is used directly (may be None, sorted to bottom).

    Args:
        results: EvalResult list from an EvalRun.
        metric:  Any numeric field of EvalResult or one of the three custom
                 function names (``citation_accuracy``, ``retrieval_hit_rate``,
                 ``article_coverage_rate``).
        n:       Number of results to return.

    Returns:
        Up to N EvalResult objects, ascending order (worst first).
    """
    _COMPUTED = {
        "citation_accuracy": citation_accuracy,
        "retrieval_hit_rate": retrieval_hit_rate,
        "article_coverage_rate": article_coverage_rate,
    }
    ok = [r for r in results if r.error is None]

    def _score(r: EvalResult) -> float:
        if metric in _COMPUTED:
            return _COMPUTED[metric](r)
        val = getattr(r, metric, None)
        return val if val is not None else float("inf")  # None sorts last (worst)

    return sorted(ok, key=_score)[:n]


def best_performers(
    results: list[EvalResult],
    metric: str = "citation_accuracy",
    n: int = 5,
) -> list[EvalResult]:
    """Return the N results with the highest score for a given metric.

    Args:
        results: EvalResult list from an EvalRun.
        metric:  Same accepted values as ``worst_performers``.
        n:       Number of results to return.

    Returns:
        Up to N EvalResult objects, descending order (best first).
    """
    _COMPUTED = {
        "citation_accuracy": citation_accuracy,
        "retrieval_hit_rate": retrieval_hit_rate,
        "article_coverage_rate": article_coverage_rate,
    }
    ok = [r for r in results if r.error is None]

    def _score(r: EvalResult) -> float:
        if metric in _COMPUTED:
            return _COMPUTED[metric](r)
        val = getattr(r, metric, None)
        return val if val is not None else float("-inf")  # None sorts last

    return sorted(ok, key=_score, reverse=True)[:n]

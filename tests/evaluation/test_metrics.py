"""Unit tests for src/evaluation/metrics.py.

Tests cover all public functions:
  citation_accuracy, retrieval_hit_rate, detect_hallucination,
  latency_breakdown, article_coverage_rate, oos_correctly_refused,
  score_result, aggregate, stratified_breakdown, worst_performers,
  best_performers.
"""

from __future__ import annotations

import pytest

from src.evaluation.evaluator import EvalResult
from src.evaluation.metrics import (
    AggregateMetrics,
    ResultMetrics,
    aggregate,
    article_coverage_rate,
    best_performers,
    citation_accuracy,
    detect_hallucination,
    latency_breakdown,
    oos_correctly_refused,
    retrieval_hit_rate,
    score_result,
    stratified_breakdown,
    worst_performers,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    question_id: str = "q001",
    question: str = "¿Cuál es la pena por homicidio?",
    category: str = "homicidio",
    difficulty: str = "easy",
    question_type: str = "factual_lookup",
    expected_articles: list[str] | None = None,
    actual_citations: list[str] | None = None,
    retrieved_chunks: list[dict] | None = None,
    actual_answer: str = "El Artículo 123 establece prisión de 8 a 20 años.",
    retrieval_ms: float = 5000.0,
    generation_ms: float = 2000.0,
    total_ms: float = 7500.0,
    error: str | None = None,
) -> EvalResult:
    return EvalResult(
        question_id=question_id,
        question=question,
        category=category,
        difficulty=difficulty,
        question_type=question_type,
        expected_answer="",
        expected_articles=expected_articles or [],
        actual_answer=actual_answer,
        actual_citations=actual_citations or [],
        retrieved_chunks=retrieved_chunks or [],
        retrieval_time_ms=retrieval_ms,
        generation_time_ms=generation_ms,
        total_time_ms=total_ms,
        error=error,
    )


def _chunk(article: str, text: str = "") -> dict:
    return {
        "chunk_id": f"doc-{article}",
        "text": text or f"ARTÍCULO {article}. Texto de prueba.",
        "metadata": {"article_number": article},
        "similarity_score": 0.8,
        "rerank_score": 3.0,
    }


# ---------------------------------------------------------------------------
# citation_accuracy
# ---------------------------------------------------------------------------


class TestCitationAccuracy:
    def test_all_grounded(self) -> None:
        r = _make_result(
            actual_citations=["123"],
            retrieved_chunks=[_chunk("123")],
        )
        assert citation_accuracy(r) == 1.0

    def test_all_hallucinated(self) -> None:
        r = _make_result(
            actual_citations=["999"],
            retrieved_chunks=[_chunk("123")],
        )
        assert citation_accuracy(r) == 0.0

    def test_mixed_grounded_and_hallucinated(self) -> None:
        r = _make_result(
            actual_citations=["123", "999"],
            retrieved_chunks=[_chunk("123")],
        )
        assert citation_accuracy(r) == 0.5

    def test_no_citations_returns_one(self) -> None:
        r = _make_result(actual_citations=[], retrieved_chunks=[])
        assert citation_accuracy(r) == 1.0

    def test_bis_article_normalised(self) -> None:
        r = _make_result(
            actual_citations=["13 BIS"],
            retrieved_chunks=[{"chunk_id": "x", "text": "", "metadata": {"article_number": "13 bis"}, "similarity_score": 0.8, "rerank_score": 2.0}],
        )
        assert citation_accuracy(r) == 1.0


# ---------------------------------------------------------------------------
# retrieval_hit_rate
# ---------------------------------------------------------------------------


class TestRetrievalHitRate:
    def test_all_expected_found(self) -> None:
        r = _make_result(
            expected_articles=["123", "124"],
            retrieved_chunks=[_chunk("123"), _chunk("124")],
        )
        assert retrieval_hit_rate(r) == 1.0

    def test_none_found(self) -> None:
        r = _make_result(
            expected_articles=["302"],
            retrieved_chunks=[_chunk("123")],
        )
        assert retrieval_hit_rate(r) == 0.0

    def test_partial_hit(self) -> None:
        r = _make_result(
            expected_articles=["123", "302"],
            retrieved_chunks=[_chunk("123")],
        )
        assert retrieval_hit_rate(r) == 0.5

    def test_no_expected_articles_returns_one(self) -> None:
        r = _make_result(expected_articles=[], retrieved_chunks=[])
        assert retrieval_hit_rate(r) == 1.0

    def test_article_found_in_chunk_text(self) -> None:
        # article not in metadata but present in chunk text
        chunk = {
            "chunk_id": "x",
            "text": "ARTÍCULO 302. Comete homicidio…",
            "metadata": {},
            "similarity_score": 0.7,
            "rerank_score": 2.0,
        }
        r = _make_result(expected_articles=["302"], retrieved_chunks=[chunk])
        assert retrieval_hit_rate(r) == 1.0


# ---------------------------------------------------------------------------
# detect_hallucination
# ---------------------------------------------------------------------------


class TestDetectHallucination:
    def test_no_hallucination(self) -> None:
        r = _make_result(
            actual_citations=["123"],
            retrieved_chunks=[_chunk("123")],
        )
        assert detect_hallucination(r) is False

    def test_hallucination_detected(self) -> None:
        r = _make_result(
            actual_citations=["999"],
            retrieved_chunks=[_chunk("123")],
        )
        assert detect_hallucination(r) is True

    def test_no_citations_no_hallucination(self) -> None:
        r = _make_result(actual_citations=[], retrieved_chunks=[])
        assert detect_hallucination(r) is False


# ---------------------------------------------------------------------------
# latency_breakdown
# ---------------------------------------------------------------------------


class TestLatencyBreakdown:
    def test_proportions_sum_to_100(self) -> None:
        r = _make_result(retrieval_ms=5000.0, generation_ms=2000.0, total_ms=7500.0)
        lb = latency_breakdown(r)
        total_pct = lb["retrieval_pct"] + lb["generation_pct"] + lb["overhead_pct"]
        assert abs(total_pct - 100.0) < 0.2  # rounding to 1 dp can sum to 100.1

    def test_ms_sum_equals_total(self) -> None:
        r = _make_result(retrieval_ms=5000.0, generation_ms=2000.0, total_ms=7500.0)
        lb = latency_breakdown(r)
        assert lb["retrieval_ms"] + lb["generation_ms"] + lb["overhead_ms"] == pytest.approx(
            lb["total_ms"], abs=0.2
        )

    def test_zero_total_returns_zeros(self) -> None:
        r = _make_result(retrieval_ms=0.0, generation_ms=0.0, total_ms=0.0)
        lb = latency_breakdown(r)
        assert lb["total_ms"] == 0.0
        assert lb["retrieval_pct"] == 0.0


# ---------------------------------------------------------------------------
# article_coverage_rate
# ---------------------------------------------------------------------------


class TestArticleCoverageRate:
    def test_all_cited(self) -> None:
        r = _make_result(
            expected_articles=["123"],
            actual_citations=["123"],
        )
        assert article_coverage_rate(r) == 1.0

    def test_none_cited(self) -> None:
        r = _make_result(
            expected_articles=["123"],
            actual_citations=["999"],
        )
        assert article_coverage_rate(r) == 0.0

    def test_no_expected_returns_one(self) -> None:
        r = _make_result(expected_articles=[], actual_citations=[])
        assert article_coverage_rate(r) == 1.0


# ---------------------------------------------------------------------------
# oos_correctly_refused
# ---------------------------------------------------------------------------


class TestOosCorrectlyRefused:
    def test_non_oos_returns_none(self) -> None:
        r = _make_result(question_type="factual_lookup")
        assert oos_correctly_refused(r) is None

    def test_oos_with_refusal_phrase(self) -> None:
        r = _make_result(
            question_type="adversarial_oos",
            actual_answer="No puedo responder, está fuera del alcance del código penal.",
        )
        assert oos_correctly_refused(r) is True

    def test_oos_with_no_citations_is_refused(self) -> None:
        r = _make_result(
            question_type="adversarial_oos",
            actual_citations=[],
            actual_answer="El IVA es un impuesto.",
        )
        assert oos_correctly_refused(r) is True

    def test_oos_answered_incorrectly(self) -> None:
        r = _make_result(
            question_type="adversarial_oos",
            actual_citations=["123"],
            actual_answer="El artículo 123 establece algo.",
        )
        assert oos_correctly_refused(r) is False

    def test_error_result_returns_none(self) -> None:
        r = _make_result(question_type="adversarial_oos", error="Pipeline failed")
        assert oos_correctly_refused(r) is None


# ---------------------------------------------------------------------------
# score_result
# ---------------------------------------------------------------------------


class TestScoreResult:
    def test_returns_result_metrics_type(self) -> None:
        r = _make_result(actual_citations=["123"], retrieved_chunks=[_chunk("123")])
        m = score_result(r)
        assert isinstance(m, ResultMetrics)
        assert m.question_id == "q001"

    def test_error_result_scores_zero(self) -> None:
        r = _make_result(error="Pipeline error", expected_articles=["123"])
        m = score_result(r)
        assert m.citation_accuracy == 0.0
        assert m.retrieval_hit_rate == 0.0
        assert m.hallucination_detected is False
        assert m.missing_articles == ["123"]

    def test_hallucinated_articles_list(self) -> None:
        r = _make_result(
            actual_citations=["999"],
            retrieved_chunks=[_chunk("123")],
        )
        m = score_result(r)
        assert "999" in m.hallucinated_articles


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def _two_results(self) -> list[EvalResult]:
        r1 = _make_result(
            question_id="q001",
            actual_citations=["123"],
            retrieved_chunks=[_chunk("123")],
            expected_articles=["123"],
        )
        r2 = _make_result(
            question_id="q002",
            actual_citations=["999"],
            retrieved_chunks=[_chunk("123")],
            expected_articles=["999"],
        )
        return [r1, r2]

    def test_returns_aggregate_metrics(self) -> None:
        agg = aggregate(self._two_results())
        assert isinstance(agg, AggregateMetrics)

    def test_n_total(self) -> None:
        agg = aggregate(self._two_results())
        assert agg.n_total == 2
        assert agg.n_successful == 2
        assert agg.n_errors == 0

    def test_error_counted(self) -> None:
        results = self._two_results()
        results[0].error = "oops"
        agg = aggregate(results)
        assert agg.n_errors == 1
        assert agg.n_successful == 1

    def test_hallucination_rate_between_0_and_1(self) -> None:
        agg = aggregate(self._two_results())
        assert agg.hallucination_rate is not None
        assert 0.0 <= agg.hallucination_rate <= 1.0

    def test_mean_latency_positive(self) -> None:
        agg = aggregate(self._two_results())
        assert agg.mean_total_ms is not None
        assert agg.mean_total_ms > 0

    def test_empty_list(self) -> None:
        agg = aggregate([])
        assert agg.n_total == 0
        assert agg.mean_citation_accuracy is None


# ---------------------------------------------------------------------------
# stratified_breakdown
# ---------------------------------------------------------------------------


class TestStratifiedBreakdown:
    def test_keys_present(self) -> None:
        results = [
            _make_result(question_id="q1", difficulty="easy", category="homicidio", question_type="factual_lookup"),
            _make_result(question_id="q2", difficulty="hard", category="robo", question_type="comparison"),
        ]
        bd = stratified_breakdown(results)
        assert "by_difficulty" in bd
        assert "by_question_type" in bd
        assert "by_category" in bd

    def test_groups_by_difficulty(self) -> None:
        results = [
            _make_result(question_id="q1", difficulty="easy"),
            _make_result(question_id="q2", difficulty="hard"),
        ]
        bd = stratified_breakdown(results)
        assert "easy" in bd["by_difficulty"]
        assert "hard" in bd["by_difficulty"]


# ---------------------------------------------------------------------------
# worst_performers / best_performers
# ---------------------------------------------------------------------------


class TestPerformers:
    def _results(self) -> list[EvalResult]:
        return [
            _make_result(
                question_id=f"q{i}",
                actual_citations=["123"] if i % 2 == 0 else ["999"],
                retrieved_chunks=[_chunk("123")],
            )
            for i in range(6)
        ]

    def test_worst_performers_returns_n(self) -> None:
        worst = worst_performers(self._results(), n=3)
        assert len(worst) == 3

    def test_worst_first(self) -> None:
        worst = worst_performers(self._results(), metric="citation_accuracy", n=2)
        scores = [citation_accuracy(r) for r in worst]
        assert scores == sorted(scores)

    def test_best_performers_returns_n(self) -> None:
        best = best_performers(self._results(), n=2)
        assert len(best) == 2

    def test_best_descending(self) -> None:
        best = best_performers(self._results(), metric="citation_accuracy", n=3)
        scores = [citation_accuracy(r) for r in best]
        assert scores == sorted(scores, reverse=True)

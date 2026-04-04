"""Evaluation package for the Mexican Penal Code RAG system.

Exports:
    EvalResult          — per-question result record (pipeline + metric scores).
    EvalRun             — complete evaluation run (all questions + aggregates).
    Evaluator           — orchestrates pipeline execution against the test set.
    RAGASScorer         — scores EvalResult objects with RAGAS + Claude Haiku.
    score_with_ragas    — convenience wrapper around RAGASScorer.score().
    AggregateMetrics    — aggregated custom metric statistics.
    ResultMetrics       — all custom metrics for a single EvalResult.
    aggregate           — compute AggregateMetrics from a list of EvalResults.
    stratified_breakdown — group metrics by difficulty / question_type / category.
    score_result        — compute all custom metrics for one EvalResult.
    score_all           — apply score_result to a list.
    apply_to_run        — refresh custom metric fields in an EvalRun in-place.
    worst_performers    — N lowest-scoring results on a given metric.
    best_performers     — N highest-scoring results on a given metric.
    Reporter            — generates a markdown report from an EvalRun.
    generate_report     — convenience wrapper around Reporter.generate().
"""

from src.evaluation.evaluator import EvalResult, EvalRun, Evaluator
from src.evaluation.metrics import (
    AggregateMetrics,
    ResultMetrics,
    aggregate,
    apply_to_run,
    best_performers,
    score_all,
    score_result,
    stratified_breakdown,
    worst_performers,
)
from src.evaluation.ragas_scorer import RAGASScorer, score_with_ragas
from src.evaluation.reporter import Reporter, generate_report

__all__ = [
    "EvalResult",
    "EvalRun",
    "Evaluator",
    "RAGASScorer",
    "score_with_ragas",
    "AggregateMetrics",
    "ResultMetrics",
    "aggregate",
    "apply_to_run",
    "best_performers",
    "score_all",
    "score_result",
    "stratified_breakdown",
    "worst_performers",
    "Reporter",
    "generate_report",
]

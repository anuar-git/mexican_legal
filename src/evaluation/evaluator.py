"""Evaluation orchestrator for the Mexican Penal Code RAG system.

Runs every question in the golden test set through the full RAG pipeline,
records results, and computes cheap custom metrics inline (citation accuracy,
retrieval hit rate, hallucination detection).  RAGAS scores
(faithfulness, answer_relevance, context_relevance, context_recall) are
left as ``None`` here and filled in by ``RAGASScorer`` in ``metrics.py``.

Typical usage::

    from src.evaluation.evaluator import Evaluator

    ev = Evaluator()
    run = ev.run(limit=5)               # quick smoke-test (5 questions)
    run = ev.run()                      # full 75-question evaluation
    run.save(Path("reports/run_abc.json"))

    # Later — add RAGAS scores without re-running the pipeline
    run2 = EvalRun.load(Path("reports/run_abc.json"))

Environment variables (loaded from .env):
    PINECONE_API_KEY     — required for retrieval
    PINECONE_INDEX_NAME  — required for retrieval
    ANTHROPIC_API_KEY    — required for generation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from src.retrieval import query as _pipeline_query
from src.retrieval.generator import Citation
from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_DEFAULT_TEST_SET = Path("data/evaluation/golden_test_set.json")
_DEFAULT_CHECKPOINT_DIR = Path("data/processed/eval_checkpoints")


# ---------------------------------------------------------------------------
# Inline metric helpers
# ---------------------------------------------------------------------------


def _compute_citation_accuracy(citations: list[Citation]) -> float:
    """Fraction of citations grounded in retrieved chunks (not hallucinated).

    Returns 1.0 when there are no citations — the model correctly produced
    no article references (good for out-of-scope questions).

    Args:
        citations: Citations extracted by the Generator.

    Returns:
        Float in [0, 1].  1.0 = all grounded; 0.0 = all hallucinated.
    """
    if not citations:
        return 1.0
    n_grounded = sum(1 for c in citations if not c.hallucination_flag)
    return round(n_grounded / len(citations), 4)


def _compute_retrieval_hit_rate(
    expected_articles: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> float:
    """Fraction of expected articles that appear in the retrieved chunks.

    Checks the ``article_number`` metadata field of each retrieved chunk.
    Normalises article numbers to uppercase with collapsed whitespace before
    comparing (so "13 BIS" == "13 BIS" regardless of source formatting).

    Returns 1.0 when ``expected_articles`` is empty — i.e. out-of-scope
    questions where no article retrieval is expected.

    Args:
        expected_articles: Article numbers from the TestCase golden set.
        retrieved_chunks:  Top-k chunks returned by the retrieval pipeline.

    Returns:
        Float in [0, 1].
    """
    if not expected_articles:
        return 1.0  # out-of-scope: N/A, treated as neutral

    def _norm(s: str) -> str:
        import re
        return re.sub(r"\s+", " ", s.strip()).upper()

    expected_norm = {_norm(a) for a in expected_articles}

    retrieved_articles: set[str] = set()
    for chunk in retrieved_chunks:
        art = chunk.metadata.get("article_number")
        if art:
            retrieved_articles.add(_norm(str(art)))

    hits = expected_norm & retrieved_articles
    return round(len(hits) / len(expected_norm), 4)


def _compute_aggregate_scores(
    results: list["EvalResult"],
    total_time_s: float,
) -> dict:
    """Compute mean scores across all successful results.

    RAGAS fields (faithfulness, answer_relevance, etc.) start as None and
    are updated by ``EvalRun.recalculate_aggregates()`` after metrics.py
    populates them.

    Args:
        results:      All EvalResult objects (may include errors).
        total_time_s: Wall-clock time for the entire run in seconds.

    Returns:
        Dict with mean values and counts.
    """
    successful = [r for r in results if r.error is None]
    n_total = len(results)
    n_ok = len(successful)

    def _mean(values: list[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else None

    return {
        "n_total": n_total,
        "n_successful": n_ok,
        "n_errors": n_total - n_ok,
        "total_time_s": round(total_time_s, 2),
        # Timing
        "mean_total_time_ms": _mean([r.total_time_ms for r in successful]),
        "mean_retrieval_time_ms": _mean([r.retrieval_time_ms for r in successful]),
        "mean_generation_time_ms": _mean([r.generation_time_ms for r in successful]),
        # Custom inline metrics
        "citation_accuracy": _mean([r.citation_accuracy for r in successful]),
        "retrieval_hit_rate": _mean([r.retrieval_hit_rate for r in successful]),
        "hallucination_rate": _mean(
            [float(r.hallucination_detected) for r in successful
             if r.hallucination_detected is not None]
        ),
        # RAGAS — None until populated by metrics.py
        "faithfulness": _mean([r.faithfulness for r in successful]),
        "answer_relevance": _mean([r.answer_relevance for r in successful]),
        "context_relevance": _mean([r.context_relevance for r in successful]),
        "context_recall": _mean([r.context_recall for r in successful]),
    }


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """Result record for a single question evaluation run.

    Attributes:
        question_id:          Stable ID from the test set (e.g. "q001").
        question:             The question text.
        category:             Legal topic (e.g. "homicidio").
        difficulty:           Complexity level ("easy" | "medium" | "hard" | "adversarial").
        question_type:        Evaluation category ("factual_lookup" | "comparison" …).
        expected_answer:      Reference answer from the golden test set.
        expected_articles:    Article numbers the answer should cite.
        actual_answer:        Text generated by the RAG pipeline.
        actual_citations:     Article numbers the pipeline actually cited.
        retrieved_chunks:     Serialised top-k chunks (chunk_id, text, metadata, scores).
        model:                Anthropic model ID used (from GenerationResult).
        prompt_version:       Active prompt version string (from GenerationResult).
        retrieval_time_ms:    Wall time for the retrieval stage.
        generation_time_ms:   Wall time for the generation stage.
        total_time_ms:        End-to-end wall time for this question.
        error:                Exception message if the pipeline raised; None otherwise.
        faithfulness:         RAGAS faithfulness score (None until metrics.py runs).
        answer_relevance:     RAGAS answer relevance score.
        context_relevance:    RAGAS context relevance score.
        context_recall:       RAGAS context recall score.
        citation_accuracy:    Fraction of non-hallucinated citations (inline).
        retrieval_hit_rate:   Fraction of expected articles in retrieved chunks (inline).
        hallucination_detected: True if any citation is flagged as hallucinated (inline).
    """

    model_config = ConfigDict(frozen=False)

    question_id: str
    question: str
    category: str
    difficulty: str
    question_type: str
    expected_answer: str
    expected_articles: list[str]
    actual_answer: str
    actual_citations: list[str]
    retrieved_chunks: list[dict]
    model: str = ""
    prompt_version: str = ""

    # Timing
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float

    # Error — None means success
    error: str | None = None

    # RAGAS scores — populated by metrics.py
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_relevance: float | None = None
    context_recall: float | None = None

    # Custom inline scores — populated during pipeline execution
    citation_accuracy: float | None = None
    retrieval_hit_rate: float | None = None
    hallucination_detected: bool | None = None


class EvalRun(BaseModel):
    """Complete evaluation run — pipeline results for all questions.

    Attributes:
        run_id:            Short 8-hex-char UUID for this run.
        timestamp:         ISO 8601 UTC timestamp when the run started.
        test_set_version:  Version string from golden_test_set.json.
        pipeline_config:   Retriever/Generator config captured from the run.
        results:           Per-question EvalResult objects.
        aggregate_scores:  Means of numeric metrics across successful results.
    """

    model_config = ConfigDict(frozen=False)

    run_id: str
    timestamp: str
    test_set_version: str
    pipeline_config: dict
    results: list[EvalResult]
    aggregate_scores: dict

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serialise the run to a JSON file.

        Creates parent directories if they do not exist.

        Args:
            path: Destination file path (typically ``reports/run_{id}.json``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        logger.info("EvalRun %s saved → %s", self.run_id, path)

    @classmethod
    def load(cls, path: Path | str) -> "EvalRun":
        """Deserialise an EvalRun from a JSON file.

        Args:
            path: Path to a previously saved EvalRun JSON.

        Returns:
            EvalRun instance with all results and scores restored.
        """
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def recalculate_aggregates(self) -> None:
        """Recompute ``aggregate_scores`` in-place.

        Call this after metrics.py has populated RAGAS scores on the
        individual results so that the aggregate reflects the latest values.
        """
        elapsed = self.aggregate_scores.get("total_time_s", 0.0)
        self.aggregate_scores = _compute_aggregate_scores(self.results, elapsed)

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @property
    def successful_results(self) -> list[EvalResult]:
        """Results where the pipeline completed without error."""
        return [r for r in self.results if r.error is None]

    @property
    def failed_results(self) -> list[EvalResult]:
        """Results where the pipeline raised an exception."""
        return [r for r in self.results if r.error is not None]

    def by_type(self, question_type: str) -> list[EvalResult]:
        """Filter results by question_type (e.g. 'factual_lookup')."""
        return [r for r in self.results if r.question_type == question_type]

    def by_difficulty(self, difficulty: str) -> list[EvalResult]:
        """Filter results by difficulty level."""
        return [r for r in self.results if r.difficulty == difficulty]

    def by_category(self, category: str) -> list[EvalResult]:
        """Filter results by legal category (e.g. 'homicidio')."""
        return [r for r in self.results if r.category == category]

    def score_summary(self) -> dict:
        """Return a concise score summary suitable for console display."""
        agg = self.aggregate_scores
        ok = agg.get("n_successful", 0)
        total = agg.get("n_total", 0)

        return {
            "run_id": self.run_id,
            "questions": f"{ok}/{total} succeeded",
            "retrieval_hit_rate": agg.get("retrieval_hit_rate"),
            "citation_accuracy": agg.get("citation_accuracy"),
            "hallucination_rate": agg.get("hallucination_rate"),
            "faithfulness": agg.get("faithfulness"),
            "answer_relevance": agg.get("answer_relevance"),
            "context_recall": agg.get("context_recall"),
            "context_relevance": agg.get("context_relevance"),
            "mean_total_ms": agg.get("mean_total_time_ms"),
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Runs every test-set question through the RAG pipeline and records results.

    Initialisation is lightweight — the Pinecone/Anthropic clients and the
    cross-encoder model are loaded lazily on the first ``pipeline.query()``
    call.  Expect ~30 s of warm-up time before the first result appears.

    Args:
        test_set_path:   Path to ``golden_test_set.json``.
        checkpoint_dir:  Directory to write per-question checkpoints.
                         Pass ``None`` to disable checkpointing (not recommended
                         for full 75-question runs).
        top_k_candidates: Pinecone candidates per query variant (Stage 2).
        top_k_final:      Chunks kept after reranking (Stage 3).
    """

    def __init__(
        self,
        test_set_path: Path | str = _DEFAULT_TEST_SET,
        checkpoint_dir: Path | str | None = _DEFAULT_CHECKPOINT_DIR,
        top_k_candidates: int = 20,
        top_k_final: int = 5,
    ) -> None:
        self._test_set_path = Path(test_set_path)
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._top_k_candidates = top_k_candidates
        self._top_k_final = top_k_final

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        limit: int | None = None,
        question_types: list[str] | None = None,
        difficulties: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> EvalRun:
        """Run the evaluation pipeline on the test set and return an EvalRun.

        Filters are applied in order (question_types → difficulties → categories
        → limit) before running.  Results are checkpointed after each question
        so a partial run survives process failure.

        Args:
            limit:          Cap on the number of questions to evaluate.
            question_types: Whitelist of question_type values to include.
                            E.g. ``["factual_lookup", "comparison"]``.
            difficulties:   Whitelist of difficulty values.
                            E.g. ``["easy", "medium"]``.
            categories:     Whitelist of category values.
                            E.g. ``["homicidio", "robo"]``.

        Returns:
            EvalRun with all results and inline custom metrics populated.
            RAGAS score fields remain None until metrics.py is run.
        """
        version, all_questions = self._load_test_set()
        questions = self._filter_questions(
            all_questions, question_types, difficulties, categories, limit
        )

        run_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("=" * 62)
        logger.info("EvalRun %s  —  %d question(s)  (test set v%s)",
                    run_id, len(questions), version)
        logger.info("  test_set : %s", self._test_set_path)
        if self._checkpoint_dir:
            logger.info("  checkpoint: %s/eval_%s_*.json",
                        self._checkpoint_dir, run_id)
        logger.info("=" * 62)

        t0 = time.monotonic()
        results = asyncio.run(self._run_all_async(questions, run_id))
        total_time_s = time.monotonic() - t0

        pipeline_config = self._extract_pipeline_config(results)
        aggregate_scores = _compute_aggregate_scores(results, total_time_s)

        eval_run = EvalRun(
            run_id=run_id,
            timestamp=timestamp,
            test_set_version=version,
            pipeline_config=pipeline_config,
            results=results,
            aggregate_scores=aggregate_scores,
        )

        # Write final checkpoint so the run is always recoverable
        self._save_checkpoint(run_id, results, final=True)

        n_ok = sum(1 for r in results if r.error is None)
        logger.info("=" * 62)
        logger.info("Run complete in %.1f s  —  %d/%d succeeded",
                    total_time_s, n_ok, len(results))
        logger.info("  citation_accuracy  : %s",
                    aggregate_scores.get("citation_accuracy"))
        logger.info("  retrieval_hit_rate : %s",
                    aggregate_scores.get("retrieval_hit_rate"))
        logger.info("  hallucination_rate : %s",
                    aggregate_scores.get("hallucination_rate"))
        logger.info("=" * 62)

        return eval_run

    # ------------------------------------------------------------------
    # Internal — test set loading & filtering
    # ------------------------------------------------------------------

    def _load_test_set(self) -> tuple[str, list[dict]]:
        """Load and parse the golden test set JSON.

        Returns:
            Tuple of (version_string, list_of_question_dicts).

        Raises:
            FileNotFoundError: If the test set file does not exist.
        """
        if not self._test_set_path.exists():
            raise FileNotFoundError(
                f"Test set not found at {self._test_set_path}. "
                "Run `python scripts/evaluate.py` from the project root."
            )
        with self._test_set_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data["test_set_version"], data["questions"]

    @staticmethod
    def _filter_questions(
        questions: list[dict],
        question_types: list[str] | None,
        difficulties: list[str] | None,
        categories: list[str] | None,
        limit: int | None,
    ) -> list[dict]:
        if question_types:
            questions = [q for q in questions if q["question_type"] in question_types]
        if difficulties:
            questions = [q for q in questions if q["difficulty"] in difficulties]
        if categories:
            questions = [q for q in questions if q["category"] in categories]
        if limit is not None:
            questions = questions[:limit]
        return questions

    # ------------------------------------------------------------------
    # Internal — async pipeline execution
    # ------------------------------------------------------------------

    async def _run_all_async(
        self,
        questions: list[dict],
        run_id: str,
    ) -> list[EvalResult]:
        """Execute all questions sequentially inside one event loop.

        Sequential execution is intentional: parallel calls would saturate
        the Pinecone and Anthropic rate limits and make timing data noisy.

        Args:
            questions: Filtered list of question dicts from the test set.
            run_id:    Current run identifier (used in checkpoint filenames).

        Returns:
            List of EvalResult objects in the same order as ``questions``.
        """
        results: list[EvalResult] = []
        n = len(questions)

        for i, q in enumerate(questions, 1):
            logger.info(
                "[%d/%d] %-20s  %-12s  %-14s  %s",
                i, n, q["id"], q["difficulty"], q["question_type"],
                q["question"][:60] + ("…" if len(q["question"]) > 60 else ""),
            )

            result = await self._run_single_async(q)
            results.append(result)

            if result.error:
                logger.warning("           ✗ ERROR: %s", result.error)
            else:
                logger.info(
                    "           ✓ hit=%.2f  cit_acc=%.2f  halluc=%s"
                    "  ret=%.0fms  gen=%.0fms",
                    result.retrieval_hit_rate or 0.0,
                    result.citation_accuracy or 0.0,
                    result.hallucination_detected,
                    result.retrieval_time_ms,
                    result.generation_time_ms,
                )

            self._save_checkpoint(run_id, results)

        return results

    async def _run_single_async(self, q: dict) -> EvalResult:
        """Run one question through the RAG pipeline and build an EvalResult.

        On pipeline failure the result is recorded with ``error`` set and all
        metric fields left as None so downstream aggregation can skip it.

        Args:
            q: Question dict from golden_test_set.json.

        Returns:
            EvalResult with pipeline outputs and inline custom metrics.
        """
        t0 = time.monotonic()

        try:
            gen_result = await _pipeline_query(q["question"])
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
            return EvalResult(
                question_id=q["id"],
                question=q["question"],
                category=q["category"],
                difficulty=q["difficulty"],
                question_type=q["question_type"],
                expected_answer=q["expected_answer"],
                expected_articles=q["expected_articles"],
                actual_answer="",
                actual_citations=[],
                retrieved_chunks=[],
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
                total_time_ms=elapsed_ms,
                error=str(exc),
            )

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

        # Serialise retrieved chunks to plain dicts (JSON-safe)
        retrieved_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "similarity_score": chunk.similarity_score,
                "rerank_score": chunk.rerank_score,
            }
            for chunk in gen_result.retrieval.chunks
        ]

        # Extract article numbers from citations
        actual_citations = [c.article_number for c in gen_result.citations]

        # Inline custom metrics — no LLM required
        citation_accuracy = _compute_citation_accuracy(gen_result.citations)
        retrieval_hit_rate = _compute_retrieval_hit_rate(
            q["expected_articles"],
            gen_result.retrieval.chunks,
        )
        hallucination_detected = (
            any(c.hallucination_flag for c in gen_result.citations)
            if gen_result.citations
            else False
        )

        return EvalResult(
            question_id=q["id"],
            question=q["question"],
            category=q["category"],
            difficulty=q["difficulty"],
            question_type=q["question_type"],
            expected_answer=q["expected_answer"],
            expected_articles=q["expected_articles"],
            actual_answer=gen_result.answer,
            actual_citations=actual_citations,
            retrieved_chunks=retrieved_chunks,
            model=gen_result.model,
            prompt_version=gen_result.prompt_version,
            retrieval_time_ms=gen_result.retrieval.retrieval_time_ms,
            generation_time_ms=gen_result.generation_time_ms,
            total_time_ms=elapsed_ms,
            citation_accuracy=citation_accuracy,
            retrieval_hit_rate=retrieval_hit_rate,
            hallucination_detected=hallucination_detected,
        )

    # ------------------------------------------------------------------
    # Internal — config extraction & checkpointing
    # ------------------------------------------------------------------

    def _extract_pipeline_config(self, results: list[EvalResult]) -> dict:
        """Build pipeline_config from the first successful result.

        Args:
            results: All EvalResult objects from the run.

        Returns:
            Dict with retriever/generator settings.
        """
        first_ok = next((r for r in results if r.error is None), None)
        return {
            "top_k_candidates": self._top_k_candidates,
            "top_k_final": self._top_k_final,
            "model": first_ok.model if first_ok else None,
            "prompt_version": first_ok.prompt_version if first_ok else None,
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "embed_model": "llama-text-embed-v2",
            "test_set_path": str(self._test_set_path),
        }

    def _save_checkpoint(
        self,
        run_id: str,
        results: list[EvalResult],
        final: bool = False,
    ) -> None:
        """Write a checkpoint JSON so partial runs are recoverable.

        Intermediate checkpoints are written to ``eval_{run_id}.json``.
        The final checkpoint is written to ``eval_{run_id}_final.json``.
        Both are overwritten on each call so disk usage stays constant.

        Args:
            run_id:  Current run identifier.
            results: Results collected so far.
            final:   If True, write the ``_final`` filename instead.
        """
        if self._checkpoint_dir is None:
            return

        suffix = "_final" if final else ""
        path = self._checkpoint_dir / f"eval_{run_id}{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        serialised = [r.model_dump() for r in results]
        path.write_text(
            json.dumps(serialised, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug(
            "Checkpoint: %s  (%d/%d results)",
            path.name, len(results),
            len(results),
        )

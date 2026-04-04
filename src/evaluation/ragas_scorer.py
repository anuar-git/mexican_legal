"""RAGAS integration for the Mexican Penal Code RAG evaluation system.

Scores a list of ``EvalResult`` objects (or a full ``EvalRun``) using four
RAGAS metrics with Claude Haiku as the judge LLM:

+--------------------------+----------------------------------------------+
| Metric                   | What it measures                             |
+==========================+==============================================+
| faithfulness             | Answer is grounded in retrieved context —   |
|                          | primary hallucination signal                 |
+--------------------------+----------------------------------------------+
| answer_relevancy         | Answer addresses the question that was asked |
+--------------------------+----------------------------------------------+
| context_precision        | Retrieved chunks are relevant to the query   |
|                          | (retriever precision signal)                 |
+--------------------------+----------------------------------------------+
| context_recall           | Retriever found all info needed to answer    |
|                          | (retriever completeness signal)              |
+--------------------------+----------------------------------------------+

RAGAS 0.4.x uses module-level metric singletons (legacy API).  This module
configures each singleton's ``.llm`` attribute to point at Claude Haiku so
we stay within the Anthropic ecosystem and avoid OpenAI API costs.
``answer_relevancy`` additionally requires embeddings; we supply a minimal
wrapper around ``sentence-transformers/all-MiniLM-L6-v2`` which is already
present in the project's ``sentence-transformers`` dependency.

Out-of-scope questions (empty ``expected_answer``) receive a neutral
placeholder reference so RAGAS does not crash on missing ground truth.
Their ``context_recall`` will score near 0 — correct behaviour, because the
retriever returns legal articles that cannot support an out-of-scope answer.

Usage::

    from src.evaluation.ragas_scorer import RAGASScorer

    scorer = RAGASScorer(anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])
    updated_results = scorer.score(eval_results)   # mutates in-place + returns
    eval_run = scorer.score_run(eval_run)           # also refreshes aggregates

Environment variables:
    ANTHROPIC_API_KEY — required (also accepted as constructor argument).
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from typing import Optional

from src.evaluation.evaluator import EvalResult, EvalRun

logger = logging.getLogger(__name__)

# Placeholder used as ground-truth reference for questions whose expected
# answer is empty (adversarial / out-of-scope questions).  RAGAS requires a
# non-empty reference string for context_recall and context_precision.
_OOS_REFERENCE = (
    "Esta pregunta está fuera del alcance del Código Penal Federal. "
    "El sistema no puede responder con la información disponible."
)

# Mapping from RAGAS metric name keys → EvalResult field names.
_RAGAS_KEY_TO_FIELD: dict[str, str] = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevance",
    "context_precision": "context_relevance",
    "context_recall": "context_recall",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _SentenceTransformerEmbeddings:
    """Minimal langchain-compatible wrapper around sentence-transformers.

    Implements the ``embed_documents`` / ``embed_query`` interface expected by
    ``ragas.embeddings.LangchainEmbeddingsWrapper``.  We roll this ourselves
    to avoid adding ``langchain-huggingface`` as a hard dependency — the
    ``sentence-transformers`` package is already in the project requirements.

    Args:
        model_name: HuggingFace sentence-transformer model ID.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        logger.debug("Loaded ST embeddings model: %s", model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


def _configure_metrics(
    anthropic_api_key: str,
    model: str,
    embed_model: str,
) -> tuple:
    """Build and configure the four RAGAS metric singletons.

    Assigns the Claude Haiku LLM and sentence-transformer embeddings to the
    RAGAS module-level singleton instances.  This mutates the global singletons,
    which is safe for single-process, single-run usage.

    Args:
        anthropic_api_key: Anthropic API key for Claude Haiku judge calls.
        model:             Anthropic model ID to use as the RAGAS judge LLM.
        embed_model:       Sentence-transformer model name for embeddings.

    Returns:
        Tuple of (faithfulness, answer_relevancy, context_precision,
        context_recall) metric instances, configured and ready.

    Raises:
        ImportError: If ``ragas`` or ``langchain-anthropic`` are not installed.
    """
    try:
        from ragas.metrics import (  # type: ignore[import-untyped]
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "RAGAS scoring requires ragas>=0.2 and langchain-anthropic>=0.3. "
            "Install with: pip install 'ragas>=0.2' 'langchain-anthropic>=0.3'"
        ) from exc

    llm = LangchainLLMWrapper(
        ChatAnthropic(model=model, api_key=anthropic_api_key)
    )
    embeddings = LangchainEmbeddingsWrapper(_SentenceTransformerEmbeddings(embed_model))

    # Suppress the per-import deprecation warning from RAGAS 0.4.x.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        faithfulness.llm = llm
        answer_relevancy.llm = llm
        answer_relevancy.embeddings = embeddings
        context_precision.llm = llm
        context_recall.llm = llm

    logger.debug(
        "RAGAS metrics configured: llm=%s/%s, embeddings=%s",
        type(llm).__name__, model, embed_model,
    )
    return faithfulness, answer_relevancy, context_precision, context_recall


def _to_ragas_dataset(scoreable: list[EvalResult]) -> "EvaluationDataset":  # type: ignore[name-defined]
    """Convert EvalResult objects to a RAGAS EvaluationDataset.

    ``context_recall`` and ``context_precision`` require a non-empty
    ``reference`` string.  For questions with an empty ``expected_answer``
    (adversarial / out-of-scope) we substitute ``_OOS_REFERENCE`` so RAGAS
    does not crash.  Those questions will receive near-zero recall scores,
    which correctly reflects that the retriever is not finding relevant
    information for out-of-scope queries.

    Args:
        scoreable: EvalResult objects whose answers are non-empty.

    Returns:
        RAGAS EvaluationDataset ready for ``evaluate()``.
    """
    from ragas import EvaluationDataset  # type: ignore[import-untyped]
    from ragas.dataset_schema import SingleTurnSample  # type: ignore[import-untyped]

    samples = [
        SingleTurnSample(
            user_input=r.question,
            response=r.actual_answer,
            retrieved_contexts=[c["text"] for c in r.retrieved_chunks],
            reference=r.expected_answer if r.expected_answer.strip() else _OOS_REFERENCE,
        )
        for r in scoreable
    ]
    return EvaluationDataset(samples=samples)


def _safe_float(value: object) -> Optional[float]:
    """Convert a RAGAS score value to float, returning None for NaN/None."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _map_scores_back(
    ragas_result: object,
    scoreable: list[EvalResult],
) -> None:
    """Write per-sample RAGAS scores into the corresponding EvalResult fields.

    Mutates ``scoreable`` in-place.

    Args:
        ragas_result: The ``EvaluationResult`` returned by ``ragas.evaluate()``.
        scoreable:    The subset of EvalResult objects that were scored.
    """
    for ragas_key, result_field in _RAGAS_KEY_TO_FIELD.items():
        try:
            per_sample_scores: list = ragas_result[ragas_key]  # type: ignore[index]
        except (KeyError, TypeError):
            logger.warning("RAGAS result missing key %r — skipping.", ragas_key)
            continue

        if len(per_sample_scores) != len(scoreable):
            logger.warning(
                "RAGAS key %r: got %d scores for %d samples — skipping.",
                ragas_key, len(per_sample_scores), len(scoreable),
            )
            continue

        for result, raw_score in zip(scoreable, per_sample_scores):
            setattr(result, result_field, _safe_float(raw_score))
            logger.debug(
                "  %s  %s=%.4f",
                result.question_id, result_field,
                getattr(result, result_field) or float("nan"),
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RAGASScorer:
    """Scores ``EvalResult`` objects using RAGAS with Claude Haiku as judge.

    Instantiation is lightweight — RAGAS metric singletons are configured
    lazily on the first ``score()`` call.

    Args:
        anthropic_api_key: Anthropic API key.  Falls back to the
                           ``ANTHROPIC_API_KEY`` environment variable.
        model:             Claude model ID to use as the judge LLM.
        embed_model:       Sentence-transformer model name for
                           ``answer_relevancy`` embeddings.
        batch_size:        Number of samples to evaluate per RAGAS batch.
                           Smaller values reduce peak memory and API burst.
        show_progress:     Whether to display RAGAS tqdm progress bars.
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 15,
        show_progress: bool = True,
    ) -> None:
        self._api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "anthropic_api_key must be provided or ANTHROPIC_API_KEY set."
            )
        self._model = model
        self._embed_model = embed_model
        self._batch_size = batch_size
        self._show_progress = show_progress
        self._metrics: Optional[tuple] = None  # lazy init

    def _ensure_metrics(self) -> tuple:
        if self._metrics is None:
            logger.info(
                "Initialising RAGAS metrics (LLM=%s, embeddings=%s)…",
                self._model, self._embed_model,
            )
            self._metrics = _configure_metrics(
                self._api_key, self._model, self._embed_model
            )
        return self._metrics

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def score(self, eval_results: list[EvalResult]) -> list[EvalResult]:
        """Run RAGAS on a list of EvalResult objects and populate score fields.

        Only results with non-empty ``actual_answer`` and at least one
        retrieved chunk are sent to RAGAS.  Results that fail pipeline
        execution (``r.error is not None``) are silently skipped.

        RAGAS scores are written directly into the ``EvalResult`` objects
        (mutation in-place) and the same list is returned for chaining.

        Args:
            eval_results: Results from ``Evaluator.run()`` (any subset).

        Returns:
            The same list with ``faithfulness``, ``answer_relevance``,
            ``context_relevance``, and ``context_recall`` fields populated
            for all scoreable results.

        Raises:
            ImportError: If ``ragas`` or ``langchain-anthropic`` are not
                         installed.
        """
        from ragas import evaluate  # type: ignore[import-untyped]

        # Filter to results that RAGAS can actually score
        scoreable = [
            r for r in eval_results
            if r.error is None
            and r.actual_answer.strip()
            and r.retrieved_chunks
        ]

        n_skipped = len(eval_results) - len(scoreable)
        if n_skipped:
            logger.info(
                "RAGAS: skipping %d result(s) with empty answers or pipeline errors.",
                n_skipped,
            )

        if not scoreable:
            logger.warning("RAGAS: nothing to score — all results were skipped.")
            return eval_results

        metrics = self._ensure_metrics()
        dataset = _to_ragas_dataset(scoreable)

        logger.info(
            "Running RAGAS on %d sample(s) with judge LLM %s …",
            len(scoreable), self._model,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            ragas_result = evaluate(
                dataset,
                metrics=list(metrics),
                raise_exceptions=False,
                show_progress=self._show_progress,
                batch_size=self._batch_size,
            )

        logger.info("RAGAS complete: %s", ragas_result)
        _map_scores_back(ragas_result, scoreable)

        return eval_results

    def score_run(self, eval_run: EvalRun) -> EvalRun:
        """Score all results in an EvalRun and refresh its aggregate scores.

        Convenience wrapper that calls ``score(eval_run.results)`` and then
        ``eval_run.recalculate_aggregates()`` so the aggregate dict is
        immediately up to date.

        Args:
            eval_run: An EvalRun produced by ``Evaluator.run()``.

        Returns:
            The same EvalRun object with RAGAS scores populated throughout.
        """
        self.score(eval_run.results)
        eval_run.recalculate_aggregates()

        agg = eval_run.aggregate_scores
        logger.info(
            "RAGAS aggregate scores — faithfulness=%.3f  answer_rel=%.3f"
            "  ctx_prec=%.3f  ctx_recall=%.3f",
            agg.get("faithfulness") or float("nan"),
            agg.get("answer_relevance") or float("nan"),
            agg.get("context_relevance") or float("nan"),
            agg.get("context_recall") or float("nan"),
        )
        return eval_run


def score_with_ragas(
    eval_results: list[EvalResult],
    anthropic_api_key: Optional[str] = None,
    **kwargs,
) -> list[EvalResult]:
    """Convenience function — score a result list and return it.

    Equivalent to::

        RAGASScorer(anthropic_api_key=key, **kwargs).score(eval_results)

    Args:
        eval_results:      Results from ``Evaluator.run()``.
        anthropic_api_key: Anthropic API key (falls back to env var).
        **kwargs:          Forwarded to ``RAGASScorer.__init__``.

    Returns:
        The same list with RAGAS score fields populated.
    """
    return RAGASScorer(anthropic_api_key=anthropic_api_key, **kwargs).score(
        eval_results
    )

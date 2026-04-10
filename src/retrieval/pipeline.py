"""End-to-end RAG pipeline for the Mexican Penal Code.

Exposes two async entry points consumed by the FastAPI layer (Phase 4):

    query()              — full blocking pipeline, returns GenerationResult.
    query_stream()       — same pipeline but streams answer tokens as an
                           async generator; citations are not included.

Both functions use module-level singletons for Retriever and Generator so
the cross-encoder model is loaded only once per process.

Environment variables (loaded from .env):
    PINECONE_API_KEY     — required
    PINECONE_INDEX_NAME  — required
    ANTHROPIC_API_KEY    — required
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from src.retrieval.generator import GenerationResult, Generator
from src.retrieval.retriever import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# Initialised lazily on first call so import does not trigger network/disk I/O.
# ---------------------------------------------------------------------------

_retriever: Retriever | None = None
_generator: Generator | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        logger.info("Initialising Retriever singleton.")
        _retriever = Retriever()
    return _retriever


def _get_generator() -> Generator:
    global _generator
    if _generator is None:
        logger.info("Initialising Generator singleton.")
        _generator = Generator()
    return _generator


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def query(
    question: str,
    filters: dict | None = None,
    top_k: int = 5,
) -> GenerationResult:
    """Full RAG pipeline: expand → retrieve → rerank → generate.

    Runs the synchronous Retriever and Generator in a thread pool via
    ``asyncio.to_thread`` so the FastAPI event loop is never blocked.

    Args:
        question: The user's natural-language legal question.
        filters:  Optional Pinecone metadata filter dict, e.g.::

                      {"chapter": "Título Vigésimo Segundo"}
                      {"article_number": {"$gte": "200", "$lte": "250"}}

    Returns:
        GenerationResult with answer, validated citations, and timing metadata.
    """
    retriever = _get_retriever()
    generator = _get_generator()

    # Stage 1 + 2 + 3 — runs in a thread to avoid blocking the event loop.
    # Retriever.retrieve() calls Pinecone, Anthropic (expansion), and the
    # cross-encoder model — all synchronous / CPU-bound.
    retrieval: RetrievalResult = await asyncio.to_thread(
        retriever.retrieve, question, filters, top_k
    )

    logger.debug(
        "Retrieval complete: %d candidates → %d chunks (%.1f ms)",
        retrieval.candidates_retrieved,
        len(retrieval.chunks),
        retrieval.retrieval_time_ms,
    )

    # Generation — calls Anthropic API synchronously.
    result: GenerationResult = await asyncio.to_thread(generator.generate, retrieval)

    logger.debug(
        "Generation complete: %d citations, %.1f ms",
        len(result.citations),
        result.generation_time_ms,
    )

    return result


async def retrieve(
    question: str,
    filters: dict | None = None,
    top_k: int = 5,
) -> RetrievalResult:
    """Run only the retrieval stage (expand → search → rerank).

    Exposed separately so the streaming endpoint can emit a ``retrieval``
    SSE event before generation begins, without running retrieval twice.

    Args:
        question: The user's natural-language legal question.
        filters:  Optional Pinecone metadata filter dict.
        top_k:    Number of chunks to return after reranking.

    Returns:
        RetrievalResult with reranked chunks and timing metadata.
    """
    retriever = _get_retriever()
    result: RetrievalResult = await asyncio.to_thread(
        retriever.retrieve, question, filters, top_k
    )
    logger.debug(
        "Retrieval complete: %d candidates → %d chunks (%.1f ms)",
        result.candidates_retrieved,
        len(result.chunks),
        result.retrieval_time_ms,
    )
    return result


async def query_stream(
    question: str,
    filters: dict | None = None,
    retrieval: RetrievalResult | None = None,
) -> AsyncIterator[str]:
    """Stream answer tokens for a question.

    If *retrieval* is supplied the retrieval stage is skipped entirely —
    the streaming endpoint calls ``retrieve()`` first so it can emit the
    ``retrieval`` SSE event before generation starts, then passes the result
    here to avoid a redundant Pinecone round-trip.

    Tokens arrive from Claude via the Anthropic SDK streaming API.  The sync
    generator is bridged into an async generator via an ``asyncio.Queue``.
    Producer exceptions are propagated to the caller (they are not silently
    swallowed).

    Args:
        question:  The user's natural-language legal question.
        filters:   Optional Pinecone metadata filter dict (ignored when
                   *retrieval* is pre-supplied).
        retrieval: Pre-computed RetrievalResult.  When None, retrieval runs
                   inside this coroutine.

    Yields:
        Raw text token strings as they arrive from the model.
    """
    generator = _get_generator()

    if retrieval is None:
        retrieval = await retrieve(question, filters)

    logger.debug(
        "Streaming generation: %d chunks in context",
        len(retrieval.chunks),
    )

    # Bridge the sync token iterator into an async generator via a queue.
    # A thread-pool worker runs generate_stream(); it pushes tokens (and a
    # None sentinel) onto the queue using call_soon_threadsafe so the event
    # loop stays unblocked.  Exceptions in the producer are captured and
    # re-raised on the consumer side after the sentinel is received.
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    _producer_error: list[BaseException] = []

    def _produce() -> None:
        """Thread-pool worker: push tokens then sentinel onto the queue."""
        try:
            for token in generator.generate_stream(retrieval):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:  # noqa: BLE001
            _producer_error.append(exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _produce)

    while True:
        token = await queue.get()
        if token is None:
            if _producer_error:
                raise _producer_error[0]
            break
        yield token

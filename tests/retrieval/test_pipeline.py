"""Integration tests for the end-to-end RAG pipeline.

These tests hit live external APIs (Pinecone, Anthropic).  They are gated
behind the ``integration`` pytest marker and skipped automatically in CI
unless the environment variable ``RUN_INTEGRATION=1`` is set.

Run locally:
    RUN_INTEGRATION=1 pytest tests/retrieval/test_pipeline.py -v

Skip in CI:
    pytest tests/ -v  (integration tests are skipped automatically)
"""

import os

import pytest

from src.retrieval.generator import Citation, GenerationResult
from src.retrieval.pipeline import query, query_stream
from src.retrieval.retriever import RetrievalResult

# ---------------------------------------------------------------------------
# Marker: skip unless RUN_INTEGRATION=1
# ---------------------------------------------------------------------------

integration = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run integration tests against live APIs.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_retrieval(retrieval: RetrievalResult, question: str) -> None:
    """Shared assertions on the retrieval sub-result."""
    assert retrieval.query == question
    assert retrieval.candidates_retrieved > 0, "No candidates retrieved from Pinecone"
    assert len(retrieval.chunks) > 0, "No chunks after reranking"
    assert retrieval.retrieval_time_ms > 0


def _assert_citations(citations: list[Citation]) -> None:
    """Each Citation must have an article number and a non-negative confidence."""
    for c in citations:
        assert c.article_number, "Citation missing article_number"
        assert c.confidence >= 0.0
        # Source text must be present for non-hallucinated citations
        if not c.hallucination_flag:
            assert c.source_text, f"Artículo {c.article_number}: missing source_text"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@integration
@pytest.mark.asyncio
async def test_full_pipeline_homicidio() -> None:
    """Ask about homicide penalties — a well-covered topic in the penal code."""
    question = "¿Cuál es la pena por homicidio?"
    result: GenerationResult = await query(question)

    # Answer quality
    assert result.answer is not None
    assert len(result.answer) > 50, "Answer is suspiciously short"

    # Citations
    assert len(result.citations) > 0, "Expected at least one article citation"
    _assert_citations(result.citations)

    # Retrieval sub-result
    _assert_retrieval(result.retrieval, question)

    # Timing and metadata
    assert result.generation_time_ms > 0
    assert result.model, "model field must not be empty"
    assert result.prompt_version, "prompt_version field must not be empty"


@integration
@pytest.mark.asyncio
async def test_full_pipeline_robo() -> None:
    """Ask about theft — verifies the pipeline works for a second query."""
    question = "¿Qué artículos regulan el delito de robo?"
    result: GenerationResult = await query(question)

    assert len(result.answer) > 50
    assert len(result.citations) > 0
    _assert_retrieval(result.retrieval, question)


@integration
@pytest.mark.asyncio
async def test_pipeline_with_metadata_filter() -> None:
    """Verify optional metadata filters are forwarded to Pinecone correctly."""
    question = "¿Cuáles son las penas?"
    result: GenerationResult = await query(
        question,
        filters={"article_number": {"$gt": ""}},  # all chunks with a known article
    )

    assert result.answer is not None
    # Every returned chunk must have a non-empty article_number in metadata
    for chunk in result.retrieval.chunks:
        assert chunk.metadata.get("article_number"), (
            f"Chunk {chunk.chunk_id} has no article_number despite filter"
        )


@integration
@pytest.mark.asyncio
async def test_streaming_yields_tokens() -> None:
    """Streaming pipeline must yield at least one non-empty token."""
    question = "¿Qué es el homicidio culposo?"
    tokens: list[str] = []

    async for token in query_stream(question):
        tokens.append(token)

    assert len(tokens) > 0, "query_stream yielded no tokens"
    full_text = "".join(tokens)
    assert len(full_text) > 50, "Streamed answer is suspiciously short"


@integration
@pytest.mark.asyncio
async def test_hallucination_flag_for_unknown_query() -> None:
    """A question far outside the penal code should produce no hallucinated cites,
    because the model should acknowledge insufficient context rather than invent
    article numbers."""
    question = "¿Cuál es la regulación fiscal del IVA en México?"
    result: GenerationResult = await query(question)

    hallucinated = [c for c in result.citations if c.hallucination_flag]
    # We do not assert zero hallucinations — the model may still mention an
    # article.  We just check that the flag is correctly set when the cited
    # article is absent from the retrieved chunks.
    for c in hallucinated:
        assert c.confidence == 0.0
        assert c.source_text == ""

"""Generation pipeline for the Mexican Penal Code RAG system.

Takes a RetrievalResult, formats the top-5 chunks into a prompt, sends it to
Claude Haiku, parses cited article numbers from the response, cross-references
them against the retrieved chunks to validate citations, and returns a
GenerationResult.

Streaming:
    ``Generator.generate_stream()`` yields raw text tokens via the Anthropic
    SDK streaming API.  The SSE endpoint in Phase 4 consumes this iterator.

Environment variables (loaded from .env):
    ANTHROPIC_API_KEY — required
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from src.retrieval.prompts import QUERY_TEMPLATE, SYSTEM_PROMPT
from src.retrieval.retriever import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Matches "Artículo 123", "Artículo 13 BIS", "artículo 7", etc.
_ARTICLE_RE = re.compile(r"[Aa]rt[ií]culo\s+([\w\s]+?)(?=[,;.\s]|$)", re.MULTILINE)

# Active prompt versions — used to populate GenerationResult.prompt_version
_SYSTEM_VERSION = "v1"
_QUERY_VERSION = "v1"

# Public constants — used by the streaming endpoint to build the complete event
# without needing a GenerationResult or access to the Generator singleton.
ACTIVE_MODEL: str = _HAIKU_MODEL
ACTIVE_PROMPT_VERSION: str = f"{_SYSTEM_VERSION}.{_QUERY_VERSION}"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single article citation extracted from the model's answer.

    Attributes:
        article_number:  Normalised article identifier (e.g. "123", "13 BIS").
        source_text:     Text of the retrieved chunk that contains this article.
        confidence:      Similarity score of that chunk (0–1).  Set to 0.0 and
                         ``hallucination_flag=True`` when no matching chunk found.
        hallucination_flag: True when the cited article does not appear in any
                            retrieved chunk — potential hallucination.
    """

    article_number: str
    source_text: str
    confidence: float
    hallucination_flag: bool = False


class GenerationResult(BaseModel):
    """Full output of one RAG pipeline run.

    Attributes:
        answer:            The model's natural-language answer.
        citations:         Validated (and flagged) citations extracted from answer.
        retrieval:         The RetrievalResult that fed this generation.
        generation_time_ms: Wall-clock time for the generation step only (ms).
        model:             Anthropic model ID used.
        prompt_version:    Dot-joined active version labels, e.g. ``"v1.v1"``.
    """

    answer: str
    citations: list[Citation]
    retrieval: RetrievalResult
    generation_time_ms: float
    model: str
    prompt_version: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into the context block for QUERY_TEMPLATE.

    Each chunk is preceded by a metadata header so the model can refer to
    article / chapter / title information even if it appears only in the
    header and not the chunk body.

    Format::

        ### Artículo 123 | Capítulo IV | Título Segundo
        <chunk text>

        ### [Sin artículo] | ...
        <chunk text>
    """
    parts: list[str] = []
    for chunk in chunks:
        meta = chunk.metadata
        article = meta.get("article_number") or ""
        chapter = meta.get("chapter") or ""
        title = meta.get("title") or ""

        header_parts: list[str] = []
        if article:
            header_parts.append(f"Artículo {article}")
        else:
            header_parts.append("[Sin artículo]")
        if chapter:
            header_parts.append(f"Capítulo {chapter}")
        if title:
            header_parts.append(f"Título {title}")

        header = " | ".join(header_parts)
        parts.append(f"### {header}\n{chunk.text.strip()}")

    return "\n\n".join(parts)


def extract_article_numbers(text: str) -> list[str]:
    """Return deduplicated, normalised article numbers mentioned in *text*.

    Strips surrounding whitespace and collapses internal spaces so that
    ``"13  BIS"`` normalises to ``"13 BIS"``.

    Args:
        text: Raw model output or any Spanish legal text.

    Returns:
        Ordered list of unique normalised article number strings.
    """
    seen: dict[str, None] = {}  # ordered set
    for match in _ARTICLE_RE.finditer(text):
        raw = match.group(1).strip()
        normalised = re.sub(r"\s+", " ", raw).upper()
        seen[normalised] = None
    return list(seen)


def build_citations(
    article_numbers: list[str],
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """Cross-reference extracted article numbers against retrieved chunks.

    For each article number found in the answer:

    - **Grounded**: chunk metadata or text contains the article → ``Citation``
      with ``confidence = chunk.similarity_score``, ``hallucination_flag=False``.
    - **Hallucinated**: no chunk contains the article → ``Citation`` with
      ``confidence=0.0``, ``hallucination_flag=True``.

    Args:
        article_numbers: Normalised article numbers from ``extract_article_numbers``.
        chunks:          Top-k reranked chunks from the retrieval pipeline.

    Returns:
        One ``Citation`` per article number, in the same order as the input.
    """
    citations: list[Citation] = []

    for article in article_numbers:
        matched_chunk: RetrievedChunk | None = None

        for chunk in chunks:
            # Check metadata field first (exact match after normalisation)
            meta_article = re.sub(r"\s+", " ", (chunk.metadata.get("article_number") or "")).upper()
            if meta_article and meta_article == article:
                matched_chunk = chunk
                break

            # Fallback: check chunk text for the article reference
            if re.search(
                rf"[Aa]rt[ií]culo\s+{re.escape(article)}",
                chunk.text,
            ):
                matched_chunk = chunk
                break

        if matched_chunk is not None:
            citations.append(
                Citation(
                    article_number=article,
                    source_text=matched_chunk.text,
                    confidence=matched_chunk.similarity_score,
                    hallucination_flag=False,
                )
            )
        else:
            logger.warning(
                "Potential hallucination: Artículo %s cited but not in retrieved chunks.",
                article,
            )
            citations.append(
                Citation(
                    article_number=article,
                    source_text="",
                    confidence=0.0,
                    hallucination_flag=True,
                )
            )

    return citations


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator:
    """Wraps Claude Haiku to turn a RetrievalResult into a GenerationResult.

    Args:
        model: Anthropic model ID.  Defaults to Claude Haiku.
        max_tokens: Maximum tokens in the model response.
    """

    def __init__(
        self,
        model: str = _HAIKU_MODEL,
        max_tokens: int = 1024,
    ) -> None:
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._prompt_version = f"{_SYSTEM_VERSION}.{_QUERY_VERSION}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, retrieval: RetrievalResult) -> GenerationResult:
        """Generate an answer from a RetrievalResult.

        Steps:
        1. Format top-5 chunks into the QUERY_TEMPLATE context block.
        2. Call Claude Haiku with SYSTEM_PROMPT as the system message.
        3. Extract cited article numbers from the response via regex.
        4. Cross-reference citations against retrieved chunks.
        5. Return a GenerationResult.

        Args:
            retrieval: Output of ``Retriever.retrieve()``.

        Returns:
            GenerationResult with answer, validated citations, and timing.
        """
        context = _format_context(retrieval.chunks)
        user_message = QUERY_TEMPLATE.format(
            context=context,
            question=retrieval.query,
        )

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        generation_time_ms = (time.monotonic() - t0) * 1000

        answer = response.content[0].text  # type: ignore[union-attr]

        article_numbers = extract_article_numbers(answer)
        citations = build_citations(article_numbers, retrieval.chunks)

        hallucinated = [c for c in citations if c.hallucination_flag]
        if hallucinated:
            logger.warning(
                "%d hallucinated citation(s): %s",
                len(hallucinated),
                [c.article_number for c in hallucinated],
            )

        logger.debug(
            "Generated answer in %.1f ms — %d citations (%d hallucinated)",
            generation_time_ms,
            len(citations),
            len(hallucinated),
        )

        return GenerationResult(
            answer=answer,
            citations=citations,
            retrieval=retrieval,
            generation_time_ms=round(generation_time_ms, 2),
            model=self._model,
            prompt_version=self._prompt_version,
        )

    def generate_stream(
        self,
        retrieval: RetrievalResult,
    ) -> Iterator[str]:
        """Stream answer tokens for a RetrievalResult.

        Yields raw text deltas as they arrive from the Anthropic streaming
        API.  The SSE endpoint (Phase 4) should consume this iterator and
        forward each token to the client.

        Note: Citations are NOT extracted during streaming.  Call
        ``generate()`` when you need the full GenerationResult with citations.

        Args:
            retrieval: Output of ``Retriever.retrieve()``.

        Yields:
            Individual text token strings as they arrive.
        """
        context = _format_context(retrieval.chunks)
        user_message = QUERY_TEMPLATE.format(
            context=context,
            question=retrieval.query,
        )

        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            yield from stream.text_stream

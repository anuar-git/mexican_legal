"""Pydantic request/response schemas for the Mexican Penal Code RAG API.

These models form the public API contract — all FastAPI endpoints accept and
return instances of these classes.  They are intentionally decoupled from the
internal pipeline models (GenerationResult, RetrievalResult, etc.) so that
internal refactors do not break the API surface.

Factory class-methods on each response model handle the translation from
internal pipeline types to API shapes.
"""

from __future__ import annotations

import statistics
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def split_citations(
    citations: list,
) -> tuple[list[CitationResponse], list[str]]:
    """Partition a list of internal Citation objects into grounded and flagged.

    This is the single authoritative place that maps internal ``Citation``
    objects (from generator.py) to the API's ``CitationResponse`` shape and
    separates hallucination flags. Both the blocking and streaming endpoints
    call this function.

    Args:
        citations: List of ``Citation`` objects with ``hallucination_flag``,
                   ``article_number``, ``source_text``, and ``confidence``.

    Returns:
        ``(grounded, flagged)`` where *grounded* is a list of
        ``CitationResponse`` objects and *flagged* is a list of article number
        strings for articles the model cited but that were absent from the
        retrieved context.
    """
    grounded: list[CitationResponse] = []
    flagged: list[str] = []
    for c in citations:
        if c.hallucination_flag:
            flagged.append(c.article_number)
        else:
            grounded.append(
                CitationResponse(
                    article_number=c.article_number,
                    source_text=c.source_text,
                    confidence=round(c.confidence, 4),
                )
            )
    return grounded, flagged


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /query and POST /query/stream."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "¿Cuál es la pena por homicidio doloso en el CDMX?",
                "filters": None,
                "top_k": 5,
                "stream": False,
            }
        }
    )

    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Natural language question about the Mexican Penal Code (CDMX)",
        examples=[
            "¿Cuál es la pena por homicidio doloso en el CDMX?",
            "¿Qué artículo regula el robo con violencia?",
            "¿Cuándo procede la libertad provisional bajo caución?",
        ],
    )
    filters: Optional[dict] = Field(
        None,
        description=(
            "Optional Pinecone metadata filters. "
            "Examples: {'chapter': 'Título II'} or "
            "{'article_number': {'$gte': '200', '$lte': '250'}}"
        ),
        examples=[
            None,
            {"chapter": "Título Segundo"},
            {"article_number": {"$gte": "200", "$lte": "250"}},
        ],
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of chunks to return after reranking (1–20)",
        examples=[5, 10],
    )
    stream: bool = Field(
        False,
        description="When True the endpoint streams answer tokens via SSE",
        examples=[False, True],
    )

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, v: str) -> str:
        """Remove accidental leading/trailing whitespace."""
        return v.strip()


class EvalRequest(BaseModel):
    """Request body for POST /eval — runs the RAGAS evaluation pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "test_set_path": "data/evaluation/golden_set.json",
                "category": "homicidio",
                "question_id": None,
            }
        }
    )

    test_set_path: str = Field(
        ...,
        description="Relative or absolute path to the golden test set JSON file",
        examples=["data/evaluation/golden_set.json"],
    )
    category: Optional[str] = Field(
        None,
        description="Filter evaluation to a specific question category",
        examples=["homicidio", "robo", "fraude", None],
    )
    question_id: Optional[str] = Field(
        None,
        description="Run evaluation for a single question identified by its ID",
        examples=[None, "q-001", "q-042"],
    )


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------


class CitationResponse(BaseModel):
    """A single validated article citation extracted from the model's answer."""

    model_config = ConfigDict(frozen=True)

    article_number: str = Field(
        ...,
        description="Normalised article identifier, e.g. '123' or '13 BIS'",
        examples=["123", "13 BIS", "200"],
    )
    source_text: str = Field(
        ...,
        description="Text of the retrieved chunk that grounds this citation",
        examples=[
            "Artículo 123. Al que prive de la vida a otro, se le impondrá "
            "de ocho a veinte años de prisión..."
        ],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score of the grounding chunk (0–1)",
        examples=[0.91, 0.78, 0.65],
    )


class RetrievalMetadata(BaseModel):
    """Diagnostic metadata about the retrieval stage of a pipeline run."""

    model_config = ConfigDict(frozen=True)

    chunks_retrieved: int = Field(
        ...,
        description="Total unique candidate chunks before reranking",
        examples=[47, 60],
    )
    chunks_after_rerank: int = Field(
        ...,
        description="Chunks returned after cross-encoder reranking",
        examples=[5, 10],
    )
    avg_similarity_score: float = Field(
        ...,
        description="Mean cosine similarity of the reranked chunks",
        examples=[0.78, 0.82],
    )
    retrieval_time_ms: float = Field(
        ...,
        description="Wall-clock time for the full retrieval stage (ms)",
        examples=[8320.5, 5100.0],
    )
    expanded_queries: list[str] = Field(
        ...,
        description="Alternative phrasings generated by the query-expansion step",
        examples=[
            [
                "pena homicidio doloso CDMX",
                "sanción homicidio intencional código penal",
                "años prisión matar persona México",
            ]
        ],
    )

    @classmethod
    def from_retrieval_result(cls, result: "RetrievalResultProtocol") -> RetrievalMetadata:
        """Build a RetrievalMetadata from an internal RetrievalResult.

        Args:
            result: An object with ``candidates_retrieved``, ``chunks``
                    (each with ``similarity_score``), ``retrieval_time_ms``,
                    and ``expanded_queries`` attributes.

        Returns:
            RetrievalMetadata populated from the retrieval result.
        """
        scores = [c.similarity_score for c in result.chunks]
        avg_score = round(statistics.mean(scores), 4) if scores else 0.0

        return cls(
            chunks_retrieved=result.candidates_retrieved,
            chunks_after_rerank=len(result.chunks),
            avg_similarity_score=avg_score,
            retrieval_time_ms=result.retrieval_time_ms,
            expanded_queries=result.expanded_queries,
        )


# ---------------------------------------------------------------------------
# Top-level response models
# ---------------------------------------------------------------------------


class QueryResponse(BaseModel):
    """Full response for a non-streaming RAG query."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "answer": (
                    "El homicidio doloso se sanciona con prisión de 8 a 20 años "
                    "conforme al Artículo 123 del Código Penal del CDMX."
                ),
                "citations": [
                    {
                        "article_number": "123",
                        "source_text": "Artículo 123. Al que prive...",
                        "confidence": 0.91,
                    }
                ],
                "retrieval": {
                    "chunks_retrieved": 47,
                    "chunks_after_rerank": 5,
                    "avg_similarity_score": 0.78,
                    "retrieval_time_ms": 8320.5,
                    "expanded_queries": ["pena homicidio doloso CDMX"],
                },
                "generation_time_ms": 2100.3,
                "total_time_ms": 10420.8,
                "model": "claude-haiku-4-5-20251001",
                "prompt_version": "v1.v1",
                "hallucination_flags": [],
            }
        },
    )

    answer: str = Field(
        ...,
        description="Natural-language answer from the model",
        examples=[
            "El homicidio doloso se sanciona con prisión de 8 a 20 años "
            "conforme al Artículo 123 del Código Penal del CDMX."
        ],
    )
    citations: list[CitationResponse] = Field(
        ...,
        description="Articles cited in the answer that were found in retrieved chunks",
    )
    retrieval: RetrievalMetadata = Field(
        ..., description="Diagnostic metadata for the retrieval stage"
    )
    generation_time_ms: float = Field(
        ...,
        description="Wall-clock time for the generation step (ms)",
        examples=[2100.3, 1850.0],
    )
    total_time_ms: float = Field(
        ...,
        description="Total wall-clock time for the full pipeline run (ms)",
        examples=[10420.8, 9300.5],
    )
    model: str = Field(
        ...,
        description="Anthropic model ID used for generation",
        examples=["claude-haiku-4-5-20251001"],
    )
    prompt_version: str = Field(
        ...,
        description="Active prompt version string, e.g. 'v1.v1'",
        examples=["v1.v1"],
    )
    hallucination_flags: list[str] = Field(
        ...,
        description=(
            "Article numbers cited by the model that were absent from the "
            "retrieved context — potential hallucinations"
        ),
        examples=[[], ["45", "200"]],
    )

    @classmethod
    def from_generation_result(
        cls,
        result: "GenerationResultProtocol",
        total_time_ms: float,
    ) -> QueryResponse:
        """Build a QueryResponse from an internal GenerationResult.

        Splits citations into grounded ones (returned in ``citations``) and
        hallucinated ones (article numbers only, in ``hallucination_flags``).

        Args:
            result:        Internal ``GenerationResult`` from the pipeline.
            total_time_ms: Full pipeline wall-clock time measured at the
                           endpoint level (retrieval + generation + overhead).

        Returns:
            QueryResponse ready to be serialised by FastAPI.
        """
        grounded, flagged = split_citations(result.citations)

        return cls(
            answer=result.answer,
            citations=grounded,
            retrieval=RetrievalMetadata.from_retrieval_result(result.retrieval),
            generation_time_ms=result.generation_time_ms,
            total_time_ms=round(total_time_ms, 2),
            model=result.model,
            prompt_version=result.prompt_version,
            hallucination_flags=flagged,
        )


class HealthResponse(BaseModel):
    """Response for GET /health — indicates service readiness."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(
        ...,
        description="Overall service status: 'healthy' | 'degraded' | 'unhealthy'",
        pattern="^(healthy|degraded|unhealthy)$",
        examples=["healthy", "degraded", "unhealthy"],
    )
    pinecone_connected: bool = Field(
        ...,
        description="Whether the Pinecone index is reachable",
        examples=[True],
    )
    anthropic_connected: bool = Field(
        ...,
        description="Whether the Anthropic API responded to a probe",
        examples=[True],
    )
    index_vector_count: int = Field(
        ...,
        ge=0,
        description="Number of vectors currently stored in the Pinecone index",
        examples=[12847, 0],
    )
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Seconds since the FastAPI process started",
        examples=[3661.2, 86400.0],
    )
    version: str = Field(
        ...,
        description="Application version from pyproject.toml",
        examples=["0.1.0"],
    )


class ErrorResponse(BaseModel):
    """Uniform error envelope returned on 4xx/5xx responses."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "error": "Validation error",
                "detail": "question: ensure this value has at least 5 characters",
                "request_id": "req-3f2a1b",
            }
        },
    )

    error: str = Field(
        ...,
        description="Short human-readable error category",
        examples=["Validation error", "Rate limit exceeded", "internal_error"],
    )
    detail: Optional[str] = Field(
        None,
        description="Extended description or validation detail",
        examples=[
            "question: ensure this value has at least 5 characters",
            "Retry after 42s (20 req/60s per IP)",
            None,
        ],
    )
    request_id: str = Field(
        ...,
        description="Unique ID for correlating logs with the failed request",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890", "req-3f2a1b"],
    )


# ---------------------------------------------------------------------------
# Protocol stubs (for type-checker only — avoid circular imports at runtime)
# ---------------------------------------------------------------------------

# These are structural type hints used in the factory classmethods above.
# At runtime the real objects (GenerationResult, RetrievalResult) are passed
# in; no import of internal modules is needed here.


class RetrievalResultProtocol:  # pragma: no cover
    candidates_retrieved: int
    chunks: list  # list[RetrievedChunk]
    retrieval_time_ms: float
    expanded_queries: list[str]


class GenerationResultProtocol:  # pragma: no cover
    answer: str
    citations: list  # list[Citation]
    retrieval: RetrievalResultProtocol
    generation_time_ms: float
    model: str
    prompt_version: str

"""Unit tests for src/api/models.py.

Tests cover:
- split_citations: partitions Citation objects into grounded and flagged lists
- CitationResponse: field validation and immutability
- RetrievalMetadata.from_retrieval_result: factory from internal result duck-type
- QueryResponse.from_generation_result: factory from internal result duck-type
- HealthResponse / ErrorResponse: field constraints
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models import (
    CitationResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    RetrievalMetadata,
    QueryResponse,
    split_citations,
)


# ---------------------------------------------------------------------------
# Helpers — minimal duck-typed stand-ins for internal pipeline objects
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, score: float) -> None:
        self.similarity_score = score


class _FakeRetrieval:
    def __init__(self, n_candidates: int = 10, n_chunks: int = 3) -> None:
        self.candidates_retrieved = n_candidates
        self.chunks = [_FakeChunk(0.8 - i * 0.05) for i in range(n_chunks)]
        self.retrieval_time_ms = 1234.5
        self.expanded_queries = ["alternative phrasing 1", "alternative phrasing 2"]


class _FakeCitation:
    def __init__(self, article: str, hallucination: bool) -> None:
        self.article_number = article
        self.source_text = "" if hallucination else f"Artículo {article}. Text…"
        self.confidence = 0.0 if hallucination else 0.75
        self.hallucination_flag = hallucination


class _FakeGeneration:
    def __init__(self, *, hallucinate: bool = False) -> None:
        self.answer = "El homicidio doloso se sanciona con prisión."
        self.citations = [
            _FakeCitation("123", hallucination=False),
            _FakeCitation("999", hallucination=hallucinate),
        ]
        self.retrieval = _FakeRetrieval()
        self.generation_time_ms = 987.6
        self.model = "claude-haiku-4-5-20251001"
        self.prompt_version = "v1.v1"


# ---------------------------------------------------------------------------
# split_citations
# ---------------------------------------------------------------------------


class TestSplitCitations:
    def test_all_grounded(self) -> None:
        citations = [_FakeCitation("123", False), _FakeCitation("124", False)]
        grounded, flagged = split_citations(citations)
        assert len(grounded) == 2
        assert flagged == []

    def test_all_hallucinated(self) -> None:
        citations = [_FakeCitation("999", True), _FakeCitation("998", True)]
        grounded, flagged = split_citations(citations)
        assert grounded == []
        assert flagged == ["999", "998"]

    def test_mixed(self) -> None:
        citations = [
            _FakeCitation("123", False),
            _FakeCitation("999", True),
            _FakeCitation("124", False),
        ]
        grounded, flagged = split_citations(citations)
        assert len(grounded) == 2
        assert flagged == ["999"]

    def test_empty(self) -> None:
        grounded, flagged = split_citations([])
        assert grounded == []
        assert flagged == []

    def test_grounded_confidence_rounded(self) -> None:
        citations = [_FakeCitation("1", False)]
        grounded, _ = split_citations(citations)
        assert isinstance(grounded[0].confidence, float)
        assert grounded[0].confidence == round(0.75, 4)


# ---------------------------------------------------------------------------
# CitationResponse
# ---------------------------------------------------------------------------


class TestCitationResponse:
    def test_valid(self) -> None:
        c = CitationResponse(article_number="123", source_text="Text…", confidence=0.91)
        assert c.article_number == "123"
        assert c.confidence == 0.91

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CitationResponse(article_number="1", source_text="x", confidence=1.5)
        with pytest.raises(ValidationError):
            CitationResponse(article_number="1", source_text="x", confidence=-0.1)

    def test_frozen(self) -> None:
        c = CitationResponse(article_number="1", source_text="x", confidence=0.5)
        with pytest.raises(ValidationError):
            c.confidence = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetrievalMetadata.from_retrieval_result
# ---------------------------------------------------------------------------


class TestRetrievalMetadata:
    def test_from_retrieval_result(self) -> None:
        fake = _FakeRetrieval(n_candidates=20, n_chunks=5)
        meta = RetrievalMetadata.from_retrieval_result(fake)

        assert meta.chunks_retrieved == 20
        assert meta.chunks_after_rerank == 5
        assert meta.retrieval_time_ms == 1234.5
        assert meta.expanded_queries == fake.expanded_queries
        # avg_similarity: _FakeRetrieval with n_chunks=5 builds scores
        # [0.80, 0.75, 0.70, 0.65, 0.60]; mean ≈ 0.70
        assert 0.0 < meta.avg_similarity_score <= 1.0

    def test_empty_chunks_gives_zero_similarity(self) -> None:
        fake = _FakeRetrieval(n_chunks=0)
        meta = RetrievalMetadata.from_retrieval_result(fake)
        assert meta.avg_similarity_score == 0.0
        assert meta.chunks_after_rerank == 0


# ---------------------------------------------------------------------------
# QueryResponse.from_generation_result
# ---------------------------------------------------------------------------


class TestQueryResponse:
    def test_no_hallucinations(self) -> None:
        gen = _FakeGeneration(hallucinate=False)
        resp = QueryResponse.from_generation_result(gen, total_time_ms=2222.1)  # type: ignore[arg-type]

        assert resp.answer == gen.answer
        assert len(resp.citations) == 2
        assert resp.hallucination_flags == []
        assert resp.total_time_ms == 2222.1
        assert resp.model == "claude-haiku-4-5-20251001"
        assert resp.prompt_version == "v1.v1"

    def test_with_hallucination(self) -> None:
        gen = _FakeGeneration(hallucinate=True)
        resp = QueryResponse.from_generation_result(gen, total_time_ms=3000.0)  # type: ignore[arg-type]

        assert len(resp.citations) == 1  # only the grounded one
        assert resp.hallucination_flags == ["999"]

    def test_total_time_rounded(self) -> None:
        gen = _FakeGeneration()
        resp = QueryResponse.from_generation_result(gen, total_time_ms=10420.777)  # type: ignore[arg-type]
        assert resp.total_time_ms == round(10420.777, 2)


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


class TestHealthResponse:
    def test_valid_statuses(self) -> None:
        for status in ("healthy", "degraded", "unhealthy"):
            h = HealthResponse(
                status=status,
                pinecone_connected=True,
                anthropic_connected=True,
                index_vector_count=1000,
                uptime_seconds=3600.0,
                version="0.1.0",
            )
            assert h.status == status

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            HealthResponse(
                status="unknown",
                pinecone_connected=True,
                anthropic_connected=True,
                index_vector_count=0,
                uptime_seconds=0.0,
                version="0.1.0",
            )


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


class TestErrorResponse:
    def test_optional_detail(self) -> None:
        e = ErrorResponse(error="Pipeline error", detail=None, request_id="req-123")
        assert e.detail is None

    def test_with_detail(self) -> None:
        e = ErrorResponse(error="Validation error", detail="field x invalid", request_id="r1")
        assert e.detail == "field x invalid"


# ---------------------------------------------------------------------------
# QueryRequest validation
# ---------------------------------------------------------------------------


class TestQueryRequest:
    def test_strips_whitespace(self) -> None:
        req = QueryRequest(question="  ¿Cuál es la pena?  ")
        assert req.question == "¿Cuál es la pena?"

    def test_min_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="hi")

    def test_max_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 1001)

    def test_top_k_bounds(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="¿Cuál es la pena?", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(question="¿Cuál es la pena?", top_k=21)

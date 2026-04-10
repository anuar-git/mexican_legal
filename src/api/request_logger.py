"""SQLite request/response logger for the Mexican Penal Code RAG API.

Persists every completed query to a local SQLite database so the Phase 6
Streamlit dashboard can query historical data without hitting Prometheus or
log files.

Thread safety: a single ``threading.Lock`` serialises all writes; the
connection is opened once with ``check_same_thread=False``.

Schema (``requests`` table)
---------------------------
id                      TEXT PRIMARY KEY  — UUID from X-Request-ID header
timestamp               TEXT              — ISO 8601 UTC string
question                TEXT              — raw user question
answer                  TEXT              — model answer text
retrieval_time_ms       REAL              — retrieval stage wall time (ms)
generation_time_ms      REAL              — generation stage wall time (ms)
total_time_ms           REAL              — full pipeline wall time (ms)
avg_similarity          REAL              — mean cosine similarity of reranked chunks
num_citations           INTEGER           — grounded citations in the answer
hallucinations_detected INTEGER           — 1 if any hallucination flags, else 0
status_code             INTEGER           — HTTP response status (200, 500, …)
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from src.api.models import QueryResponse


class RequestLogger:
    """Persists query/response pairs to a local SQLite database.

    Args:
        db_path: Path to the SQLite file.  Parent directories are created
                 automatically if they do not exist.
    """

    def __init__(self, db_path: str = "data/requests.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    question TEXT,
                    answer TEXT,
                    retrieval_time_ms REAL,
                    generation_time_ms REAL,
                    total_time_ms REAL,
                    avg_similarity REAL,
                    num_citations INTEGER,
                    hallucinations_detected INTEGER,
                    status_code INTEGER
                )
            """)
            self.conn.commit()

    def log(
        self,
        request_id: str,
        question: str,
        result: QueryResponse,
        status: int,
    ) -> None:
        """Persist a completed request to the database.

        Args:
            request_id: UUID from ``X-Request-ID`` (set by RequestIDMiddleware).
            question:   Original user question text.
            result:     Full ``QueryResponse`` returned to the client.
            status:     HTTP status code (200 for success, 500 for errors, …).
        """
        ts = datetime.now(UTC).isoformat()
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO requests
                    (id, timestamp, question, answer,
                     retrieval_time_ms, generation_time_ms, total_time_ms,
                     avg_similarity, num_citations, hallucinations_detected,
                     status_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    ts,
                    question,
                    result.answer,
                    result.retrieval.retrieval_time_ms,
                    result.generation_time_ms,
                    result.total_time_ms,
                    result.retrieval.avg_similarity_score,
                    len(result.citations),
                    1 if result.hallucination_flags else 0,
                    status,
                ),
            )
            self.conn.commit()

    def close(self) -> None:
        """Close the SQLite connection.  Call during application shutdown."""
        with self._lock:
            self.conn.close()

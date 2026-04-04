"""In-memory operational metrics collector for the Mexican Penal Code RAG API.

Tracks request counts, error counts, latency distributions, and retrieval
similarity scores. Thread-safe via ``threading.Lock``.

Consumed by:
  - GET /v1/metrics  (Prometheus text exposition)
  - Phase 6 Streamlit monitoring dashboard
"""

from __future__ import annotations

import threading


class MetricsCollector:
    """Thread-safe in-memory metrics store.

    Collects per-request latency and retrieval similarity scores and exposes
    them as Prometheus text-format via ``to_prometheus()``.

    Note: ``latencies`` and ``retrieval_scores`` grow unboundedly over the
    process lifetime.  For long-running deployments consider a sliding window
    or rely on the prometheus_client histograms already in ``src/api/main.py``.
    """

    def __init__(self) -> None:
        self.request_count: int = 0
        self.error_count: int = 0
        self.latencies: list[float] = []
        self.retrieval_scores: list[float] = []
        self._lock = threading.Lock()

    def record_request(self, latency_ms: float, similarity_score: float) -> None:
        """Record a successfully completed request."""
        with self._lock:
            self.request_count += 1
            self.latencies.append(latency_ms)
            self.retrieval_scores.append(similarity_score)

    def record_error(self) -> None:
        """Increment the error counter."""
        with self._lock:
            self.error_count += 1

    def get_percentile(self, values: list[float], p: int) -> float:
        """Return the p-th percentile of *values*.

        Args:
            values: List of floats to compute over.
            p:      Integer percentile in [0, 100].

        Returns:
            Percentile value, or ``0.0`` if *values* is empty.
        """
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def to_prometheus(self) -> str:
        """Format current metrics as Prometheus text exposition.

        Returns a ``text/plain; version=0.0.4`` compatible block with
        ``# HELP`` / ``# TYPE`` headers for each metric family.

        Acquires the lock once to snapshot state, then computes percentiles
        outside the lock so heavy sorting never blocks writers.
        """
        with self._lock:
            latencies = list(self.latencies)
            scores = list(self.retrieval_scores)
            req = self.request_count
            err = self.error_count

        avg_sim = sum(scores) / len(scores) if scores else 0.0

        lines = [
            "# HELP rag_requests_total Total RAG requests handled",
            "# TYPE rag_requests_total counter",
            f"rag_requests_total {req}",
            "# HELP rag_errors_total Total failed RAG requests",
            "# TYPE rag_errors_total counter",
            f"rag_errors_total {err}",
            "# HELP rag_latency_p50_ms 50th-percentile end-to-end latency (ms)",
            "# TYPE rag_latency_p50_ms gauge",
            f"rag_latency_p50_ms {self.get_percentile(latencies, 50)}",
            "# HELP rag_latency_p95_ms 95th-percentile end-to-end latency (ms)",
            "# TYPE rag_latency_p95_ms gauge",
            f"rag_latency_p95_ms {self.get_percentile(latencies, 95)}",
            "# HELP rag_latency_p99_ms 99th-percentile end-to-end latency (ms)",
            "# TYPE rag_latency_p99_ms gauge",
            f"rag_latency_p99_ms {self.get_percentile(latencies, 99)}",
            "# HELP rag_avg_similarity Mean retrieval cosine similarity of reranked chunks",
            "# TYPE rag_avg_similarity gauge",
            f"rag_avg_similarity {avg_sim}",
        ]
        return "\n".join(lines)

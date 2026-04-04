# =============================================================================
# Legal Intelligence Engine — Multi-stage Dockerfile
#
# Stage 1 (builder): install all Python dependencies into an isolated venv and
#   pre-download the HuggingFace cross-encoder model so the runtime image has
#   zero cold-start penalty for the first reranking call.
#
# Stage 2 (runtime): copy only the venv, HF model cache, and application
#   source from the builder. The final image carries no build toolchain.
# =============================================================================

# ---- Builder Stage ----------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Build tools required by compiled extensions (e.g. tokenizers, numpy).
# Installed only in the builder; they never reach the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment so the runtime stage gets a clean,
# relocatable directory with nothing from the system Python.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

# ── Dependency-caching layer ─────────────────────────────────────────────────
# Copy only the project manifest and a minimal src stub. Docker caches this
# layer until pyproject.toml changes, so a code-only change doesn't re-run the
# full pip install (which pulls PyTorch + transformers and takes ~3 min).
COPY pyproject.toml .
RUN mkdir -p src && touch src/__init__.py

# Install runtime dependencies only (no dev / test / eval / loadtest extras).
# The stub src/__init__.py satisfies setuptools' package discovery so the
# package metadata (version string) is registered in the venv.
RUN pip install --no-cache-dir .

# ── HuggingFace model pre-download ───────────────────────────────────────────
# Download the cross-encoder at build time into a known cache directory.
# Copying it to the runtime image eliminates the ~200 MB download on first
# request and removes the HF network dependency at runtime.
ENV HF_HOME=/opt/hf_cache
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"


# ---- Runtime Stage ----------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user — defence-in-depth if the container is ever compromised.
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup \
               --no-create-home --shell /bin/false appuser

# Bring in the pre-built venv and the pre-downloaded model cache.
COPY --from=builder /opt/venv    /opt/venv
COPY --from=builder /opt/hf_cache /opt/hf_cache

# Application source and data required at runtime.
COPY src/              /app/src/
COPY data/evaluation/  /app/data/evaluation/
COPY scripts/          /app/scripts/

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"

# Tell sentence-transformers / HuggingFace where the pre-downloaded model is.
ENV HF_HOME=/opt/hf_cache

# Unbuffered stdout/stderr → log lines appear immediately in container logs.
ENV PYTHONUNBUFFERED=1

# Skip writing .pyc bytecode files — keeps the writable layer clean and saves
# a few MB. Bytecode is generated in-memory on import instead.
ENV PYTHONDONTWRITEBYTECODE=1

# Create writable runtime directories:
#   /app/data    — SQLite request log (requests.db)
#   /app/results — eval output files written by POST /v1/evaluate
RUN mkdir -p /app/data /app/results \
    && chown -R appuser:appgroup /app/data /app/results

USER appuser

EXPOSE 8000

# Health check: poll /v1/health every 30 s.
# --start-period=60s gives the app time to finish loading the cross-encoder
# model into memory before Docker starts counting failures.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" \
        2>/dev/null || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

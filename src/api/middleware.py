"""Request middleware, rate limiting, structured logging, and exception handlers.

All components in this module are framework-agnostic (no FastAPI-specific imports
outside the exception handlers) so they can be unit-tested without spinning up
an ASGI server.

Public API consumed by ``src/api/main.py``::

    from src.api.middleware import (
        JSONFormatter,          # log formatter
        configure_logging,      # root-logger setup
        RateLimiter,            # per-IP fixed-window limiter
        RequestIDMiddleware,    # Starlette ASGI middleware
        http_exception_handler,         # register with app.exception_handler
        validation_exception_handler,   # register with app.exception_handler
        generic_exception_handler,      # register with app.exception_handler
    )
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.config import settings
from src.api.models import ErrorResponse

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

# Capture the *instance* attribute names that every LogRecord carries by
# default, so the formatter can skip them when forwarding ``extra`` keys.
_LOG_RECORD_BUILTIN_KEYS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)

# Third-party loggers silenced at WARNING to keep output clean.
_NOISY_LOGGERS = ("httpx", "httpcore", "pinecone", "anthropic", "urllib3")


class JSONFormatter(logging.Formatter):
    """Emit every log record as a single-line JSON object.

    Standard fields present on every line:

    ========= ==============================================================
    timestamp ISO 8601 timestamp (``%Y-%m-%dT%H:%M:%S``)
    level     Log level string (``INFO``, ``WARNING``, …)
    logger    Logger name (``src.api.main``, ``src.api.middleware``, …)
    message   Formatted log message
    ========= ==============================================================

    Named extras added automatically when present on the log record:

    ========== =============================================================
    request_id UUID from ``RequestIDMiddleware`` (set via ``extra=``)
    latency_ms Response time in ms (set by ``RequestIDMiddleware``)
    ========== =============================================================

    Any additional keys passed via ``logger.info("…", extra={…})`` are
    forwarded verbatim.  Unknown non-standard keys that begin with ``_``
    are silently dropped.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Forward extra keys injected by the caller via logger.xxx(..., extra={})
        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_BUILTIN_KEYS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up structured JSON logging on the root logger.

    Replaces all existing handlers with a single ``StreamHandler`` using
    ``JSONFormatter``.  Safe to call multiple times (each call rebuilds the
    handler list from scratch).

    Args:
        level: Root logger level.  Defaults to ``logging.INFO``.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Synchronous fixed-window rate limiter keyed by client IP.

    Uses a ``threading.Lock`` so it is safe to call from both the asyncio
    event loop and from thread-pool workers (e.g. FastAPI ``BackgroundTasks``).

    Because the lock is never held across an ``await`` point, calling
    ``is_allowed()`` from async code does **not** block the event loop.

    Args:
        max_requests:   Maximum requests allowed per window per IP.
        window_seconds: Window duration in seconds.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_allowed(self, client_ip: str) -> bool:
        """Check and record a request.  Returns ``True`` if within the limit.

        Evicts timestamps older than the window, then either:

        - **Allows**: records the current timestamp and returns ``True``.
        - **Denies**: returns ``False`` without recording (the caller can
          follow up with ``time_until_next()`` for the ``Retry-After`` value).

        Args:
            client_ip: Client IP address string used as the bucket key.

        Returns:
            ``True`` if the request is allowed; ``False`` if rate-limited.
        """
        now = time.time()
        with self._lock:
            timestamps = [
                t for t in self.requests.get(client_ip, []) if now - t < self.window
            ]
            if len(timestamps) >= self.max_requests:
                self.requests[client_ip] = timestamps
                return False
            timestamps.append(now)
            self.requests[client_ip] = timestamps
            return True

    def time_until_next(self, client_ip: str) -> float:
        """Return seconds until the next request from this IP is permitted.

        Does **not** modify state — safe to call immediately after
        ``is_allowed()`` returns ``False``.

        The reset time is derived from the oldest timestamp still inside the
        current window: once that timestamp expires, a slot opens up.

        Args:
            client_ip: Client IP address string.

        Returns:
            Non-negative float.  Returns ``0.0`` if no window data exists
            (the next request would be allowed immediately).
        """
        now = time.time()
        with self._lock:
            q = self.requests.get(client_ip)
            if not q:
                return 0.0
            return max(self.window - (now - q[0]), 0.0)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------

_mw_logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a UUID4 request ID to every request/response pair.

    Behaviour:

    1. **Incoming**: reads ``X-Request-ID`` header when present (supports
       distributed tracing where upstream services set a correlation ID);
       otherwise generates a fresh ``str(uuid.uuid4())``.
    2. **Request state**: stores the ID on ``request.state.request_id`` so
       any downstream handler or dependency can read it for error responses
       and log correlation.
    3. **Response**: echoes the ID back as ``X-Request-ID`` on every response
       regardless of status code.
    4. **Access log**: emits a structured JSON log entry after every response
       with ``request_id``, ``method``, ``path``, ``status``, and
       ``latency_ms`` (picked up by ``JSONFormatter``).

    Args:
        app: The wrapped ASGI application (provided by FastAPI automatically).
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        t0 = time.monotonic()
        response = await call_next(request)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        response.headers["X-Request-ID"] = request_id

        _mw_logger.info(
            "%s %s → %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response


# ---------------------------------------------------------------------------
# Exception handlers
#
# Register in main.py with:
#     app.exception_handler(HTTPException)(http_exception_handler)
#     app.exception_handler(RequestValidationError)(validation_exception_handler)
#     app.exception_handler(Exception)(generic_exception_handler)
# ---------------------------------------------------------------------------

# Show str(exc) in the `detail` field only when settings.debug is True.
# In production this is always False so stack traces are never leaked.
_DEBUG: bool = settings.debug

_eh_logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a uniform ``ErrorResponse`` envelope for all ``HTTPException``s.

    If ``exc.detail`` is already an ``ErrorResponse``-shaped dict (i.e. it
    has an ``"error"`` key), it is returned verbatim.  This preserves the
    rich structured payloads that endpoint handlers embed in their
    ``HTTPException`` details.

    Args:
        request: The incoming Starlette request.
        exc:     The raised ``HTTPException``.

    Returns:
        ``JSONResponse`` with the appropriate HTTP status code.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    detail = exc.detail

    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=detail,
            headers=dict(exc.headers or {}),
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(detail),
            detail=None,
            request_id=request_id,
        ).model_dump(),
        headers=dict(exc.headers or {}),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a 422 ``ErrorResponse`` with a human-readable validation message.

    Extracts the first validation error from Pydantic's error list and
    formats it as ``"field → sub-field: message"`` for clarity.

    Args:
        request: The incoming Starlette request.
        exc:     The Pydantic ``RequestValidationError``.

    Returns:
        422 ``JSONResponse`` with an ``ErrorResponse`` body.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = " → ".join(str(p) for p in first.get("loc", []))
    msg = first.get("msg", "invalid input")

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation error",
            detail=f"{loc}: {msg}" if loc else msg,
            request_id=request_id,
        ).model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all 500 handler — never exposes a stack trace in production.

    - **Production** (``DEBUG`` env var unset or falsy): ``detail`` is set to
      a generic ``"An unexpected error occurred"`` message.
    - **Debug** (``DEBUG=true``): ``detail`` includes ``str(exc)`` to aid
      local development.

    Always logs the full traceback at ``ERROR`` level regardless of mode.

    Args:
        request: The incoming Starlette request.
        exc:     The unhandled exception.

    Returns:
        500 ``JSONResponse`` with an ``ErrorResponse`` body.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    _eh_logger.exception(
        "Unhandled exception",
        extra={
            "request_id": request_id,
            "path": str(request.url.path),
            "exc_type": type(exc).__name__,
        },
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail=str(exc) if _DEBUG else "An unexpected error occurred",
            request_id=request_id,
        ).model_dump(),
    )

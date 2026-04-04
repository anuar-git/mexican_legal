"""Centralised application configuration via pydantic-settings.

All configuration is loaded once at import time from environment variables
and/or a ``.env`` file.  Individual modules import the ``settings`` singleton
rather than calling ``os.getenv`` directly.

Validation
----------
Required fields (``pinecone_api_key``, ``anthropic_api_key``) have no
default: if they are absent from the environment at startup, pydantic-settings
raises a ``ValidationError`` immediately, which prevents the container from
starting in a misconfigured state.

Precedence (highest → lowest)
------------------------------
1. Actual environment variables (set by Docker / Railway / the shell)
2. Variables in the ``.env`` file (used for local development only)
3. Field defaults defined here
"""

from __future__ import annotations

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── External services (required — no defaults) ──────────────────────────
    pinecone_api_key: str
    anthropic_api_key: str

    # ── Pinecone index ───────────────────────────────────────────────────────
    pinecone_index_name: str = "legal-intelligence"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ── Admin ────────────────────────────────────────────────────────────────
    # Required by POST /v1/evaluate; the endpoint returns 503 if this is empty.
    admin_api_key: str = ""

    # ── Runtime behaviour ────────────────────────────────────────────────────
    environment: str = "development"   # "development" | "production"
    log_level: str = "INFO"            # DEBUG | INFO | WARNING | ERROR
    debug: bool = False                # Exposes stack traces in 500 responses

    # ── API / rate limiting ──────────────────────────────────────────────────
    # ALLOWED_ORIGINS accepts a comma-separated string (e.g. "https://a.com,https://b.com").
    # The single value "*" (default) allows all origins — lock down in production.
    allowed_origins: str = "*"
    rate_limit_requests: int = 10   # max requests per window per IP
    rate_limit_window_s: int = 60   # rolling window in seconds

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Field names are compared case-insensitively to env var names,
        # so PINECONE_API_KEY matches pinecone_api_key.
        case_sensitive=False,
        # Extra env vars in the environment (PATH, PYTHONPATH, HOME, …)
        # are silently ignored rather than raising a validation error.
        extra="ignore",
    )

    @property
    def log_level_int(self) -> int:
        """Return the ``logging`` module integer level for ``self.log_level``."""
        return getattr(logging, self.log_level.upper(), logging.INFO)

    @property
    def allowed_origins_list(self) -> list[str]:
        """Split the comma-separated ALLOWED_ORIGINS string into a list."""
        return [o.strip() for o in self.allowed_origins.split(",")]


# Module-level singleton — imported by main.py and middleware.py.
settings = Settings()

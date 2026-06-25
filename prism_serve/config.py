"""Runtime configuration for prism-serve.

Settings are read from environment variables (prefix ``PRISM_SERVE_``) with sane
defaults, so the gateway can boot locally with zero configuration. Backing services
(Redis / NATS) are placeholders for now and unused until the scheduler lands.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration.

    Override any field with an env var, e.g. ``PRISM_SERVE_PORT=9090``.
    """

    model_config = SettingsConfigDict(env_prefix="PRISM_SERVE_", env_file=".env")

    # Gateway
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    # Backing services
    redis_url: str = "redis://localhost:6379/0"
    nats_url: str = "nats://localhost:4222"

    # Engine RPC (prism-infer worker endpoint; placeholder)
    engine_endpoint: str = "http://localhost:8000"


settings = Settings()

"""Runtime configuration for prism-serve."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Process-wide configuration.

    Override any field with an env var, e.g. ``PRISM_SERVE_PORT=9090``.
    """

    model_config = SettingsConfigDict(env_prefix="PRISM_SERVE_", env_file=".env")

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    redis_url: str = "redis://localhost:6379/0"
    nats_url:  str = "nats://localhost:4222"
    nats_connect_timeout_s: float = 2.0
    nats_max_reconnect_attempts: int = 60
    # Production fails closed; local gateway-only runs must opt into mock mode.
    nats_required: bool = True

    engine_endpoint: str = "http://localhost:8000"

    # Separate thresholds prevent flow-control oscillation.
    high_watermark: float = 0.85
    low_watermark:  float = 0.70

    # Per-dst in-flight byte cap (secondary guard, bytes)
    max_bytes_inflight: int = 256 * 1024 * 1024  # 256 MB

    kv_transfer_timeout_s: float = 30.0
    prefill_timeout_s: float = 30.0
    max_dispatch_attempts: int = 3
    recompute_timeout_s: float = 30.0
    decode_timeout_s: float = 300.0
    abort_request_timeout_s: float = 5.0
    max_recompute_attempts: int = 2

    schedule_loop_tick_ms: int = 10   # milliseconds per tick

    governor_tick_s: float = 5.0
    shutdown_drain_timeout_s: float = 60.0

    slot_stale_timeout_s: float = 300.0

    min_decode_instances: int = 1
    max_decode_instances: int = 64
    # Usable KV memory per decode instance (bytes); default 56 GB
    kv_per_instance_bytes: int = 56 * 1024 ** 3

settings = Settings()

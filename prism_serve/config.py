"""Runtime configuration for prism-serve.

Settings are read from environment variables (prefix ``PRISM_SERVE_``) with sane
defaults, so the gateway can boot locally with zero configuration.

"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Process-wide configuration.

    Override any field with an env var, e.g. ``PRISM_SERVE_PORT=9090``.
    """

    model_config = SettingsConfigDict(env_prefix="PRISM_SERVE_", env_file=".env")

    # ── Gateway ───────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    # ── Backing services ──────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    nats_url:  str = "nats://localhost:4222"

    # ── Engine RPC  ──
    engine_endpoint: str = "http://localhost:8000"

    # ── TransferGovernor: dynamic watermark ──────────────────────────
    # D-instance KV usage above HIGH_WATERMARK → pause sending.
    # Below LOW_WATERMARK → resume (hysteresis prevents oscillation).
    high_watermark: float = 0.85
    low_watermark:  float = 0.70

    # Per-dst in-flight byte cap (secondary guard, bytes)
    max_bytes_inflight: int = 256 * 1024 * 1024  # 256 MB

    # ── Recompute fallback ────────────────────────────────────────────
    # KV transfer considered stuck after this many seconds.
    kv_transfer_timeout_s: float = 30.0
    # Max times a single request may be recomputed before abort.
    max_recompute_attempts: int = 2

    # ── schedule_loop (← Ray Serve reconcile loop interval) ──────────
    schedule_loop_tick_ms: int = 10   # milliseconds per tick

    # ── TransferGovernor independent tick (deferred queue flush) ─────
    governor_tick_s: float = 5.0

    # ── InstanceSlot leak detection ───────────────────────────────────
    slot_stale_timeout_s: float = 300.0

    # ── Adaptive decode instance count (← Flink AdaptiveBatch) ───────
    min_decode_instances: int = 1
    max_decode_instances: int = 64
    # Usable KV memory per decode instance (bytes); default 56 GB
    kv_per_instance_bytes: int = 56 * 1024 ** 3

settings = Settings()

"""Runtime configuration for prism-serve."""

from __future__ import annotations

import uuid

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Process-wide configuration.

    Override any field with an env var, e.g. ``PRISM_SERVE_PORT=9090``.
    """

    model_config = SettingsConfigDict(env_prefix="PRISM_SERVE_", env_file=".env")

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"

    nats_url:  str = "nats://localhost:4222"
    nats_connect_timeout_s: float = 2.0
    nats_max_reconnect_attempts: int = 60
    gateway_pod_uid: str = ""
    gateway_process_generation: str = str(uuid.uuid4())
    # Slot and quarantine state is process-local until leader failover is added.
    control_plane_replica_count: int = 1
    # Production fails closed; local gateway-only runs must opt into mock mode.
    nats_required: bool = True

    engine_endpoint: str = "http://localhost:8000"


    multinode_e2e_enabled: bool = False
    worker_topology_path: str = ""
    topology_generation: str = ""
    infer_rpc_timeout_s: float = 5.0
    operation_query_interval_ms: int = 100
    active_operation_cap: int = 512
    operation_reorder_window: int = 4096
    terminal_snapshot_cap: int = 4096
    replacement_store_path: str = ""
    replacement_store_max_records_per_run: int = 1024
    replacement_store_seal_retention: int = 2

    # Kubernetes Secret; it is never accepted from a request body or chart value.
    correctness_harness_enabled: bool = False
    # Performance and route parity share auth but never enable fault injection.
    performance_harness_enabled: bool = False
    route_parity_harness_enabled: bool = False
    performance_trace_cap: int = 8192
    correctness_harness_secret: str = ""
    correctness_fault_gate_timeout_s: float = 1200.0
    resource_report_stale_after_s: float = 2.0
    transfer_abort_timeout_s: float = 5.0
    nccl_watchdog_timeout_s: float = 30.0
    require_gpudirect_rdma: bool = False
    allowed_fallback_transport: str = "NCCL_SOCKET"
    model_profile_id: str = "week12-qwen3-0.6b"
    model_id: str = "Qwen/Qwen3-0.6B"
    model_revision: str = "9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439"
    model_config_sha256: str = (
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
    )
    runtime_dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    kv_layout: str = "NHDC"
    kv_block_bytes: int = 29_360_128
    model_num_hidden_layers: int = 28
    model_num_key_value_heads: int = 8
    model_head_dim: int = 128
    model_rope_theta: float = 1_000_000.0
    max_model_len: int = 4096
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    tokenizer_model: str = ""
    tokenizer_revision: str = "9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439"
    chat_template_version: str = "v1"
    kv_compatibility_id: str = (
        "a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19"
    )
    prefix_block_size: int = 256




    high_watermark: float = 0.85
    low_watermark:  float = 0.70


    max_bytes_inflight_per_pair: int = 1024 * 1024 * 1024



    max_bytes_inflight: int = 256 * 1024 * 1024  # 256 MB

    # ── recompute fallback ───────────────────────────────────────────────
    # Unconfirmed transfers fall back after this deadline.
    kv_transfer_timeout_s: float = 30.0
    abort_transfer_timeout_s: float = 5.0
    kv_usage_stale_after_s: float = 10.0
    # Retry lost dispatch/prefill_done messages against the original assignment.
    prefill_timeout_s: float = 30.0
    max_dispatch_attempts: int = 3
    recompute_timeout_s: float = 30.0
    decode_timeout_s: float = 300.0
    abort_request_timeout_s: float = 5.0
    reconciliation_timeout_s: float = 5.0
    max_recompute_attempts: int = 2

    schedule_loop_tick_ms: int = 10   # milliseconds per tick

    governor_tick_s: float = 5.0
    shutdown_drain_timeout_s: float = 60.0

    slot_stale_timeout_s: float = 300.0

    affinity_enabled: bool = False
    locality_wait_ms: int = 20
    max_affinity_wait_ms: int = 100
    affinity_safety_margin_ms: float = 1.0
    affinity_decode_candidate_limit: int = 8
    decode_slot_lease_timeout_s: float = 5.0
    prefix_event_log_capacity: int = 65536
    prefix_event_poll_interval_ms: int = 100
    prefix_consumer_lease_s: float = 30.0
    prefix_full_report_interval_s: float = 300.0
    prefix_load_timeout_s: float = 5.0
    suffix_prefill_timeout_s: float = 30.0
    prefix_operation_watchdog_s: float = 1.0
    prefix_block_bytes: int = 0
    prefill_ms_per_token: float = 0.05

    # Immutable build/deployment identity used by the operator benchmark. The
    # identity endpoint stays unavailable until every field is supplied.
    image_source_url: str = "https://github.com/SparkSnail/prism-serve"
    image_source_commit: str = ""
    image_digest: str = ""
    worker_image_source_url: str = "https://github.com/SparkSnail/prism-infer"
    worker_image_source_commit: str = ""
    worker_image_digest: str = ""

    min_decode_instances: int = 1
    max_decode_instances: int = 64
    # Usable KV memory per decode instance (bytes); default 56 GB
    kv_per_instance_bytes: int = 56 * 1024 ** 3

settings = Settings()

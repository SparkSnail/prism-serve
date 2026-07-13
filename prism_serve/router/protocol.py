"""Cached-prefix control protocol DTOs and explicit states."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MappedTransferStatus(str, Enum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FENCED = "FENCED"
    UNKNOWN = "UNKNOWN"


class PrefixOperationStatus(str, Enum):
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True, frozen=True)
class CachedPrefixDecision:
    source_instance: str
    source_epoch: str
    decode_instance: str
    cached_prefix_blocks: int
    cached_prefix_tokens: int
    transfer_bytes: int
    estimated_gain_ms: float


@dataclass(slots=True, frozen=True)
class MappedPrefixTransferReq:
    operation_id: str
    req_id: str
    source_instance: str
    source_epoch: str
    target_instance: str
    target_epoch: str
    src_block_ids: tuple[int, ...]
    dst_block_ids: tuple[int, ...]
    namespace: str
    kv_compatibility_id: str
    request_context_digest: str

    def __post_init__(self) -> None:
        assert self.operation_id, "mapped transfer requires operation_id"
        assert len(self.src_block_ids) == len(self.dst_block_ids), (
            f"mapped block count mismatch: {self.src_block_ids=} {self.dst_block_ids=}"
        )


@dataclass(slots=True, frozen=True)
class ResolvedPrefix:
    operation_id: str
    source_epoch: str
    src_block_ids: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class PreparedPrefix:
    operation_id: str
    mode: str
    dst_block_ids: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class CachedPrefixPlan:
    operation_id: str
    req_id: str
    source_instance: str
    target_instance: str
    source_epoch: str
    target_epoch: str
    src_block_ids: tuple[int, ...]
    dst_block_ids: tuple[int, ...]
    cached_prefix_tokens: int
    namespace: str = ""
    kv_compatibility_id: str = ""
    request_context_digest: str = ""
    token_ids: tuple[int, ...] = ()
    sampling_params: dict = field(default_factory=dict)
    mode: str = "remote_transfer"


@dataclass(slots=True, frozen=True)
class ExpectedPrefixBlock:
    block_index: int
    chain_hash: int
    token_ids: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class CleanupOutcome:
    action: str
    target_state: PrefixOperationStatus

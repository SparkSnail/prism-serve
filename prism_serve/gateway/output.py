"""Gateway-side projection of cumulative decode output."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from prism_serve.scheduler.sequence_state import RequestInfo, SeqState


OutputQueryIdentity = tuple[str, str, str, str]


def output_query_identity(
    request_info: RequestInfo | None,
) -> OutputQueryIdentity | None:
    if request_info is None or request_info.state not in {
        SeqState.DECODING, SeqState.FINISHED,
    }:
        return None
    identity = (
        request_info.decode_instance,
        request_info.decode_instance_epoch,
        request_info.req_id,
        request_info.active_operation_id,
    )
    return identity if all(identity) else None


@dataclass(slots=True)
class OutputState:
    token_ids: list[int] = field(default_factory=list)
    terminal: bool = False
    error: str | None = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class GatewayOutputCapacity(RuntimeError):
    pass


class GatewayOutputBuffer:
    def __init__(
        self, *, active_operation_cap: int = 512,
        terminal_snapshot_cap: int = 4096,
    ) -> None:
        if active_operation_cap <= 0 or terminal_snapshot_cap <= 0:
            raise ValueError("output buffer caps must be positive")
        self.active_operation_cap = active_operation_cap
        self.terminal_snapshot_cap = terminal_snapshot_cap
        self._states: dict[str, OutputState] = {}
        self._resource_free_terminal: OrderedDict[str, None] = OrderedDict()

    def ensure(self, req_id: str) -> OutputState:
        existing = self._states.get(req_id)
        if existing is not None:
            return existing
        active = len(self._states) - len(self._resource_free_terminal)
        if active >= self.active_operation_cap:
            raise GatewayOutputCapacity("active output state capacity exhausted")
        state = OutputState()
        self._states[req_id] = state
        return state

    def mark_resource_free(self, req_id: str) -> None:
        """Make only a terminal, resource-free stream eligible for LRU eviction."""
        state = self._states.get(req_id)
        if state is None:
            return
        if not state.terminal and state.error is None:
            raise ValueError("active output state cannot be evicted")
        self._resource_free_terminal[req_id] = None
        self._resource_free_terminal.move_to_end(req_id)
        while len(self._resource_free_terminal) > self.terminal_snapshot_cap:
            expired, _ = self._resource_free_terminal.popitem(last=False)
            self._states.pop(expired, None)

    def state_counts(self) -> dict[str, int]:
        return {
            "active_or_held": len(self._states) - len(self._resource_free_terminal),
            "resource_free_terminal": len(self._resource_free_terminal),
        }

    async def apply_cumulative(
        self,
        req_id: str,
        token_ids: list[int],
        output_seq_no: int,
        *,
        terminal: bool = False,
        error: str | None = None,
        still_current: Callable[[], bool] | None = None,
    ) -> bool:
        if still_current is not None and not still_current():
            raise ValueError("output query request changed")
        state = self.ensure(req_id)
        if output_seq_no != len(token_ids):
            raise ValueError("output_seq_no must equal cumulative token count")
        async with state.condition:
            if still_current is not None and not still_current():
                raise ValueError("output query request changed")
            if output_seq_no < len(state.token_ids):
                return False
            if token_ids[:len(state.token_ids)] != state.token_ids:
                raise ValueError("cumulative output changed committed prefix")
            changed = output_seq_no > len(state.token_ids) or terminal != state.terminal
            state.token_ids = list(token_ids)
            state.terminal = state.terminal or terminal
            state.error = error or state.error
            state.condition.notify_all()
            return changed

    async def wait_next(
        self, req_id: str, cursor: int
    ) -> tuple[list[int], bool, str | None]:
        state = self.ensure(req_id)
        async with state.condition:
            await state.condition.wait_for(
                lambda: len(state.token_ids) > cursor or state.terminal or state.error
            )
            return list(state.token_ids[cursor:]), state.terminal, state.error

    def snapshot(self, req_id: str) -> tuple[list[int], bool, str | None]:
        state = self.ensure(req_id)
        return list(state.token_ids), state.terminal, state.error

    async def fail(self, req_id: str, error: str) -> None:
        """Wake an old-world stream with a durable terminal error."""
        state = self.ensure(req_id)
        async with state.condition:
            state.terminal = True
            state.error = state.error or error
            state.condition.notify_all()


async def repair_output_gap(
    infer_client,
    *,
    instance_id: str,
    instance_epoch: str,
    req_id: str,
    operation_id: str,
    cursor: int,
    output_buffer: GatewayOutputBuffer,
    metrics=None,
    still_current: Callable[[], bool] | None = None,
) -> bool:
    """Query authoritative cumulative output after a missing/bad event."""
    value = await infer_client.request_output(instance_id, req_id, cursor)
    if (
        value.get("req_id") != req_id
        or value.get("instance_epoch") != instance_epoch
        or value.get("operation_id") != operation_id
    ):
        raise ValueError("output query identity changed")
    changed = await output_buffer.apply_cumulative(
        req_id,
        list(value["token_ids"]),
        int(value["output_seq_no"]),
        terminal=bool(value.get("terminal", False)),
        error=value.get("error"),
        still_current=still_current,
    )
    if metrics is not None:
        metrics.increment(
            "output_gap_repair_total", labels={"source": "query"}
        )
    return changed

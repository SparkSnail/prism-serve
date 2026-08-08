"""Typed HTTP client for prism-infer workers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import re
import time
from urllib.parse import urlsplit

import httpx


logger = logging.getLogger(__name__)

_INVALID_RPC_ERROR_CODE = "<invalid-error-code>"
_MAX_LOGGED_RPC_ERROR_CODE_LENGTH = 64
_LOGGABLE_RPC_ERROR_CODES = frozenset({
    "CONFLICT",
    "FENCED_WORKER_EPOCH",
    "HTTP_ERROR",
    "INVALID_OUTPUT",
    "INVALID_RESPONSE",
    "INVALID_SCHEMA",
    "NOT_FOUND",
    "NOT_READY",
    "PRECONDITION_FAILED",
    "REGISTRY_CAPACITY",
    "REGISTRY_ERROR",
    "REQUEST_OUTPUT_NOT_FOUND",
    "RETIRED_OWNER",
    "STALE_OPERATION",
    "UNAVAILABLE",
    "UNKNOWN_OWNER",
    "WORKER_NOT_READY",
})
_RPC_ERROR_CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*", re.ASCII)


def _safe_rpc_error_code(value: str) -> str:
    if not 1 <= len(value) <= _MAX_LOGGED_RPC_ERROR_CODE_LENGTH:
        return _INVALID_RPC_ERROR_CODE
    if _RPC_ERROR_CODE_PATTERN.fullmatch(value) is None:
        return _INVALID_RPC_ERROR_CODE
    if value not in _LOGGABLE_RPC_ERROR_CODES:
        return _INVALID_RPC_ERROR_CODE
    return value


def _rpc_error_chain(error: BaseException) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    pending: BaseException | None = error
    visited: set[int] = set()
    while pending is not None and id(pending) not in visited and len(result) < 8:
        visited.add(id(pending))
        item: dict[str, object] = {
            "module": type(pending).__module__,
            "type": type(pending).__name__,
        }
        errno = getattr(pending, "errno", None)
        if type(errno) is int:
            item["errno"] = errno
        result.append(item)
        pending = pending.__cause__ or pending.__context__
    return result


def _safe_rpc_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return "<invalid-endpoint>"
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not hostname:
        return "<invalid-endpoint>"
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{scheme}://{host}" + (f":{port}" if port is not None else "")


def _log_rpc_diagnostic(level: int, value: dict[str, object]) -> None:
    logger.log(
        level,
        "%s",
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    )


@dataclass(slots=True, frozen=True)
class EndpointOperationRef:
    topology_generation: str
    owner_generation: str
    operation_seq: int
    target_instance: str
    target_worker_epoch: str
    operation_id: str
    payload_digest: str

    def __post_init__(self) -> None:
        assert self.operation_seq > 0
        assert self.payload_digest


_RELEASE_SNAPSHOT_FIELDS = frozenset({
    "cleanup_id",
    "operation_id",
    "lease_id",
    "endpoint_epoch",
    "released_resource_kinds",
    "released_counts",
    "resources_held_after",
    "payload_digest",
})


def finalize_release_payload(
    *,
    cleanup_id: str,
    operation_id: str,
    lease_id: str,
    endpoint_refs: tuple[EndpointOperationRef, ...],
    resource_kinds: tuple[str, ...],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "cleanup_id": cleanup_id,
        "operation_id": operation_id,
        "lease_id": lease_id,
        "endpoint_refs": [asdict(ref) for ref in endpoint_refs],
        "resource_kinds": sorted(set(resource_kinds)),
        "release_basis": "ENDPOINT_TERMINAL",
    }
    payload["payload_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def validate_release_snapshot(
    value: dict[str, object],
    *,
    instance_id: str,
    cleanup_id: str,
    operation_id: str,
    lease_id: str,
    endpoint_refs: tuple[EndpointOperationRef, ...],
    resource_kinds: tuple[str, ...],
    expected_endpoint_epoch: str | None = None,
) -> dict[str, object]:
    """Bind a successful finalize response to its exact request and worker."""
    if set(value) != _RELEASE_SNAPSHOT_FIELDS:
        raise ValueError("invalid ReleaseSnapshot fields")
    if not endpoint_refs:
        raise ValueError("ReleaseSnapshot requires endpoint refs")
    endpoint_epoch = expected_endpoint_epoch or endpoint_refs[0].target_worker_epoch
    if not endpoint_epoch:
        raise ValueError("ReleaseSnapshot endpoint epoch is empty")
    if any(
        ref.target_instance != instance_id
        or ref.target_worker_epoch != endpoint_epoch
        or ref.operation_id != operation_id
        for ref in endpoint_refs
    ):
        raise ValueError("ReleaseSnapshot request identity mismatch")
    payload = finalize_release_payload(
        cleanup_id=cleanup_id,
        operation_id=operation_id,
        lease_id=lease_id,
        endpoint_refs=endpoint_refs,
        resource_kinds=resource_kinds,
    )
    expected_kinds = sorted(set(resource_kinds))
    if not expected_kinds:
        raise ValueError("ReleaseSnapshot requires resource kinds")
    if (
        value["cleanup_id"] != cleanup_id
        or value["operation_id"] != operation_id
        or value["lease_id"] != lease_id
        or value["endpoint_epoch"] != endpoint_epoch
        or value["payload_digest"] != payload["payload_digest"]
        or value["resources_held_after"] is not False
        or value["released_resource_kinds"] != expected_kinds
    ):
        raise ValueError("ReleaseSnapshot does not match finalize request")
    released_counts = value["released_counts"]
    if not isinstance(released_counts, list):
        raise ValueError("ReleaseSnapshot released_counts must be an array")
    parsed_counts: list[tuple[str, int]] = []
    for item in released_counts:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 0
        ):
            raise ValueError("ReleaseSnapshot released_counts is invalid")
        parsed_counts.append((item[0], item[1]))
    if (
        [kind for kind, _ in parsed_counts] != expected_kinds
        or len(dict(parsed_counts)) != len(parsed_counts)
    ):
        raise ValueError("ReleaseSnapshot released counts do not match resources")
    return dict(value)


class EndpointSequenceAllocator:

    def __init__(self, topology_generation: str, owner_generation: str) -> None:
        assert topology_generation and owner_generation
        self.topology_generation = topology_generation
        self.owner_generation = owner_generation
        self._next: dict[str, int] = {}

    def allocate(
        self,
        *,
        target_instance: str,
        target_worker_epoch: str,
        operation_id: str,
        payload: dict[str, object],
    ) -> EndpointOperationRef:
        sequence = self._next.get(target_instance, 1)
        self._next[target_instance] = sequence + 1
        digest = "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return EndpointOperationRef(
            topology_generation=self.topology_generation,
            owner_generation=self.owner_generation,
            operation_seq=sequence,
            target_instance=target_instance,
            target_worker_epoch=target_worker_epoch,
            operation_id=operation_id,
            payload_digest=digest,
        )


class InferRPCError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(f"infer RPC {status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code


class AmbiguousRPCError(RuntimeError):
    def __init__(self, endpoint_ref: EndpointOperationRef | None, message: str):
        super().__init__(message)
        self.endpoint_ref = endpoint_ref


CorrectnessPostSuccessHook = Callable[
    [dict[str, object]], Awaitable[bool]
]


class HttpInferClient:

    def __init__(
        self,
        endpoints: dict[str, str],
        *,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        assert endpoints
        assert timeout_s > 0
        self._endpoints = {key: value.rstrip("/") for key, value in endpoints.items()}
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)
        self._metrics = None
        self._correctness_post_success_hook: CorrectnessPostSuccessHook | None = None

        self._rpc_failure_counts: dict[tuple[str, str, str], int] = {}

    def set_metrics(self, metrics) -> None:
        self._metrics = metrics

    def set_correctness_post_success_hook(
        self, hook: CorrectnessPostSuccessHook | None
    ) -> None:
        """Install an explicit correctness-only response-discard seam.

        The hook runs only after a successful mutation response has been parsed,
        but before the control path observes it.  Returning true models an
        ambiguous post-success response loss without changing the remote truth.
        """
        self._correctness_post_success_hook = hook

    async def close(self) -> None:
        await self._client.aclose()

    def _url(self, instance_id: str, path: str) -> str:
        try:
            base = self._endpoints[instance_id]
        except KeyError as exc:
            raise ValueError(f"unknown infer endpoint: {instance_id}") from exc
        return f"{base}{path}"

    @staticmethod
    def _metric_endpoint(path: str) -> str:
        base = path.split("?", 1)[0]
        if base == "/v1/operations":
            return "/v1/operations"
        if base == "/v1/prefix/events":
            return "/v1/prefix/events"
        parts = base.split("/")
        if len(parts) >= 5 and parts[1:3] == ["v1", "requests"]:
            if parts[4] in {"abort", "output"}:
                return f"/v1/requests/{{operation_id}}/{parts[4]}"
        if len(parts) >= 5 and parts[1:3] == ["v1", "transfers"]:
            if parts[4] == "abort":
                return "/v1/transfers/{operation_id}/abort"
        if len(parts) >= 5 and parts[1:3] == ["v1", "owners"]:
            if parts[4] == "retire":
                return "/v1/owners/{owner_generation}/retire"
        if len(parts) == 4 and parts[1:3] in (
            ["v1", "requests"], ["v1", "transfers"]
        ):
            return f"/{parts[1]}/{parts[2]}/{{operation_id}}"
        if len(parts) == 5 and parts[1:4] == ["v1", "prefix", "status"]:
            return "/v1/prefix/status/{operation_id}"
        return base

    async def _request(
        self,
        method: str,
        instance_id: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        endpoint_ref: EndpointOperationRef | None = None,
    ) -> dict[str, object]:
        metric_endpoint = self._metric_endpoint(path)
        diagnostic_key = (instance_id, method, metric_endpoint)
        retry_index = self._rpc_failure_counts.get(diagnostic_key, 0)
        base_diagnostic: dict[str, object] = {
            "instance_id": instance_id,
            "method": method,
            "path": metric_endpoint,
            "endpoint": _safe_rpc_endpoint(self._endpoints.get(instance_id, "")),
            "retry_index": retry_index,
        }
        started_ns = time.perf_counter_ns()

        def elapsed_ms() -> int:
            return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)

        def record_failure(
            error: BaseException,
            *,
            http_status: int | None = None,
            error_code: str | None = None,
        ) -> None:
            self._rpc_failure_counts[diagnostic_key] = retry_index + 1
            event = {
                "event": "infer_rpc.attempt_failed",
                **base_diagnostic,
                "elapsed_ms": elapsed_ms(),
                "error_chain": _rpc_error_chain(error),
            }
            if http_status is not None:
                event["http_status"] = http_status
            if error_code is not None:
                event["error_code"] = error_code
            _log_rpc_diagnostic(logging.WARNING, event)

        try:
            response = await self._client.request(
                method, self._url(instance_id, path), json=json_body
            )
        except httpx.TransportError as exc:
            record_failure(exc)
            if self._metrics is not None:
                self._metrics.increment(
                    "infer_rpc_ambiguous_total",
                    labels={"reason": type(exc).__name__},
                )
            raise AmbiguousRPCError(endpoint_ref, str(exc)) from exc
        if self._metrics is not None:
            self._metrics.increment(
                "infer_rpc_requests_total",
                labels={"endpoint": metric_endpoint, "status": str(response.status_code)},
            )
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = {}
            code = str(body.get("code", "HTTP_ERROR"))
            if self._metrics is not None and code == "STALE_OPERATION":
                self._metrics.increment(
                    "operation_stale_total",
                    labels={"endpoint": self._metric_endpoint(path), "reason": code},
                )
            error = InferRPCError(
                response.status_code,
                code,
                str(body.get("message", response.text)),
            )
            record_failure(
                error,
                http_status=response.status_code,
                error_code=_safe_rpc_error_code(code),
            )
            raise error
        try:
            value = response.json()
        except ValueError as exc:
            record_failure(exc, http_status=response.status_code)
            raise
        if not isinstance(value, dict):
            error = InferRPCError(
                response.status_code, "INVALID_RESPONSE", "expected object"
            )
            record_failure(
                error,
                http_status=response.status_code,
                error_code="INVALID_RESPONSE",
            )
            raise error
        hook = self._correctness_post_success_hook
        if hook is not None and method == "POST" and endpoint_ref is not None:
            request_bytes = json.dumps(
                json_body, sort_keys=True, separators=(",", ":")
            ).encode()
            response_bytes = json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode()
            hook_details: dict[str, object] = {
                "checkpoint": "after_infer_success_before_control_observe",
                "instance_id": instance_id,
                "path": path,
                "endpoint_ref": asdict(endpoint_ref),
                "request_digest": (
                    "sha256:" + hashlib.sha256(request_bytes).hexdigest()
                ),
                "response_status": response.status_code,
                "response_digest": (
                    "sha256:" + hashlib.sha256(response_bytes).hexdigest()
                ),
            }
            if path == "/v1/cleanup/finalize" and isinstance(json_body, dict):
                hook_details["cleanup_id"] = str(json_body.get("cleanup_id") or "")
                hook_details["cleanup_operation_id"] = str(
                    json_body.get("operation_id") or ""
                )
                hook_details["cleanup_lease_id"] = str(
                    json_body.get("lease_id") or ""
                )
                hook_details["cleanup_payload_digest"] = str(
                    json_body.get("payload_digest") or ""
                )
            discard = await hook(hook_details)
            if discard:
                if self._metrics is not None:
                    self._metrics.increment(
                        "infer_rpc_ambiguous_total",
                        labels={"reason": "CorrectnessResponseDiscard"},
                    )
                error = AmbiguousRPCError(
                    endpoint_ref,
                    "correctness response discarded after remote success",
                )
                record_failure(error, http_status=response.status_code)
                raise error
        prior_failure_count = self._rpc_failure_counts.pop(diagnostic_key, 0)
        if prior_failure_count:
            _log_rpc_diagnostic(logging.INFO, {
                "event": "infer_rpc.retry_recovered",
                **base_diagnostic,
                "elapsed_ms": elapsed_ms(),
                "http_status": response.status_code,
                "prior_failure_count": prior_failure_count,
            })
        return value

    async def get_identity(self, instance_id: str) -> dict[str, object]:
        return await self._request("GET", instance_id, "/v1/identity")

    async def get_capabilities(self, instance_id: str) -> dict[str, object]:
        return await self._request("GET", instance_id, "/v1/capabilities")

    async def get_resources(self, instance_id: str) -> dict[str, object]:
        return await self._request("GET", instance_id, "/v1/resources")

    async def activate_owner(
        self, instance_id: str, owner_generation: str
    ) -> dict[str, object]:
        return await self._request(
            "POST",
            instance_id,
            "/v1/owners/activate",
            json_body={"owner_generation": owner_generation},
        )

    async def owner_status(self, instance_id: str) -> dict[str, object]:
        return await self._request("GET", instance_id, "/v1/owners/status")

    async def retire_owner(
        self, instance_id: str, owner_generation: str
    ) -> dict[str, object]:
        return await self._request(
            "POST", instance_id, f"/v1/owners/{owner_generation}/retire"
        )

    async def list_operations(
        self, instance_id: str, owner_generation: str
    ) -> dict[str, object]:
        return await self._request(
            "GET", instance_id,
            f"/v1/operations?owner_generation={owner_generation}",
        )

    async def request_output(
        self, instance_id: str, req_id: str, after_seq: int = 0
    ) -> dict[str, object]:
        return await self._request(
            "GET", instance_id,
            f"/v1/requests/{req_id}/output?after_seq={after_seq}",
        )

    @staticmethod
    def envelope(
        endpoint_ref: EndpointOperationRef, payload: dict[str, object]
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "endpoint_ref": asdict(endpoint_ref),
            "payload": payload,
        }

    async def _mutate(
        self,
        instance_id: str,
        path: str,
        endpoint_ref: EndpointOperationRef,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return await self._request(
            "POST",
            instance_id,
            path,
            json_body=self.envelope(endpoint_ref, payload),
            endpoint_ref=endpoint_ref,
        )

    async def prepare_request(
        self,
        instance_id: str,
        endpoint_ref: EndpointOperationRef,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return await self._mutate(
            instance_id, "/v1/requests/prepare", endpoint_ref, payload
        )

    async def prepare_receive(
        self,
        instance_id: str,
        endpoint_ref: EndpointOperationRef,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return await self._mutate(
            instance_id, "/v1/transfers/prepare-receive", endpoint_ref, payload
        )

    async def start_transfer(
        self,
        instance_id: str,
        endpoint_ref: EndpointOperationRef,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return await self._mutate(
            instance_id, "/v1/transfers/start", endpoint_ref, payload
        )

    async def prefix_mutation(
        self,
        instance_id: str,
        action: str,
        endpoint_ref: EndpointOperationRef,
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert action in {"resolve", "prepare", "commit"}
        return await self._mutate(
            instance_id, f"/v1/prefix/{action}", endpoint_ref, payload
        )

    async def _abort(
        self,
        instance_id: str,
        path: str,
        endpoint_ref: EndpointOperationRef,
        reason: str,
    ) -> dict[str, object]:
        return await self._request(
            "POST",
            instance_id,
            path,
            json_body={
                "target_operation_ref": asdict(endpoint_ref),
                "reason": reason,
            },
            endpoint_ref=endpoint_ref,
        )

    async def abort_request(
        self,
        instance_id: str,
        endpoint_ref: EndpointOperationRef,
        *,
        reason: str,
    ) -> dict[str, object]:
        return await self._abort(
            instance_id,
            f"/v1/requests/{endpoint_ref.operation_id}/abort",
            endpoint_ref,
            reason,
        )

    async def abort_transfer(
        self,
        instance_id: str,
        endpoint_ref: EndpointOperationRef,
        *,
        reason: str,
    ) -> dict[str, object]:
        return await self._abort(
            instance_id,
            f"/v1/transfers/{endpoint_ref.operation_id}/abort",
            endpoint_ref,
            reason,
        )

    async def operation_status(
        self, instance_id: str, kind: str, operation_id: str
    ) -> dict[str, object]:
        assert kind in {"requests", "transfers"}
        return await self._request("GET", instance_id, f"/v1/{kind}/{operation_id}")

    async def operation_ref_status(
        self, instance_id: str, endpoint_ref: EndpointOperationRef
    ) -> dict[str, object]:
        return await self._request(
            "POST", instance_id, "/v1/operations/status",
            json_body=asdict(endpoint_ref), endpoint_ref=endpoint_ref,
        )

    async def finalize_release(
        self,
        instance_id: str,
        *,
        cleanup_id: str,
        operation_id: str,
        lease_id: str,
        endpoint_refs: tuple[EndpointOperationRef, ...],
        resource_kinds: tuple[str, ...],
    ) -> dict[str, object]:
        payload = finalize_release_payload(
            cleanup_id=cleanup_id,
            operation_id=operation_id,
            lease_id=lease_id,
            endpoint_refs=endpoint_refs,
            resource_kinds=resource_kinds,
        )
        value = await self._request(
            "POST",
            instance_id,
            "/v1/cleanup/finalize",
            json_body=payload,
            endpoint_ref=endpoint_refs[0] if endpoint_refs else None,
        )
        return validate_release_snapshot(
            value,
            instance_id=instance_id,
            cleanup_id=cleanup_id,
            operation_id=operation_id,
            lease_id=lease_id,
            endpoint_refs=endpoint_refs,
            resource_kinds=resource_kinds,
        )

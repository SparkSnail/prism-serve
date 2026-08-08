from __future__ import annotations

import json
import logging

import httpx
import pytest

from prism_serve.router.http_rpc import (
    AmbiguousRPCError,
    EndpointSequenceAllocator,
    HttpInferClient,
    InferRPCError,
)


def _ref(payload=None):
    return EndpointSequenceAllocator(
        "world-a", "gateway-a:boot-a"
    ).allocate(
        target_instance="d0",
        target_worker_epoch="pod-d0:boot-a",
        operation_id="op-1",
        payload={} if payload is None else payload,
    )


@pytest.mark.asyncio
async def test_http_timeout_is_ambiguous_not_non_execution():
    async def timeout(request):
        raise httpx.ReadTimeout("response lost", request=request)

    client = HttpInferClient(
        {"d0": "http://d0"}, transport=httpx.MockTransport(timeout)
    )
    try:
        with pytest.raises(AmbiguousRPCError) as error:
            await client.prepare_request("d0", _ref(), {})
        assert error.value.endpoint_ref == _ref()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_retry_diagnostics_are_structured_and_secret_safe(caplog):
    secret = "rpc-secret-value"
    attempts = 0

    async def flaky(request):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            try:
                raise OSError(111, "connection refused")
            except OSError as cause:
                raise httpx.ConnectError(
                    "connect failed "
                    f"url=http://user:{secret}@d0:8001?token={secret}",
                    request=request,
                ) from cause
        return httpx.Response(
            200,
            json={"active_owner": "owner-a"},
            request=request,
        )

    client = HttpInferClient(
        {"d0": f"http://user:{secret}@d0:8001"},
        transport=httpx.MockTransport(flaky),
    )
    try:
        with caplog.at_level(
            logging.INFO, logger="prism_serve.router.http_rpc"
        ):
            for _ in range(2):
                with pytest.raises(AmbiguousRPCError):
                    await client.owner_status("d0")
            assert await client.owner_status("d0") == {
                "active_owner": "owner-a"
            }
    finally:
        await client.close()

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    ]
    failed = [
        value for value in events
        if value.get("event") == "infer_rpc.attempt_failed"
    ]
    recovered = [
        value for value in events
        if value.get("event") == "infer_rpc.retry_recovered"
    ]
    assert [value["retry_index"] for value in failed] == [0, 1]
    assert len(recovered) == 1
    assert recovered[0]["retry_index"] == 2
    assert recovered[0]["prior_failure_count"] == 2
    for value in [*failed, *recovered]:
        assert value["instance_id"] == "d0"
        assert value["method"] == "GET"
        assert value["path"] == "/v1/owners/status"
        assert value["endpoint"] == "http://d0:8001"
        assert type(value["elapsed_ms"]) is int
        assert value["elapsed_ms"] >= 0
    assert failed[0]["error_chain"][0]["type"] == "ConnectError"
    assert any(
        value.get("errno") == 111
        for value in failed[0]["error_chain"]
    )
    assert secret not in json.dumps(events, sort_keys=True)


@pytest.mark.asyncio
async def test_http_503_diagnostic_preserves_infer_rpc_error(caplog):
    attempts = 0

    async def not_ready_then_ready(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                503,
                json={"code": "NOT_READY", "message": "warming"},
                request=request,
            )
        return httpx.Response(
            200,
            json={"active_owner": None},
            request=request,
        )

    client = HttpInferClient(
        {"d0": "http://d0:8001"},
        transport=httpx.MockTransport(not_ready_then_ready),
    )
    try:
        with caplog.at_level(
            logging.INFO, logger="prism_serve.router.http_rpc"
        ):
            with pytest.raises(InferRPCError) as error:
                await client.owner_status("d0")
            assert error.value.status_code == 503
            assert error.value.code == "NOT_READY"
            assert await client.owner_status("d0") == {"active_owner": None}
    finally:
        await client.close()

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    ]
    failed = next(
        value for value in events
        if value.get("event") == "infer_rpc.attempt_failed"
    )
    assert failed["retry_index"] == 0
    assert failed["http_status"] == 503
    assert failed["error_code"] == "NOT_READY"
    assert failed["error_chain"] == [{
        "module": "prism_serve.router.http_rpc",
        "type": "InferRPCError",
    }]
    recovered = next(
        value for value in events
        if value.get("event") == "infer_rpc.retry_recovered"
    )
    assert recovered["retry_index"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_code",
    [
        "SECRET_TOKEN_7",
        "NOT_READY|SECRET_TOKEN_7",
        "A" * 52 + "SECRET_TOKEN_7",
    ],
)
async def test_http_error_code_diagnostic_redacts_untrusted_body_code(
    caplog,
    unsafe_code,
):
    async def untrusted_error(request):
        return httpx.Response(
            503,
            json={"code": unsafe_code, "message": "not ready"},
            request=request,
        )

    client = HttpInferClient(
        {"d0": "http://d0:8001"},
        transport=httpx.MockTransport(untrusted_error),
    )
    try:
        with caplog.at_level(
            logging.WARNING, logger="prism_serve.router.http_rpc"
        ):
            with pytest.raises(InferRPCError) as error:
                await client.owner_status("d0")
    finally:
        await client.close()

    assert error.value.code == unsafe_code
    event = next(
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    )
    assert event["error_code"] == "<invalid-error-code>"
    assert "SECRET_TOKEN_7" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_error_type",
    [httpx.ReadError, httpx.RemoteProtocolError],
)
async def test_post_send_transport_error_preserves_exact_mutation_ref(
    transport_error_type,
):
    calls = []

    async def response_lost(request):
        calls.append(request)
        raise transport_error_type("response lost after send", request=request)

    client = HttpInferClient(
        {"d0": "http://d0"}, transport=httpx.MockTransport(response_lost)
    )
    ref = _ref()
    try:
        with pytest.raises(AmbiguousRPCError) as error:
            await client.prepare_request("d0", ref, {})

        assert error.value.endpoint_ref is ref
        assert len(calls) == 1
        assert calls[0].url.path == "/v1/requests/prepare"
    finally:
        await client.close()


def test_cross_endpoint_refs_use_independent_target_sequences():
    allocator = EndpointSequenceAllocator("world-a", "gateway-a:boot-a")
    source = allocator.allocate(
        target_instance="d0",
        target_worker_epoch="pod-d0:boot-a",
        operation_id="shared-op",
        payload={"side": "source"},
    )
    target = allocator.allocate(
        target_instance="d1",
        target_worker_epoch="pod-d1:boot-a",
        operation_id="shared-op",
        payload={"side": "target"},
    )
    next_source = allocator.allocate(
        target_instance="d0",
        target_worker_epoch="pod-d0:boot-a",
        operation_id="another-op",
        payload={"side": "source"},
    )

    assert source.operation_seq == target.operation_seq == 1
    assert next_source.operation_seq == 2
    assert source.payload_digest != target.payload_digest

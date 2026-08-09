"""Exercise owner-scoped delivery and reconnect behavior against a local NATS process."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import queue as stdlib_queue
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from prism_serve.scheduler.queue import NATSQueue


class LocalNATSServer:
    def __init__(self, binary: str, port: int, log_path: Path) -> None:
        self.binary = binary
        self.port = port
        self.log_path = log_path
        self.process: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"nats://127.0.0.1:{self.port}"

    def start(self) -> None:
        log = self.log_path.open("ab")
        self.process = subprocess.Popen(
            [self.binary, "--addr", "127.0.0.1", "--port", str(self.port)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"nats-server exited early; inspect {self.log_path}"
                )
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.02)
        raise TimeoutError("nats-server did not accept connections")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)
        self.process = None


@pytest.fixture
def local_nats_server(tmp_path):
    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("nats-server binary is not installed")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = LocalNATSServer(binary, port, tmp_path / "nats-server.log")
    server.start()
    try:
        yield server
    finally:
        server.stop()


async def _wait_until(predicate, timeout_s: float = 8.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("condition did not become true")


async def _poll_until(queue: NATSQueue, subject: str) -> list[dict]:
    result: list[dict] = []

    async def received() -> bool:
        nonlocal result
        result = await queue.poll(subject)
        return bool(result)

    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        if await received():
            return result
        await asyncio.sleep(0.02)
    raise TimeoutError(f"no message arrived for {subject}")


def _run_stub_worker(
    nats_url: str,
    dispatch_subject: str,
    results: multiprocessing.Queue,
) -> None:

    async def run() -> None:
        import nats

        client = await nats.connect(nats_url, connect_timeout=2.0)
        completed = asyncio.Event()

        async def handle_dispatch(message) -> None:
            command = json.loads(message.data)
            results.put({"kind": "dispatch", "command": command})
            await client.publish(
                command["prefill_done_subject"],
                json.dumps({
                    "req_id": command["req_id"],
                    "instance_epoch": command["instance_epoch"],
                    "kv_size_bytes": 1,
                    "block_table": [7],
                }).encode(),
            )
            await client.flush()


            await asyncio.sleep(0.1)
            await client.publish(
                command["first_token_subject"],
                json.dumps({
                    "req_id": command["req_id"],
                    "instance_epoch": command["decode_instance_epoch"],
                    "token_ids": [101],
                    "output_seq_no": 1,
                }).encode(),
            )
            await client.publish(
                command["decode_done_subject"],
                json.dumps({
                    "req_id": command["req_id"],
                    "instance_epoch": command["decode_instance_epoch"],
                    "token_ids": [101, 102],
                    "output_seq_no": 2,
                }).encode(),
            )
            await client.flush()
            completed.set()

        await client.subscribe(dispatch_subject, cb=handle_dispatch)
        await client.flush()
        results.put({"kind": "ready"})
        try:
            await asyncio.wait_for(completed.wait(), timeout=15.0)
        finally:
            await client.drain()
            await client.close()

    try:
        asyncio.run(run())
    except BaseException as exc:
        results.put({"kind": "error", "error": repr(exc)})
        raise


def _run_gateway(gateway_port: int, nats_url: str) -> None:
    os.environ.update({
        "PRISM_SERVE_HOST": "127.0.0.1",
        "PRISM_SERVE_PORT": str(gateway_port),
        "PRISM_SERVE_LOG_LEVEL": "warning",
        "PRISM_SERVE_NATS_URL": nats_url,
        "PRISM_SERVE_NATS_REQUIRED": "true",
        "PRISM_SERVE_GATEWAY_POD_UID": "gateway-a",
        "PRISM_SERVE_GATEWAY_PROCESS_GENERATION": "boot-1",
        "PRISM_SERVE_MODEL_ID": "local-control-plane",
        "PRISM_SERVE_SCHEDULE_LOOP_TICK_MS": "5",
        "PRISM_SERVE_GOVERNOR_TICK_S": "0.05",
    })
    import uvicorn

    from prism_serve.gateway.app import app, _make_stub_infer_client

    infer_client = _make_stub_infer_client()

    async def get_kv_usage_all() -> dict:
        return {
            "d-0": {"ratio": 0.0, "instance_epoch": "d-epoch-1"},
        }

    infer_client.get_kv_usage_all = get_kv_usage_all
    app.state.infer_client = infer_client
    uvicorn.run(app, host="127.0.0.1", port=gateway_port, log_level="warning")


def _wait_for_gateway(client: httpx.Client, process) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError("gateway exited before readiness")
        try:
            if client.get("/readyz").status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.05)
    raise TimeoutError("gateway did not become ready")


@pytest.mark.asyncio
async def test_real_nats_owner_delivery_and_reconnect(local_nats_server) -> None:
    config = {
        "nats_url": local_nats_server.url,
        "nats_connect_timeout_s": 1.0,
        "nats_max_reconnect_attempts": 30,
        "scheduler_id": "gateway-a",
        "scheduler_generation": "boot-1",
    }
    owner = NATSQueue(config)
    other = NATSQueue({
        **config,
        "scheduler_id": "gateway-b",
    })
    await owner.connect()
    await other.connect()
    try:
        payload = {"req_id": "r-before", "instance_epoch": "d0:boot-1"}
        await owner.publish(owner.reply_subject("suffix_prefill_done"), payload)
        assert await _poll_until(owner, "suffix_prefill_done") == [payload]
        assert await other.poll("suffix_prefill_done") == []

        local_nats_server.stop()
        await _wait_until(lambda: not owner.is_connected)
        local_nats_server.start()
        await _wait_until(lambda: owner.is_connected and other.is_connected)

        after = {"req_id": "r-after", "instance_epoch": "d0:boot-1"}
        await owner.publish(owner.reply_subject("suffix_prefill_done"), after)
        assert await _poll_until(owner, "suffix_prefill_done") == [after]
        assert await other.poll("suffix_prefill_done") == []


        await owner._nc.publish(owner.reply_subject("suffix_prefill_done"), b"{")
        await owner._nc.flush()
        await asyncio.sleep(0.05)
        assert await owner.poll("suffix_prefill_done") == []
    finally:
        await owner.close()
        await other.close()


def test_real_gateway_and_stub_worker_cross_process_nats_http(
    local_nats_server,
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        gateway_port = listener.getsockname()[1]

    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    worker = context.Process(
        target=_run_stub_worker,
        args=(local_nats_server.url, "dispatch_prefill.p-0", results),
    )
    worker.start()
    assert results.get(timeout=10.0) == {"kind": "ready"}

    gateway = context.Process(
        target=_run_gateway,
        args=(gateway_port, local_nats_server.url),
    )
    gateway.start()
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{gateway_port}", timeout=15.0
    )
    try:
        _wait_for_gateway(client, gateway)
        for registration in (
            {
                "instance_id": "p-0",
                "instance_epoch": "p-epoch-1",
                "role": "prefill",
                "active_request_ids": [],
            },
            {
                "instance_id": "d-0",
                "instance_epoch": "d-epoch-1",
                "role": "decode",
                "max_slots": 1,
                "active_request_ids": [],
            },
        ):
            response = client.post("/internal/register_instance", json=registration)
            assert response.status_code == 200, response.text

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-control-plane",
                "messages": [{"role": "user", "content": "hello"}],
                "input_token_ids": [1, 2, 3],
                "request_id": "real-local-request",
                "max_tokens": 2,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["token_ids"] == [101, 102]

        dispatch = results.get(timeout=10.0)
        assert dispatch["kind"] == "dispatch"
        assert dispatch["command"]["req_id"] == "real-local-request"
        assert dispatch["command"]["owner_id"] == "gateway-a--boot-1"
        worker.join(timeout=10.0)
        assert worker.exitcode == 0
        with pytest.raises(stdlib_queue.Empty):
            results.get_nowait()
    finally:
        client.close()
        if gateway.is_alive():
            gateway.terminate()
            gateway.join(timeout=10.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10.0)

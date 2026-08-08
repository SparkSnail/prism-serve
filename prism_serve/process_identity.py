"""Identity-bound process publication and fault injection helpers."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
from typing import Callable


SCHEMA_VERSION = 1
_PIDFD_OPEN_SYSCALL = 434
_PIDFD_SEND_SIGNAL_SYSCALL = 424


def _linux_syscall(number: int, *args: object) -> int:
    """Invoke one Linux syscall and preserve errno as an OSError."""
    if (
        os.name != "posix"
        or not hasattr(os, "uname")
        or os.uname().machine.lower() not in {"x86_64", "amd64"}
    ):
        raise RuntimeError("pidfd syscall fallback requires Linux amd64")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = syscall(ctypes.c_long(number), *args)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def _pidfd_open(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid, 0)
    return _linux_syscall(
        _PIDFD_OPEN_SYSCALL, ctypes.c_int(pid), ctypes.c_uint(0)
    )


def _pidfd_send_signal(pidfd: int, sig: int | signal.Signals) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(pidfd, sig)
        return
    _linux_syscall(
        _PIDFD_SEND_SIGNAL_SYSCALL,
        ctypes.c_int(pidfd),
        ctypes.c_int(int(sig)),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )


def assert_pidfd_support() -> None:
    """Prove the current image can open and address an exact process pidfd."""
    pidfd = _pidfd_open(os.getpid())
    try:
        _pidfd_send_signal(pidfd, 0)
    finally:
        os.close(pidfd)


def process_start_ticks(pid: int) -> int:
    """Return Linux /proc starttime (field 22) for one exact process."""
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    try:
        tail = raw.rsplit(")", 1)[1].split()
        value = int(tail[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"invalid /proc/{pid}/stat") from exc
    if value <= 0:
        raise RuntimeError("process start ticks must be positive")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def publish_process_identity(
    path: str | Path,
    *,
    component: str,
    instance_id: str,
    pod_uid: str,
    process_generation: str,
) -> dict[str, object]:
    """Atomically publish the app PID identity used only to select a fault target."""
    pid = os.getpid()
    if pid <= 1:
        raise RuntimeError(
            "fault profile requires a shared PID namespace with app_pid > 1"
        )
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "component": component,
        "instance_id": instance_id,
        "pod_uid": pod_uid,
        "process_generation": process_generation,
        "instance_epoch": f"{pod_uid}:{process_generation}",
        "app_pid": pid,
        "process_start_ticks": process_start_ticks(pid),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=destination.name + ".",
            suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(_canonical_bytes(identity))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o444)
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return identity


def signal_exact_process(
    path: str | Path,
    *,
    component: str,
    instance_id: str,
    pod_uid: str,
    process_generation: str,
    before_signal: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Validate one pidfile and SIGKILL its exact pidfd without PID-reuse races."""
    identity = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise RuntimeError("process identity pidfile must contain a JSON object")
    expected = {
        "component": component,
        "instance_id": instance_id,
        "pod_uid": pod_uid,
        "process_generation": process_generation,
        "instance_epoch": f"{pod_uid}:{process_generation}",
    }
    pid = identity.get("app_pid")
    start_ticks = identity.get("process_start_ticks")
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("fault target app_pid must be greater than 1")
    if type(start_ticks) is not int or start_ticks <= 0:
        raise RuntimeError("fault target process_start_ticks must be positive")
    pidfd = _pidfd_open(pid)
    try:
        current_identity = json.loads(Path(path).read_text(encoding="utf-8"))
        if current_identity != identity:
            raise RuntimeError("process identity changed after pidfd open")
        if identity.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("process identity schema mismatch")
        for field, value in expected.items():
            if identity.get(field) != value:
                raise RuntimeError(f"process identity mismatch: {field}")
        if process_start_ticks(pid) != start_ticks:
            raise RuntimeError("fault target start ticks changed after pidfd open")
        result = {
            **identity,
            "signal": int(signal.SIGKILL),
            "pidfd": True,
            "identity_sha256": "sha256:"
            + hashlib.sha256(_canonical_bytes(identity)).hexdigest(),
        }
        if before_signal is not None:
            before_signal(result)
        _pidfd_send_signal(pidfd, signal.SIGKILL)
    finally:
        os.close(pidfd)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("kill",))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--expected-component", required=True)
    parser.add_argument("--expected-instance-id", required=True)
    parser.add_argument("--expected-pod-uid", required=True)
    parser.add_argument("--expected-process-generation", required=True)
    args = parser.parse_args()
    signal_exact_process(
        args.path,
        component=args.expected_component,
        instance_id=args.expected_instance_id,
        pod_uid=args.expected_pod_uid,
        process_generation=args.expected_process_generation,
        before_signal=lambda value: print(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            flush=True,
        ),
    )


if __name__ == "__main__":
    main()

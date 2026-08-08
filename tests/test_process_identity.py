from __future__ import annotations

import json

import pytest

from prism_serve import process_identity


def _identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "component": "gateway",
        "instance_id": "gateway",
        "pod_uid": "pod-a",
        "process_generation": "boot-a",
        "instance_epoch": "pod-a:boot-a",
        "app_pid": 8,
        "process_start_ticks": 12345,
    }


def test_publish_process_identity_is_atomic_and_binds_proc_start(
    tmp_path, monkeypatch,
):
    path = tmp_path / "identity.json"
    monkeypatch.setattr(process_identity.os, "getpid", lambda: 8)
    monkeypatch.setattr(process_identity, "process_start_ticks", lambda pid: 12345)

    published = process_identity.publish_process_identity(
        path,
        component="gateway",
        instance_id="gateway",
        pod_uid="pod-a",
        process_generation="boot-a",
    )

    assert published == _identity()
    assert json.loads(path.read_text(encoding="utf-8")) == _identity()
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_process_identity_rejects_namespace_init(tmp_path, monkeypatch):
    monkeypatch.setattr(process_identity.os, "getpid", lambda: 1)

    with pytest.raises(RuntimeError, match="app_pid > 1"):
        process_identity.publish_process_identity(
            tmp_path / "identity.json",
            component="gateway",
            instance_id="gateway",
            pod_uid="pod-a",
            process_generation="boot-a",
        )


def test_pidfd_syscall_fallback_when_python_omits_wrappers(monkeypatch):
    calls: list[object] = []
    monkeypatch.delattr(process_identity.os, "pidfd_open", raising=False)
    monkeypatch.delattr(
        process_identity.signal, "pidfd_send_signal", raising=False
    )
    monkeypatch.setattr(process_identity.os, "getpid", lambda: 21)
    monkeypatch.setattr(
        process_identity.os, "close", lambda fd: calls.append(("close", fd))
    )

    def syscall(number, *args):
        values = tuple(getattr(argument, "value", argument) for argument in args)
        calls.append(("syscall", number, values))
        return 17 if number == process_identity._PIDFD_OPEN_SYSCALL else 0

    monkeypatch.setattr(process_identity, "_linux_syscall", syscall)

    process_identity.assert_pidfd_support()

    assert calls == [
        ("syscall", process_identity._PIDFD_OPEN_SYSCALL, (21, 0)),
        (
            "syscall",
            process_identity._PIDFD_SEND_SIGNAL_SYSCALL,
            (17, 0, None, 0),
        ),
        ("close", 17),
    ]


def test_signal_exact_process_opens_pidfd_before_identity_checks(
    tmp_path, monkeypatch,
):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(_identity()), encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(
        process_identity.os, "pidfd_open",
        lambda pid, flags: calls.append(("open", pid, flags)) or 17,
        raising=False,
    )
    monkeypatch.setattr(
        process_identity, "process_start_ticks",
        lambda pid: calls.append(("ticks", pid)) or 12345,
    )
    monkeypatch.setattr(
        process_identity.signal, "pidfd_send_signal",
        lambda fd, sig: calls.append(("signal", fd, int(sig))),
        raising=False,
    )
    monkeypatch.setattr(
        process_identity.os, "close", lambda fd: calls.append(("close", fd))
    )

    result = process_identity.signal_exact_process(
        path,
        component="gateway",
        instance_id="gateway",
        pod_uid="pod-a",
        process_generation="boot-a",
        before_signal=lambda value: calls.append(
            ("evidence", value["identity_sha256"])
        ),
    )

    assert result["app_pid"] == 8
    assert calls == [
        ("open", 8, 0),
        ("ticks", 8),
        ("evidence", result["identity_sha256"]),
        ("signal", 17, 9),
        ("close", 17),
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pod_uid", "pod-b", "pod_uid"),
        ("process_generation", "boot-b", "process_generation"),
        ("instance_id", "worker", "instance_id"),
        ("process_start_ticks", 999, "start ticks"),
    ],
)
def test_signal_exact_process_mismatch_is_zero_signal(
    tmp_path, monkeypatch, field, value, message,
):
    identity = _identity()
    identity[field] = value
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(identity), encoding="utf-8")
    signalled: list[object] = []
    monkeypatch.setattr(
        process_identity.os, "pidfd_open", lambda pid, flags: 17,
        raising=False,
    )
    monkeypatch.setattr(process_identity.os, "close", lambda fd: None)
    monkeypatch.setattr(process_identity, "process_start_ticks", lambda pid: 12345)
    monkeypatch.setattr(
        process_identity.signal, "pidfd_send_signal",
        lambda fd, sig: signalled.append((fd, sig)),
        raising=False,
    )

    with pytest.raises(RuntimeError, match=message):
        process_identity.signal_exact_process(
            path,
            component="gateway",
            instance_id="gateway",
            pod_uid="pod-a",
            process_generation="boot-a",
        )

    assert signalled == []


def test_signal_exact_process_rejects_pidfile_replacement_after_pidfd_open(
    tmp_path, monkeypatch,
):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(_identity()), encoding="utf-8")
    signalled: list[object] = []

    def open_and_replace(pid, flags):
        replacement = _identity()
        replacement["process_generation"] = "boot-replacement"
        path.write_text(json.dumps(replacement), encoding="utf-8")
        return 17

    monkeypatch.setattr(
        process_identity.os, "pidfd_open", open_and_replace, raising=False,
    )
    monkeypatch.setattr(process_identity.os, "close", lambda fd: None)
    monkeypatch.setattr(
        process_identity.signal, "pidfd_send_signal",
        lambda fd, sig: signalled.append((fd, sig)),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="changed after pidfd open"):
        process_identity.signal_exact_process(
            path,
            component="gateway",
            instance_id="gateway",
            pod_uid="pod-a",
            process_generation="boot-a",
        )

    assert signalled == []

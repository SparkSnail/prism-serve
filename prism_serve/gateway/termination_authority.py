"""Validate Kubernetes process-termination evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
import re


TERMINATION_RECORD_FIELDS = frozenset({
    "logical_instance_id",
    "topology_generation",
    "pod_uid",
    "node_uid",
    "container_name",
    "captured_container_id",
    "process_generation",
    "watch_start_resource_version",
    "observed_resource_version",
    "deletion_resource_version",
    "restart_count_before",
    "restart_count_observed",
    "termination_source",
    "termination_event_type",
    "deletion_event_type",
    "terminated",
    "adjacent_current_container_id",
    "pod_deletion_observed",
    "raw_pod_json_sha256",
    "termination_raw_observation_sequence",
    "deletion_raw_pod_json_sha256",
    "deletion_raw_observation_sequence",
    "observation_sha256",
})
TERMINATED_FIELDS = frozenset({
    "exit_code",
    "reason",
    "signal",
    "started_at",
    "finished_at",
})
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def termination_observation_digest(record: Mapping[str, object]) -> str:

    if set(record) != TERMINATION_RECORD_FIELDS:
        raise ValueError("termination record fields are not exact")
    canonical = {
        key: value
        for key, value in record.items()
        if key != "observation_sha256"
    }
    return "sha256:" + hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _text(
    record: Mapping[str, object],
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"termination {field} must be a string")
    return value


def _integer(
    record: Mapping[str, object],
    field: str,
    *,
    minimum: int = 0,
) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"termination {field} must be an integer")
    return value


def validate_termination_record(
    value: Mapping[str, object],
    *,
    expected_generation: str | None = None,
) -> dict[str, object]:

    if not isinstance(value, Mapping):
        raise ValueError("termination record must be an object")
    record = dict(value)
    if set(record) != TERMINATION_RECORD_FIELDS:
        raise ValueError("termination record fields are not exact")
    expected_digest = termination_observation_digest(record)
    if record.get("observation_sha256") != expected_digest:
        raise ValueError("termination observation_sha256 mismatch")

    for field in (
        "logical_instance_id",
        "topology_generation",
        "pod_uid",
        "node_uid",
        "container_name",
        "captured_container_id",
        "process_generation",
        "watch_start_resource_version",
        "observed_resource_version",
        "deletion_resource_version",
        "termination_source",
        "termination_event_type",
        "deletion_event_type",
        "raw_pod_json_sha256",
        "deletion_raw_pod_json_sha256",
    ):
        _text(record, field)
    if record["container_name"] != "worker":
        raise ValueError("termination container_name must be worker")
    if (
        expected_generation is not None
        and record["topology_generation"] != expected_generation
    ):
        raise ValueError("termination topology generation mismatch")
    if record["termination_event_type"] != "MODIFIED":
        raise ValueError(
            "termination transition must begin with MODIFIED"
        )
    if record["deletion_event_type"] != "DELETED":
        raise ValueError("termination transition must end with DELETED")
    if record.get("pod_deletion_observed") is not True:
        raise ValueError("termination Pod deletion is not proven")

    before = _integer(record, "restart_count_before")
    observed = _integer(record, "restart_count_observed")
    termination_sequence = _integer(
        record,
        "termination_raw_observation_sequence",
        minimum=1,
    )
    deletion_sequence = _integer(
        record,
        "deletion_raw_observation_sequence",
        minimum=1,
    )
    if deletion_sequence <= termination_sequence:
        raise ValueError(
            "termination raw transitions must link MODIFIED before DELETED"
        )
    if not _SHA256.fullmatch(str(record["raw_pod_json_sha256"])):
        raise ValueError("termination raw Pod digest is invalid")
    if not _SHA256.fullmatch(
        str(record["deletion_raw_pod_json_sha256"])
    ):
        raise ValueError("termination deletion raw Pod digest is invalid")



    watch_resource_version = str(record["watch_start_resource_version"])
    if watch_resource_version in {
        str(record["observed_resource_version"]),
        str(record["deletion_resource_version"]),
    }:
        raise ValueError(
            "termination resourceVersion must advance beyond watch start"
        )

    terminated = record.get("terminated")
    if not isinstance(terminated, Mapping):
        raise ValueError("termination terminated must be an object")
    terminated = dict(terminated)
    if set(terminated) != TERMINATED_FIELDS:
        raise ValueError("termination terminated fields are not exact")
    _integer(terminated, "exit_code")
    _text(terminated, "reason", allow_empty=True)
    _text(terminated, "started_at")
    _text(terminated, "finished_at")
    signal = terminated.get("signal")
    if (
        signal is not None
        and (
            isinstance(signal, bool)
            or not isinstance(signal, int)
            or signal < 0
        )
    ):
        raise ValueError("termination signal must be null or integer")

    source = record["termination_source"]
    adjacent = record.get("adjacent_current_container_id")
    captured_container = str(record["captured_container_id"])
    if source == "state.terminated":
        if observed != before:
            raise ValueError(
                "state.terminated restartCount must equal captured count"
            )
        if adjacent is not None:
            raise ValueError(
                "state.terminated cannot have an adjacent current container"
            )
    elif source == "lastState.terminated":
        if observed != before + 1:
            raise ValueError(
                "lastState.terminated restartCount must increase exactly once"
            )
        if (
            not isinstance(adjacent, str)
            or not adjacent
            or adjacent == captured_container
        ):
            raise ValueError(
                "lastState.terminated requires a distinct adjacent current "
                "container"
            )
    else:
        raise ValueError("termination source is invalid")
    return record


def seal_termination_record(
    payload: Mapping[str, object],
) -> dict[str, object]:

    if set(payload) != TERMINATION_RECORD_FIELDS - {"observation_sha256"}:
        raise ValueError("termination payload fields are not exact")
    record = {
        **dict(payload),
        "observation_sha256": "",
    }
    record["observation_sha256"] = termination_observation_digest(record)
    return validate_termination_record(record)


def validate_termination_records(
    values: Iterable[Mapping[str, object]],
    *,
    expected_generation: str,
    expected_members: Iterable[str],
) -> list[dict[str, object]]:

    records = [
        validate_termination_record(
            value,
            expected_generation=expected_generation,
        )
        for value in values
    ]
    members = {
        str(value["logical_instance_id"]) for value in records
    }
    if len(records) != 4 or members != set(expected_members):
        raise ValueError("restart requires four member termination records")
    pod_uids = {str(value["pod_uid"]) for value in records}
    container_ids = {
        str(value["captured_container_id"]) for value in records
    }
    if len(pod_uids) != 4 or len(container_ids) != 4:
        raise ValueError("termination records must bind four distinct processes")
    raw_sequences = [
        int(value[field])
        for value in records
        for field in (
            "termination_raw_observation_sequence",
            "deletion_raw_observation_sequence",
        )
    ]
    if len(set(raw_sequences)) != len(raw_sequences):
        raise ValueError(
            "termination raw transition sequences must be globally unique"
        )
    return records

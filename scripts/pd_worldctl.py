"""Apply one fixed 2P2D topology replacement and record its evidence."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
from dataclasses import dataclass, replace
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import uuid

from prism_serve.gateway.termination_authority import (
    seal_termination_record,
    validate_termination_records,
)


MEMBERS = ("p0", "p1", "d0", "d1")
WORKER_PROCESS_IDENTITY_PATH = "/tmp/prism-worker-process-identity.json"
OBSERVATION_SCHEMA = "prism.pd_worldctl.observation/v1"
STARTUP_PERMIT_SCHEMA = "prism.week12.worker-startup-permit/v1"
STARTUP_PERMIT_FIELDS = frozenset({
    "schema_version",
    "issuance_mode",
    "permit_id",
    "topology_generation",
    "members",
    "canonical_digest",
})
INITIALIZATION_STATE_KEY = "initialize-state.json"
STARTUP_PERMIT_KEY = "startup-permit.json"


def _canonical_uuid(value: str) -> str:

    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "topology generation must be a canonical UUID v1-v5"
        ) from exc
    if str(parsed) != value or parsed.version not in {1, 2, 3, 4, 5}:
        raise argparse.ArgumentTypeError(
            "topology generation must be a canonical UUID v1-v5"
        )
    return value


@dataclass(slots=True, frozen=True)
class TerminationProof:
    member: str
    pod_uid: str
    node_uid: str
    container_id: str
    process_generation: str
    exit_code: int
    container_terminated: bool
    pod_deleted: bool
    termination_reason: str = ""
    termination_message: str = ""
    finished_at: str = ""
    topology_generation: str = ""
    container_name: str = "worker"
    watch_start_resource_version: str = ""
    observed_resource_version: str = ""
    deletion_resource_version: str = ""
    restart_count_before: int = 0
    restart_count_observed: int = 0
    termination_source: str = ""
    termination_event_type: str = ""
    deletion_event_type: str = ""
    signal: int | None = None
    started_at: str = ""
    adjacent_current_container_id: str | None = None
    termination_raw_pod_json_sha256: str = ""
    termination_raw_observation_sequence: int | None = None
    deletion_raw_pod_json_sha256: str = ""
    deletion_raw_observation_sequence: int | None = None


@dataclass(slots=True, frozen=True)
class TerminationObservation:

    proof: TerminationProof
    raw_watch_event: dict[str, object] | None = None
    raw_watch_event_text: str = ""
    raw_watch_event_sha256: str = ""
    resource_version: str = ""
    raw_observation_sequence: int | None = None


@dataclass(slots=True, frozen=True)
class ProcessIdentity:
    member: str
    pod_uid: str
    node_uid: str
    container_id: str
    process_generation: str
    pod_name: str = ""
    node_name: str = ""
    resource_version: str = ""
    topology_generation: str = ""
    container_name: str = "worker"
    restart_count: int = 0


@dataclass(slots=True, frozen=True)
class InitializationSnapshot:

    configmap_uid: str
    resource_version: str
    state_raw: str
    state: dict[str, object]
    startup_permit_raw: str | None
    startup_permit: dict[str, object] | None


@dataclass(slots=True, frozen=True)
class _HelmReleaseSnapshot:

    revision: int
    values: dict[str, object]


@dataclass(slots=True)
class PodTerminationWatch:
    """A watch fixed to one captured Pod UID and resource version."""

    expected: ProcessIdentity
    process: subprocess.Popen[str]
    events: Queue[str | KubernetesWatchEvent | BaseException]
    reader: Thread


@dataclass(slots=True, frozen=True)
class KubernetesWatchEvent:

    raw_text: str
    raw_sha256: str


@dataclass(slots=True, frozen=True)
class DecodedKubernetesWatchEvent:
    normalized: str
    raw_object: dict[str, object]
    raw_text: str
    raw_sha256: str
    resource_version: str
    raw_observation_sequence: int | None = None


class TerminationObservationError(RuntimeError):

    def __init__(
        self,
        message: str,
        *,
        raw_watch_event: object | None,
        raw_watch_event_text: str,
        raw_watch_event_sha256: str,
        resource_version: str,
        raw_observation_sequence: int | None = None,
        partial_observation: TerminationObservation | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_watch_event = raw_watch_event
        self.raw_watch_event_text = raw_watch_event_text
        self.raw_watch_event_sha256 = raw_watch_event_sha256
        self.resource_version = resource_version
        self.raw_observation_sequence = raw_observation_sequence
        self.partial_observation = partial_observation


def _as_termination_observation(
    value: TerminationProof | TerminationObservation,
) -> TerminationObservation:
    if isinstance(value, TerminationObservation):
        return value
    if isinstance(value, TerminationProof):
        return TerminationObservation(proof=value)
    raise RuntimeError("backend returned an invalid termination observation")


def _termination_proof_document(
    proof: TerminationProof,
) -> dict[str, object]:

    return {
        "member": proof.member,
        "pod_uid": proof.pod_uid,
        "node_uid": proof.node_uid,
        "container_id": proof.container_id,
        "process_generation": proof.process_generation,
        "exit_code": proof.exit_code,
        "container_terminated": proof.container_terminated,
        "pod_deleted": proof.pod_deleted,
        "termination_reason": proof.termination_reason,
        "termination_message": proof.termination_message,
        "finished_at": proof.finished_at,
        "topology_generation": proof.topology_generation,
        "container_name": proof.container_name,
        "watch_start_resource_version":
            proof.watch_start_resource_version,
        "observed_resource_version": proof.observed_resource_version,
        "deletion_resource_version": proof.deletion_resource_version,
        "restart_count_before": proof.restart_count_before,
        "restart_count_observed": proof.restart_count_observed,
        "termination_source": proof.termination_source,
        "termination_event_type": proof.termination_event_type,
        "deletion_event_type": proof.deletion_event_type,
        "signal": proof.signal,
        "started_at": proof.started_at,
        "adjacent_current_container_id":
            proof.adjacent_current_container_id,
        "termination_raw_pod_json_sha256":
            proof.termination_raw_pod_json_sha256,
        "termination_raw_observation_sequence":
            proof.termination_raw_observation_sequence,
        "deletion_raw_pod_json_sha256":
            proof.deletion_raw_pod_json_sha256,
        "deletion_raw_observation_sequence":
            proof.deletion_raw_observation_sequence,
    }


def _termination_authority_payload(
    proof: TerminationProof,
) -> dict[str, object]:

    return {
        "logical_instance_id": proof.member,
        "topology_generation": proof.topology_generation,
        "pod_uid": proof.pod_uid,
        "node_uid": proof.node_uid,
        "container_name": proof.container_name,
        "captured_container_id": proof.container_id,
        "process_generation": proof.process_generation,
        "watch_start_resource_version":
            proof.watch_start_resource_version,
        "observed_resource_version": proof.observed_resource_version,
        "deletion_resource_version": proof.deletion_resource_version,
        "restart_count_before": proof.restart_count_before,
        "restart_count_observed": proof.restart_count_observed,
        "termination_source": proof.termination_source,
        "termination_event_type": proof.termination_event_type,
        "deletion_event_type": proof.deletion_event_type,
        "terminated": {
            "exit_code": proof.exit_code,
            "reason": proof.termination_reason,
            "signal": proof.signal,
            "started_at": proof.started_at,
            "finished_at": proof.finished_at,
        },
        "adjacent_current_container_id":
            proof.adjacent_current_container_id,
        "pod_deletion_observed": proof.pod_deleted,
        "raw_pod_json_sha256":
            proof.termination_raw_pod_json_sha256,
        "termination_raw_observation_sequence":
            proof.termination_raw_observation_sequence,
        "deletion_raw_pod_json_sha256":
            proof.deletion_raw_pod_json_sha256,
        "deletion_raw_observation_sequence":
            proof.deletion_raw_observation_sequence,
    }


def _termination_authority_record(
    proof: TerminationProof,
) -> dict[str, object]:
    return seal_termination_record(_termination_authority_payload(proof))


def _validate_durable_termination_record(
    run_state: dict[str, object],
    proof: TerminationProof,
) -> dict[str, object]:

    record = _termination_authority_record(proof)
    journal = run_state.get("observation_journal")
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    references = (
        (
            proof.termination_raw_observation_sequence,
            proof.termination_raw_pod_json_sha256,
            proof.observed_resource_version,
            "MODIFIED",
        ),
        (
            proof.deletion_raw_observation_sequence,
            proof.deletion_raw_pod_json_sha256,
            proof.deletion_resource_version,
            "DELETED",
        ),
    )
    for sequence, pod_digest, resource_version, event_type in references:
        if (
            not isinstance(sequence, int)
            or not 1 <= sequence <= len(journal)
        ):
            raise RuntimeError(
                "termination raw observation journal reference is invalid"
            )
        raw = journal[sequence - 1]
        raw_event = (
            raw.get("raw_kubernetes_watch_event")
            if isinstance(raw, dict) else None
        )
        if (
            not isinstance(raw, dict)
            or raw.get("kind")
            != "raw_kubernetes_watch_event_observed"
            or raw.get("raw_pod_json_sha256") != pod_digest
            or raw.get("resource_version") != resource_version
            or not isinstance(raw_event, dict)
            or raw_event.get("type") != event_type
        ):
            raise RuntimeError(
                "termination raw observation journal reference mismatch"
            )
    return record


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _canonical_sha256(value: object) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_startup_permit(
    value: object,
    *,
    expected_generation: str | None = None,
    expected_mode: str | None = None,
) -> dict[str, object]:

    if not isinstance(value, dict) or set(value) != STARTUP_PERMIT_FIELDS:
        raise ValueError("startup permit fields are not exact")
    permit = dict(value)
    if permit["schema_version"] != STARTUP_PERMIT_SCHEMA:
        raise ValueError("startup permit schema_version mismatch")
    if permit["issuance_mode"] not in {"INIT", "RESTART"}:
        raise ValueError("startup permit issuance_mode is invalid")
    for field in ("permit_id", "topology_generation"):
        if not isinstance(permit[field], str) or not permit[field]:
            raise ValueError(f"startup permit {field} must be a string")
    if (
        expected_generation is not None
        and permit["topology_generation"] != expected_generation
    ):
        raise ValueError("startup permit topology_generation mismatch")
    if expected_mode is not None and permit["issuance_mode"] != expected_mode:
        raise ValueError("startup permit issuance_mode mismatch")

    members = permit["members"]
    if not isinstance(members, dict) or set(members) != set(MEMBERS):
        raise ValueError("startup permit requires exact four members")
    if any(
        not isinstance(pod_uid, str) or not pod_uid
        for pod_uid in members.values()
    ):
        raise ValueError("startup permit Pod UIDs must be non-empty strings")
    if len(set(members.values())) != len(MEMBERS):
        raise ValueError("startup permit Pod UIDs must be unique")
    permit["members"] = dict(members)

    digest = permit["canonical_digest"]
    unsigned = {
        key: item for key, item in permit.items()
        if key != "canonical_digest"
    }
    if (
        not isinstance(digest, str)
        or digest != _canonical_sha256(unsigned)
    ):
        raise ValueError("startup permit canonical_digest mismatch")
    return permit


def build_startup_permit(
    *,
    topology_generation: str,
    members: dict[str, str],
    issuance_mode: str,
    permit_id: str | None = None,
) -> dict[str, object]:

    unsigned: dict[str, object] = {
        "schema_version": STARTUP_PERMIT_SCHEMA,
        "issuance_mode": issuance_mode,
        "permit_id": permit_id or str(uuid.uuid4()),
        "topology_generation": topology_generation,
        "members": dict(members),
    }
    permit = {
        **unsigned,
        "canonical_digest": _canonical_sha256(unsigned),
    }
    return validate_startup_permit(
        permit,
        expected_generation=topology_generation,
        expected_mode=issuance_mode,
    )


def validate_initialization_state(value: object) -> dict[str, object]:

    if not isinstance(value, dict):
        raise ValueError("initialization state must be an object")
    state = dict(value)
    phase = state.get("phase")
    fields_by_phase = {
        "UNINITIALIZED": {"phase"},
        "INITIALIZING": {
            "phase",
            "initialize_run_id",
            "permit_id",
            "topology_generation",
        },
        "ACCEPTED": {
            "phase",
            "initialize_run_id",
            "permit_id",
            "topology_generation",
            "accepted_generation",
            "accepted_permit_id",
        },
    }
    expected_fields = fields_by_phase.get(phase)
    if expected_fields is None:
        raise ValueError("initialization state phase is invalid")
    if set(state) != expected_fields:
        raise ValueError("initialization state fields are not exact")
    for field in expected_fields - {"phase"}:
        if not isinstance(state[field], str) or not state[field]:
            raise ValueError(f"initialization state {field} must be a string")
    if phase == "ACCEPTED" and (
        state["accepted_generation"] != state["topology_generation"]
        or state["accepted_permit_id"] != state["permit_id"]
    ):
        raise ValueError("accepted initialization identity is inconsistent")
    return state


def _initializing_state(
    *,
    initialize_run_id: str,
    permit_id: str,
    topology_generation: str,
) -> dict[str, object]:
    return validate_initialization_state({
        "phase": "INITIALIZING",
        "initialize_run_id": initialize_run_id,
        "permit_id": permit_id,
        "topology_generation": topology_generation,
    })


def _accepted_initialization_state(
    initializing: dict[str, object],
) -> dict[str, object]:
    state = validate_initialization_state(initializing)
    if state["phase"] != "INITIALIZING":
        raise ValueError("only INITIALIZING can become ACCEPTED")
    return validate_initialization_state({
        **state,
        "phase": "ACCEPTED",
        "accepted_generation": state["topology_generation"],
        "accepted_permit_id": state["permit_id"],
    })


def _require_initialization_identity(
    state: dict[str, object],
    *,
    initialize_run_id: str,
    permit_id: str,
    topology_generation: str,
    phases: tuple[str, ...] = ("INITIALIZING", "ACCEPTED"),
) -> dict[str, object]:
    state = validate_initialization_state(state)
    if state["phase"] not in phases or any((
        state.get("initialize_run_id") != initialize_run_id,
        state.get("permit_id") != permit_id,
        state.get("topology_generation") != topology_generation,
    )):
        raise RuntimeError(
            "topology ConfigMap belongs to a foreign initialization run"
        )
    return state


def _validate_permitted_identities(
    evidence: dict[str, object],
    startup_permit: dict[str, object],
) -> None:

    permit = validate_startup_permit(startup_permit)
    identities = evidence.get("identities")
    if not isinstance(identities, list) or len(identities) != len(MEMBERS):
        raise RuntimeError(
            "fresh evidence does not contain four startup permit identities"
        )
    observed: dict[str, str] = {}
    for identity in identities:
        if not isinstance(identity, dict):
            raise RuntimeError("fresh startup permit identity is invalid")
        member = identity.get("instance_id")
        pod_uid = identity.get("pod_uid")
        process_generation = identity.get("process_generation")
        if (
            member not in MEMBERS
            or member in observed
            or not isinstance(pod_uid, str)
            or not pod_uid
            or not isinstance(process_generation, str)
            or not process_generation
            or identity.get("topology_generation")
            != permit["topology_generation"]
        ):
            raise RuntimeError(
                "fresh evidence identity does not match startup permit"
            )
        observed[str(member)] = pod_uid
    if observed != permit["members"]:
        raise RuntimeError(
            "fresh evidence Pod UIDs do not match startup permit"
        )


def _append_observation(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    stage: str,
    observation: TerminationProof | TerminationObservation,
    observation_status: str | None = None,
) -> str:

    journal = run_state.setdefault("observation_journal", [])
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    observed = _as_termination_observation(observation)
    document = _termination_proof_document(observed.proof)
    digest = _canonical_sha256(document)
    entry = {
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": len(journal) + 1,
        "kind": "termination_observed",
        "stage": stage,
        "observed_at_ns": time.monotonic_ns(),
        "proof": document,
        "proof_sha256": digest,
    }
    if observation_status is not None:
        entry["observation_status"] = observation_status
    has_raw_watch_event = any((
        observed.raw_watch_event is not None,
        bool(observed.raw_watch_event_text),
        bool(observed.raw_watch_event_sha256),
        bool(observed.resource_version),
    ))
    if has_raw_watch_event:
        if (
            observed.raw_watch_event is None
            or not observed.raw_watch_event_text
            or not observed.raw_watch_event_sha256
        ):
            raise RuntimeError(
                "raw Kubernetes watch event evidence is incomplete"
            )
        actual_raw_digest = (
            "sha256:"
            + hashlib.sha256(
                observed.raw_watch_event_text.encode("utf-8")
            ).hexdigest()
        )
        if observed.raw_watch_event_sha256 != actual_raw_digest:
            raise RuntimeError("raw Kubernetes watch event digest mismatch")
        try:
            decoded = json.loads(observed.raw_watch_event_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "raw Kubernetes watch event is not JSON"
            ) from exc
        if decoded != observed.raw_watch_event:
            raise RuntimeError(
                "raw Kubernetes watch event object/text mismatch"
            )
        raw_pod_json = observed.raw_watch_event.get("object")
        if not isinstance(raw_pod_json, dict):
            raise RuntimeError(
                "raw Kubernetes watch event lacks Pod object"
            )
        entry.update({
            "raw_kubernetes_watch_event": observed.raw_watch_event,
            "raw_kubernetes_watch_event_text": observed.raw_watch_event_text,
            "raw_kubernetes_watch_event_sha256": actual_raw_digest,
            "raw_pod_json": raw_pod_json,
            "raw_pod_json_sha256": _canonical_sha256(raw_pod_json),
            "resource_version": observed.resource_version,
        })
    if observed.raw_observation_sequence is not None:
        sequence = observed.raw_observation_sequence
        if not 1 <= sequence <= len(journal):
            raise RuntimeError("raw observation journal sequence is invalid")
        raw_entry = journal[sequence - 1]
        if (
            not isinstance(raw_entry, dict)
            or raw_entry.get("kind")
            != "raw_kubernetes_watch_event_observed"
            or raw_entry.get("raw_kubernetes_watch_event_sha256")
            != observed.raw_watch_event_sha256
        ):
            raise RuntimeError("raw observation journal reference mismatch")
        entry["raw_observation_sequence"] = sequence
    journal.append(entry)
    _atomic_write_json(run_state_path, run_state)
    return digest


def _append_semantic_result(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    stage: str,
    proof_sha256: str,
    status: str,
    message: str,
) -> None:
    if status not in {"PASS", "FAIL"}:
        raise ValueError("termination semantic status must be PASS or FAIL")
    journal = run_state.setdefault("observation_journal", [])
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    journal.append({
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": len(journal) + 1,
        "kind": "termination_semantic_validation",
        "stage": stage,
        "observed_at_ns": time.monotonic_ns(),
        "proof_sha256": proof_sha256,
        "semantic_status": status,
        "message": message,
    })
    _atomic_write_json(run_state_path, run_state)


def _append_raw_watch_observation(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    stage: str,
    event: KubernetesWatchEvent,
) -> tuple[int, str]:

    journal = run_state.setdefault("observation_journal", [])
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    actual_digest = (
        "sha256:"
        + hashlib.sha256(
            event.raw_text.encode("utf-8")
        ).hexdigest()
    )
    if event.raw_sha256 != actual_digest:
        raise RuntimeError("raw Kubernetes watch event digest mismatch")
    entry = {
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": len(journal) + 1,
        "kind": "raw_kubernetes_watch_event_observed",
        "stage": stage,
        "observed_at_ns": time.monotonic_ns(),
        "raw_kubernetes_watch_event_text": event.raw_text,
        "raw_kubernetes_watch_event_sha256": actual_digest,
    }
    try:
        decoded = json.loads(event.raw_text)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        entry["raw_kubernetes_watch_event"] = decoded
    if isinstance(decoded, dict) and isinstance(decoded.get("object"), dict):
        raw_pod_json = decoded["object"]
        entry["raw_pod_json"] = raw_pod_json
        entry["raw_pod_json_sha256"] = _canonical_sha256(raw_pod_json)
        metadata = raw_pod_json.get("metadata")
        if isinstance(metadata, dict):
            resource_version = metadata.get("resourceVersion")
            if resource_version is not None:
                entry["resource_version"] = str(resource_version)
    journal.append(entry)
    _atomic_write_json(run_state_path, run_state)
    return int(entry["sequence"]), actual_digest


def _append_incomplete_observation(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    stage: str,
    error: Exception,
) -> None:

    partial = getattr(error, "partial_observation", None)
    partial_digest = ""
    if isinstance(partial, TerminationObservation):
        partial_digest = _append_observation(
            run_state,
            run_state_path,
            stage=stage,
            observation=partial,
            observation_status="PARTIAL",
        )
    journal = run_state.setdefault("observation_journal", [])
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    entry = {
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": len(journal) + 1,
        "kind": "termination_observation_incomplete",
        "stage": stage,
        "observed_at_ns": time.monotonic_ns(),
        "observation_status": "INCOMPLETE",
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if partial_digest:
        entry["partial_proof_sha256"] = partial_digest
    raw_sequence = getattr(error, "raw_observation_sequence", None)
    raw_digest = str(
        getattr(error, "raw_watch_event_sha256", "") or ""
    )
    if isinstance(raw_sequence, int):
        if not 1 <= raw_sequence <= len(journal):
            raise RuntimeError("raw observation journal sequence is invalid")
        raw_entry = journal[raw_sequence - 1]
        if (
            not isinstance(raw_entry, dict)
            or raw_entry.get("kind")
            != "raw_kubernetes_watch_event_observed"
            or raw_entry.get("raw_kubernetes_watch_event_sha256")
            != raw_digest
        ):
            raise RuntimeError("raw observation journal reference mismatch")
        entry["raw_observation_sequence"] = raw_sequence
        entry["raw_kubernetes_watch_event_sha256"] = raw_digest
        raw_event = getattr(error, "raw_watch_event", None)
        if raw_event is not None:
            entry["raw_kubernetes_watch_event"] = raw_event
            if (
                isinstance(raw_event, dict)
                and raw_event.get("type") != "ERROR"
                and isinstance(raw_event.get("object"), dict)
            ):
                raw_pod_json = raw_event["object"]
                entry["raw_pod_json"] = raw_pod_json
                entry["raw_pod_json_sha256"] = _canonical_sha256(
                    raw_pod_json
                )
                entry["resource_version"] = str(
                    getattr(error, "resource_version", "") or ""
                )
    else:
        raw_text = str(
            getattr(error, "raw_watch_event_text", "") or ""
        )
        if raw_text:
            actual_digest = (
                "sha256:"
                + hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            )
            if raw_digest != actual_digest:
                raise RuntimeError(
                    "raw Kubernetes watch event digest mismatch"
                )
            entry["raw_kubernetes_watch_event_text"] = raw_text
            entry["raw_kubernetes_watch_event_sha256"] = actual_digest
            raw_event = getattr(error, "raw_watch_event", None)
            if raw_event is not None:
                entry["raw_kubernetes_watch_event"] = raw_event
        else:
            entry["no_raw_reason"] = (
                "raw Kubernetes watch event was not observed before failure"
            )
    journal.append(entry)
    _atomic_write_json(run_state_path, run_state)


def _fault_expected_identity(
    identity: ProcessIdentity,
) -> dict[str, object]:
    return {
        "logical_instance_id": identity.member,
        "topology_generation": identity.topology_generation,
        "pod_uid": identity.pod_uid,
        "node_uid": identity.node_uid,
        "container_name": identity.container_name,
        "captured_container_id": identity.container_id,
        "process_generation": identity.process_generation,
        "watch_start_resource_version": identity.resource_version,
        "restart_count_before": identity.restart_count,
    }


def _append_fault_injection_attempt(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    identity: ProcessIdentity,
) -> int:
    journal = run_state.setdefault("observation_journal", [])
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    entry = {
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": len(journal) + 1,
        "kind": "fault_injection_attempt",
        "stage": "worker_crash_injection",
        "observed_at_ns": time.monotonic_ns(),
        "expected_identity": _fault_expected_identity(identity),
    }
    journal.append(entry)
    _atomic_write_json(run_state_path, run_state)
    return int(entry["sequence"])


def _injection_command_bundle(
    value: object | None = None,
    *,
    error: Exception | None = None,
) -> dict[str, object]:
    mapping = value if isinstance(value, dict) else {}
    command = mapping.get("command")
    return_code = mapping.get("exec_return_code")
    stdout = mapping.get("stdout", "")
    stderr = mapping.get("stderr", "")
    if error is not None:
        command = getattr(error, "cmd", command)
        return_code = getattr(error, "returncode", return_code)
        stdout = getattr(error, "stdout", stdout)
        stderr = getattr(error, "stderr", stderr)
    if not isinstance(command, (list, tuple)):
        command = []
    return {
        "command": [str(item) for item in command],
        "exec_return_code": (
            return_code
            if isinstance(return_code, int)
            and not isinstance(return_code, bool)
            else None
        ),
        "stdout": "" if stdout is None else str(stdout),
        "stderr": "" if stderr is None else str(stderr),
    }


def _append_fault_injection_incomplete(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    attempt_sequence: int,
    bundle: dict[str, object],
    error: Exception,
) -> None:
    journal = run_state.setdefault("observation_journal", [])
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    journal.append({
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": len(journal) + 1,
        "kind": "fault_injection_incomplete",
        "stage": "worker_crash_injection",
        "observed_at_ns": time.monotonic_ns(),
        "observation_status": "INCOMPLETE",
        "attempt_sequence": attempt_sequence,
        "error_type": type(error).__name__,
        "message": str(error),
        **bundle,
    })
    _atomic_write_json(run_state_path, run_state)


def _validate_fault_injection_result(
    value: object,
    expected: ProcessIdentity,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(value, dict):
        raise RuntimeError(
            "worker crash injection lacks exact process selection evidence"
        )
    bundle = _injection_command_bundle(value)
    return_code = bundle["exec_return_code"]
    if return_code not in {0, 137}:
        raise RuntimeError(
            f"worker crash injection helper exited {return_code}"
        )
    selection_value = value.get("selection")
    if isinstance(selection_value, dict):
        selection = dict(selection_value)
    elif value.get("component") is not None:
        selection = {
            key: item
            for key, item in value.items()
            if key not in {
                "command",
                "exec_return_code",
                "stdout",
                "stderr",
            }
        }
    else:
        lines = str(bundle["stdout"]).strip().splitlines()
        try:
            selection = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "worker crash injection did not return process selection "
                "evidence"
            ) from exc
    expected_epoch = f"{expected.pod_uid}:{expected.process_generation}"
    if (
        not isinstance(selection, dict)
        or selection.get("component") != "worker"
        or selection.get("instance_id") != expected.member
        or selection.get("pod_uid") != expected.pod_uid
        or selection.get("process_generation")
        != expected.process_generation
        or selection.get("instance_epoch") != expected_epoch
        or type(selection.get("app_pid")) is not int
        or int(selection["app_pid"]) <= 1
        or type(selection.get("process_start_ticks")) is not int
        or int(selection["process_start_ticks"]) <= 0
        or selection.get("pidfd") is not True
        or selection.get("signal") != 9
    ):
        raise RuntimeError(
            "worker crash process selection does not bind captured identity"
        )
    return selection, bundle


def _append_fault_injection_result(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    attempt_sequence: int,
    selection: dict[str, object],
    bundle: dict[str, object],
) -> None:
    journal = run_state.setdefault("observation_journal", [])
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    journal.append({
        "schema_version": OBSERVATION_SCHEMA,
        "sequence": len(journal) + 1,
        "kind": "fault_injection_result",
        "stage": "worker_crash_injection",
        "observed_at_ns": time.monotonic_ns(),
        "observation_status": "COMPLETE",
        "attempt_sequence": attempt_sequence,
        "selection": selection,
        **bundle,
    })
    _atomic_write_json(run_state_path, run_state)


def _observe_or_persist_incomplete(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    stage: str,
    observer: Callable[[], TerminationProof | TerminationObservation],
) -> TerminationProof | TerminationObservation:
    try:
        return observer()
    except Exception as exc:
        _append_incomplete_observation(
            run_state,
            run_state_path,
            stage=stage,
            error=exc,
        )
        raise


def _persist_fault_termination_observation(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    stage: str,
    observation: TerminationProof | TerminationObservation,
) -> tuple[TerminationProof, str]:
    observed = _as_termination_observation(observation)
    proof = observed.proof
    fault = run_state.get("fault")
    if not isinstance(fault, dict):
        raise RuntimeError("fault observation lacks durable fault state")
    fault.update({
        "observed_exit_code": proof.exit_code,
        "termination_reason": proof.termination_reason,
        "termination_message": proof.termination_message,
        "termination_message_sha256": (
            "sha256:"
            + hashlib.sha256(
                proof.termination_message.encode("utf-8")
            ).hexdigest()
        ),
        "finished_at": proof.finished_at,
    })
    digest = _append_observation(
        run_state,
        run_state_path,
        stage=stage,
        observation=observed,
    )
    return proof, digest


def _validate_and_record_semantics(
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    stage: str,
    proof_sha256: str,
    validator: Callable[[], None],
) -> None:
    try:
        validator()
    except Exception as exc:
        _append_semantic_result(
            run_state,
            run_state_path,
            stage=stage,
            proof_sha256=proof_sha256,
            status="FAIL",
            message=str(exc),
        )
        raise
    _append_semantic_result(
        run_state,
        run_state_path,
        stage=stage,
        proof_sha256=proof_sha256,
        status="PASS",
        message="termination observation satisfies the fixed semantic contract",
    )


def _validate_observed_process_identity(
    proof: TerminationProof,
    expected: ProcessIdentity,
) -> None:
    if (
        proof.member != expected.member
        or proof.pod_uid != expected.pod_uid
        or proof.node_uid != expected.node_uid
        or proof.container_id != expected.container_id
        or proof.process_generation != expected.process_generation
        or proof.container_terminated is not True
    ):
        raise RuntimeError(
            "termination observation does not bind the captured old process"
        )


def _validate_worker_crash_termination(
    proof: TerminationProof,
    expected: ProcessIdentity,
) -> None:
    _validate_observed_process_identity(proof, expected)
    if proof.exit_code != 137:
        raise RuntimeError(
            "worker_crash proof requires the injected container to exit 137"
        )


def _validate_watchdog_termination(
    proof: TerminationProof,
    expected: ProcessIdentity,
    *,
    expected_generation: str,
    expected_operation_ids: tuple[str, ...],
) -> None:
    _validate_observed_process_identity(proof, expected)
    if proof.exit_code != 70:
        raise RuntimeError("NCCL watchdog process must exit 70")
    try:
        message = json.loads(proof.termination_message)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "NCCL watchdog termination message is not JSON"
        ) from exc
    if not isinstance(message, dict):
        raise RuntimeError(
            "NCCL watchdog termination message must be an object"
        )
    ref = message.get("endpoint_ref")
    expected_epoch = f"{expected.pod_uid}:{expected.process_generation}"
    pair_members = str(message.get("pair_id") or "").split("--")
    if (
        message.get("schema_version") != 1
        or message.get("kind") != "nccl_watchdog_timeout"
        or message.get("instance_id") != expected.member
        or message.get("pod_uid") != expected.pod_uid
        or message.get("process_generation") != expected.process_generation
        or message.get("instance_epoch") != expected_epoch
        or message.get("topology_generation") != expected_generation
        or message.get("expected_exit_code") != 70
        or not isinstance(ref, dict)
        or ref.get("target_instance") != expected.member
        or ref.get("target_worker_epoch") != expected_epoch
        or ref.get("topology_generation") != expected_generation
        or not str(ref.get("owner_generation") or "")
        or isinstance(ref.get("operation_seq"), bool)
        or not isinstance(ref.get("operation_seq"), int)
        or ref.get("operation_seq") <= 0
        or not str(ref.get("payload_digest") or "")
        or message.get("operation_id") not in set(expected_operation_ids)
        or ref.get("operation_id") != message.get("operation_id")
        or len(pair_members) != 2
        or expected.member not in pair_members
        or any(member not in MEMBERS for member in pair_members)
        or not str(message.get("reason") or "")
    ):
        raise RuntimeError(
            "NCCL watchdog termination message identity mismatch"
        )


def _record_protocol_event(
    run_state: dict[str, object], *, name: str, evidence: object,
) -> None:
    """Append one actuator-observed event using this process' monotonic clock."""
    events = run_state.setdefault("protocol_events", [])
    if not isinstance(events, list):
        raise RuntimeError("pd-worldctl protocol_events must be a list")
    if any(
        isinstance(value, dict) and value.get("name") == name
        for value in events
    ):
        raise RuntimeError(f"pd-worldctl protocol event repeated: {name}")
    evidence_bytes = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    events.append({
        "name": name,
        "sequence": len(events) + 1,
        "observed_at_ns": time.monotonic_ns(),
        "clock": "operator_monotonic",
        "producer_epoch": str(run_state["restart_run_id"]),
        "evidence": json.loads(evidence_bytes.decode("utf-8")),
        "evidence_sha256": "sha256:" + hashlib.sha256(evidence_bytes).hexdigest(),
    })


def _raw_observation_sink_options(
    backend: object,
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    stage: str,
) -> dict[str, object]:
    if not getattr(backend, "raw_observation_sink_supported", False):
        return {}

    def persist(
        event: KubernetesWatchEvent,
    ) -> tuple[int, str]:
        return _append_raw_watch_observation(
            run_state,
            run_state_path,
            stage=stage,
            event=event,
        )

    return {"raw_observation_sink": persist}


def _capture_ambiguous_injection_evidence(
    backend: object,
    run_state: dict[str, object],
    run_state_path: Path,
    *,
    release: str,
    namespace: str,
    watch: object,
    expected: ProcessIdentity,
) -> None:

    observer = getattr(backend, "observe_termination", None)
    if observer is None:
        observer = getattr(backend, "wait_termination")
    options = _raw_observation_sink_options(
        backend,
        run_state,
        run_state_path,
        stage="worker_crash_injection_ambiguous",
    )
    try:
        observation = _observe_or_persist_incomplete(
            run_state,
            run_state_path,
            stage="worker_crash_injection_ambiguous",
            observer=lambda: observer(
                release,
                namespace,
                watch,
                expected,
                **options,
            ),
        )
    except Exception:
        return
    _append_observation(
        run_state,
        run_state_path,
        stage="worker_crash_injection_ambiguous",
        observation=observation,
        observation_status="INCOMPLETE",
    )


class WorldInitializeActuator:

    def __init__(self, backend) -> None:
        self.backend = backend

    def initialize(
        self,
        *,
        release: str,
        namespace: str,
        chart: str,
        generation: str,
        run_state_path: Path,
    ) -> dict[str, object]:
        run_state = _load_or_create_initialize_run_state(
            run_state_path,
            release=release,
            namespace=namespace,
            chart=chart,
            generation=generation,
        )
        initialize_run_id = str(run_state["initialize_run_id"])
        permit_id = str(run_state["permit_id"])
        snapshot = self.backend.read_initialization_snapshot(
            release, namespace
        )
        bound_uid = run_state.get("configmap_uid")
        if bound_uid is None:
            run_state["configmap_uid"] = snapshot.configmap_uid
            _atomic_write_json(run_state_path, run_state)
        elif bound_uid != snapshot.configmap_uid:
            raise RuntimeError(
                "worker topology ConfigMap UID changed during initialization"
            )

        remote_state = validate_initialization_state(snapshot.state)
        if (
            run_state["phase"] == "ACCEPTED"
            and remote_state["phase"] != "ACCEPTED"
        ):
            raise RuntimeError(
                "accepted initialization state rolled back remotely"
            )
        if remote_state["phase"] == "UNINITIALIZED":
            if run_state["phase"] != "CREATED":
                raise RuntimeError(
                    "initialization state rolled back after claim"
                )
            if snapshot.startup_permit is not None:
                raise RuntimeError(
                    "UNINITIALIZED topology already contains a startup permit"
                )
            snapshot = self.backend.claim_initialization(
                release,
                namespace,
                snapshot,
                initialize_run_id=initialize_run_id,
                permit_id=permit_id,
                generation=generation,
            )
            remote_state = _require_initialization_identity(
                snapshot.state,
                initialize_run_id=initialize_run_id,
                permit_id=permit_id,
                topology_generation=generation,
                phases=("INITIALIZING",),
            )
            run_state["phase"] = "CLAIMED"
            _atomic_write_json(run_state_path, run_state)
        else:
            remote_state = _require_initialization_identity(
                remote_state,
                initialize_run_id=initialize_run_id,
                permit_id=permit_id,
                topology_generation=generation,
            )

        if remote_state["phase"] == "ACCEPTED":
            permit = validate_startup_permit(
                snapshot.startup_permit,
                expected_generation=generation,
                expected_mode="INIT",
            )
            if permit["permit_id"] != permit_id:
                raise RuntimeError(
                    "accepted initialization permit identity changed"
                )
            expected_permit = run_state.get("startup_permit")
            if expected_permit is not None and permit != validate_startup_permit(
                expected_permit,
                expected_generation=generation,
                expected_mode="INIT",
            ):
                raise RuntimeError(
                    "accepted initialization permit changed after resume"
                )
            accepted = {
                "accepted": True,
                "initialize_run_id": initialize_run_id,
                "permit_id": permit_id,
                "topology_generation": generation,
            }
            persist = getattr(
                self.backend, "persist_initialized_world", None
            )
            if persist is not None:
                persist(
                    release,
                    namespace,
                    chart,
                    generation,
                    permit,
                    remote_state,
                )
            run_state["phase"] = "ACCEPTED"
            run_state["startup_permit"] = permit
            run_state["accepted_response"] = accepted
            _atomic_write_json(run_state_path, run_state)
            return accepted

        self.backend.verify_initial_worker_templates(
            release, namespace, generation=generation
        )
        for member in MEMBERS:
            self.backend.start_member(release, namespace, member)
        pod_uids = self.backend.wait_initial_pod_uids(
            release, namespace, generation=generation
        )

        if snapshot.startup_permit is None:
            permit = build_startup_permit(
                topology_generation=generation,
                members=pod_uids,
                issuance_mode="INIT",
                permit_id=permit_id,
            )
            run_state["startup_permit"] = permit
            run_state["phase"] = "PERMIT_READY"
            _atomic_write_json(run_state_path, run_state)
            snapshot = self.backend.publish_initialization_permit(
                release,
                namespace,
                snapshot,
                startup_permit=permit,
                initialize_run_id=initialize_run_id,
                permit_id=permit_id,
                generation=generation,
            )
        else:
            permit = validate_startup_permit(
                snapshot.startup_permit,
                expected_generation=generation,
                expected_mode="INIT",
            )
            if permit["permit_id"] != permit_id:
                raise RuntimeError(
                    "initialization contains a foreign startup permit"
                )
            expected_permit = run_state.get("startup_permit")
            if expected_permit is not None and permit != validate_startup_permit(
                expected_permit,
                expected_generation=generation,
                expected_mode="INIT",
            ):
                raise RuntimeError(
                    "initialization startup permit changed after resume"
                )
            run_state["startup_permit"] = permit
            run_state["phase"] = "PERMIT_READY"
            _atomic_write_json(run_state_path, run_state)

        if permit["members"] != pod_uids:
            raise RuntimeError(
                "INIT permit Pod UID drift requires guarded whole-world restart"
            )
        current_pod_uids = self.backend.wait_initial_pod_uids(
            release, namespace, generation=generation
        )
        if current_pod_uids != permit["members"]:
            raise RuntimeError(
                "INIT permit Pod UID drift requires guarded whole-world restart"
            )

        evidence = self.backend.wait_initialization_evidence(
            generation=generation
        )
        _validate_permitted_identities(evidence, permit)
        if evidence.get("ready") is not True:
            raise RuntimeError("initialization evidence is not ready")
        self.backend.wait_gateway_ready()
        accepted_pod_uids = self.backend.wait_initial_pod_uids(
            release, namespace, generation=generation
        )
        if accepted_pod_uids != permit["members"]:
            raise RuntimeError(
                "INIT permit Pod UID drift requires guarded whole-world restart"
            )

        latest = self.backend.read_initialization_snapshot(
            release, namespace
        )
        if latest.configmap_uid != snapshot.configmap_uid:
            raise RuntimeError(
                "worker topology ConfigMap UID changed before acceptance"
            )
        _require_initialization_identity(
            latest.state,
            initialize_run_id=initialize_run_id,
            permit_id=permit_id,
            topology_generation=generation,
            phases=("INITIALIZING",),
        )
        if latest.startup_permit != permit:
            raise RuntimeError(
                "initialization startup permit changed before acceptance"
            )
        accepted_state = self.backend.accept_initialization(
            release,
            namespace,
            latest,
            initialize_run_id=initialize_run_id,
            permit_id=permit_id,
            generation=generation,
        )
        accepted_state = _require_initialization_identity(
            accepted_state,
            initialize_run_id=initialize_run_id,
            permit_id=permit_id,
            topology_generation=generation,
            phases=("ACCEPTED",),
        )
        persist = getattr(self.backend, "persist_initialized_world", None)
        if persist is not None:
            persist(
                release,
                namespace,
                chart,
                generation,
                permit,
                accepted_state,
            )
        accepted = {
            "accepted": True,
            "initialize_run_id": initialize_run_id,
            "permit_id": permit_id,
            "topology_generation": generation,
        }
        run_state["phase"] = "ACCEPTED"
        run_state["accepted_response"] = accepted
        _atomic_write_json(run_state_path, run_state)
        return accepted


RESTART_CHECKPOINTS = (
    "CREATED",
    "CAPTURED",
    "TERMINATED",
    "GENERATION_PATCHED",
    "STARTED",
    "PERMIT_READY",
    "PERMIT_PUBLISHED",
    "EVIDENCE_READY",
    "ACCEPTED",
)


def _captured_identity_document(
    identity: ProcessIdentity,
) -> dict[str, object]:
    return {
        "member": identity.member,
        "pod_uid": identity.pod_uid,
        "node_uid": identity.node_uid,
        "container_id": identity.container_id,
        "process_generation": identity.process_generation,
        "pod_name": identity.pod_name,
        "node_name": identity.node_name,
        "resource_version": identity.resource_version,
        "topology_generation": identity.topology_generation,
        "container_name": identity.container_name,
        "restart_count": identity.restart_count,
    }


def _restore_captured_world(
    run_state: dict[str, object],
    *,
    expected_generation: str,
) -> dict[str, ProcessIdentity]:
    values = run_state.get("captured_identities")
    if not isinstance(values, list) or len(values) != len(MEMBERS):
        raise RuntimeError(
            "STOPPING run state lacks the exact captured old world"
        )
    fields = set(ProcessIdentity.__dataclass_fields__)
    restored: dict[str, ProcessIdentity] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != fields:
            raise RuntimeError("captured old process identity is invalid")
        try:
            identity = ProcessIdentity(**value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "captured old process identity is invalid"
            ) from exc
        if (
            identity.member not in MEMBERS
            or identity.member in restored
            or identity.topology_generation != expected_generation
        ):
            raise RuntimeError(
                "captured old process identity changed across resume"
            )
        restored[identity.member] = identity
    if set(restored) != set(MEMBERS):
        raise RuntimeError("captured old world is incomplete")
    return restored


def _proof_from_document(value: object) -> TerminationProof:
    if (
        not isinstance(value, dict)
        or set(value) != set(TerminationProof.__dataclass_fields__)
    ):
        raise RuntimeError("durable termination proof document is invalid")
    try:
        proof = TerminationProof(**value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "durable termination proof document is invalid"
        ) from exc
    if _termination_proof_document(proof) != value:
        raise RuntimeError("durable termination proof document is not exact")
    return proof


def _validated_journal_proofs(
    run_state: dict[str, object],
    *,
    stage: str,
) -> dict[str, TerminationProof]:
    journal = run_state.get("observation_journal")
    if not isinstance(journal, list):
        raise RuntimeError("pd-worldctl observation_journal must be a list")
    passed = {
        str(entry.get("proof_sha256"))
        for entry in journal
        if (
            isinstance(entry, dict)
            and entry.get("kind") == "termination_semantic_validation"
            and entry.get("stage") == stage
            and entry.get("semantic_status") == "PASS"
        )
    }
    failed = {
        str(entry.get("proof_sha256"))
        for entry in journal
        if (
            isinstance(entry, dict)
            and entry.get("kind") == "termination_semantic_validation"
            and entry.get("stage") == stage
            and entry.get("semantic_status") == "FAIL"
        )
    }
    if passed & failed:
        raise RuntimeError(
            "durable termination semantic verdict changed for one proof"
        )
    proofs: dict[str, TerminationProof] = {}
    for entry in journal:
        if (
            not isinstance(entry, dict)
            or entry.get("kind") != "termination_observed"
            or entry.get("stage") != stage
            or entry.get("proof_sha256") not in passed
        ):
            continue
        document = entry.get("proof")
        proof = _proof_from_document(document)
        if entry.get("proof_sha256") != _canonical_sha256(document):
            raise RuntimeError("durable termination proof digest mismatch")
        prior = proofs.get(proof.member)
        if prior is not None and prior != proof:
            raise RuntimeError(
                "durable termination proof changed across resume"
            )
        proofs[proof.member] = proof
    return proofs


def _checkpoint_termination_proof(
    run_state: dict[str, object],
    run_state_path: Path,
    proof: TerminationProof,
) -> None:
    values = run_state.setdefault("termination_proofs", {})
    if not isinstance(values, dict):
        raise RuntimeError("termination_proofs checkpoint must be an object")
    document = _termination_proof_document(proof)
    prior = values.get(proof.member)
    if prior is not None and prior != document:
        raise RuntimeError("termination proof checkpoint changed for member")
    values[proof.member] = document
    _atomic_write_json(run_state_path, run_state)


def _restore_termination_checkpoint(
    run_state: dict[str, object],
) -> dict[str, TerminationProof]:
    values = run_state.get("termination_proofs", {})
    if not isinstance(values, dict):
        raise RuntimeError("termination_proofs checkpoint must be an object")
    if not set(values).issubset(set(MEMBERS)):
        raise RuntimeError("termination proof checkpoint has unknown member")
    return {
        member: _proof_from_document(document)
        for member, document in values.items()
    }


def _restart_checkpoint(run_state: dict[str, object]) -> str:
    checkpoint = run_state.get("checkpoint")
    if checkpoint not in RESTART_CHECKPOINTS:
        raise RuntimeError("pd-worldctl restart checkpoint is invalid")
    return str(checkpoint)


class WorldRestartActuator:
    """Execute the stop/watch/start/accept protocol against an injected backend."""

    def __init__(self, backend) -> None:
        self.backend = backend

    def restart(
        self, *, release: str, namespace: str, chart: str,
        old_generation: str, new_generation: str, run_state_path: Path,
        fault_member: str | None = None,
        watchdog_member: str | None = None,
        expected_operation_ids: tuple[str, ...] = (),
        required_old_operation_ids: tuple[str, ...] = (),
    ) -> dict[str, object]:
        if fault_member is not None and fault_member not in MEMBERS:
            raise ValueError("fault_member must be one of p0, p1, d0, d1")
        if watchdog_member is not None and watchdog_member not in MEMBERS:
            raise ValueError("watchdog_member must be one of p0, p1, d0, d1")
        if fault_member is not None and watchdog_member is not None:
            raise ValueError("worker crash and NCCL watchdog modes are exclusive")
        if watchdog_member is not None and not expected_operation_ids:
            raise ValueError(
                "NCCL watchdog mode requires an exact expected operation id"
            )
        run_state = _load_or_create_run_state(
            run_state_path,
            release=release,
            namespace=namespace,
            chart=chart,
            old_generation=old_generation,
            new_generation=new_generation,
            fault_member=fault_member,
            watchdog_member=watchdog_member,
            expected_operation_ids=expected_operation_ids,
            required_old_operation_ids=required_old_operation_ids,
        )
        if run_state["phase"] == "ACCEPTED":
            return dict(run_state["accepted_response"])
        if run_state["phase"] == "EVIDENCE_READY":
            if _restart_checkpoint(run_state) != "EVIDENCE_READY":
                raise RuntimeError(
                    "EVIDENCE_READY phase/checkpoint mismatch"
                )
            evidence = dict(run_state["evidence"])
            startup_permit = validate_startup_permit(
                run_state.get("startup_permit"),
                expected_generation=new_generation,
                expected_mode="RESTART",
            )
            _validate_permitted_identities(evidence, startup_permit)
            termination_records = evidence.get("termination_records")
            if not isinstance(termination_records, list):
                raise RuntimeError(
                    "durable acceptance evidence lacks termination records"
                )
            validate_termination_records(
                termination_records,
                expected_generation=old_generation,
                expected_members=MEMBERS,
            )
            accepted = self.backend.accept_topology(evidence)
            if accepted.get("accepted") is not True:
                raise RuntimeError("Gateway rejected replacement topology")
            if watchdog_member is not None:
                _record_protocol_event(
                    run_state,
                    name="topology_accept_succeeded",
                    evidence={"request": evidence, "response": accepted},
                )
            persist = getattr(self.backend, "persist_started_world", None)
            if persist is not None:
                persist(
                    release,
                    namespace,
                    chart,
                    new_generation,
                    startup_permit,
                )
            run_state["phase"] = "ACCEPTED"
            run_state["checkpoint"] = "ACCEPTED"
            run_state["accepted_response"] = accepted
            _atomic_write_json(run_state_path, run_state)
            return accepted
        if run_state["phase"] not in {"CREATED", "STOPPING"}:
            raise RuntimeError("unsupported pd-worldctl run-state phase")

        expected_ids = set(expected_operation_ids)
        checkpoint = _restart_checkpoint(run_state)
        if run_state["phase"] == "CREATED":
            if checkpoint != "CREATED":
                raise RuntimeError("CREATED phase/checkpoint mismatch")
            old = self.backend.capture_world(
                release, namespace, expected_generation=old_generation
            )
            if set(old) != set(MEMBERS):
                raise RuntimeError(
                    "capture must return exactly four old processes"
                )
            observed_ids = set(getattr(
                self.backend, "captured_operation_ids", ()
            ))
            if expected_ids and not expected_ids.issubset(observed_ids):
                raise RuntimeError(
                    "live old-world resource reports omit expected fault "
                    "operations"
                )
            captured_operation_ids = observed_ids | set(
                required_old_operation_ids
            )
            captured_resource_ids = set(getattr(
                self.backend, "captured_resource_ids", ()
            ))
        else:
            if checkpoint == "CREATED":
                raise RuntimeError("STOPPING phase/checkpoint mismatch")
            old = _restore_captured_world(
                run_state,
                expected_generation=old_generation,
            )
            operation_values = run_state.get("captured_operation_ids")
            resource_values = run_state.get("captured_resource_ids")
            if (
                not isinstance(operation_values, list)
                or not all(isinstance(value, str) for value in operation_values)
                or len(operation_values) != len(set(operation_values))
                or not isinstance(resource_values, list)
                or not all(isinstance(value, str) for value in resource_values)
                or len(resource_values) != len(set(resource_values))
            ):
                raise RuntimeError(
                    "captured old resource identity checkpoint is invalid"
                )
            captured_operation_ids = set(operation_values)
            captured_resource_ids = set(resource_values)
            if not expected_ids.issubset(captured_operation_ids):
                raise RuntimeError(
                    "durable old-world checkpoint omits expected fault "
                    "operations"
                )
        include_required = getattr(
            self.backend, "include_required_old_operation_ids", None
        )
        if include_required is not None:
            include_required(tuple(sorted(captured_operation_ids)))
        include_resources = getattr(
            self.backend, "include_required_old_resource_ids", None
        )
        if include_resources is not None:
            include_resources(tuple(sorted(captured_resource_ids)))
        guard = WorldRestartGuard(
            old_generation,
            new_generation,
            old,
            require_canonical_authority=True,
        )
        durable_proofs = _restore_termination_checkpoint(run_state)
        for member, proof in _validated_journal_proofs(
            run_state,
            stage="whole_world_termination",
        ).items():
            prior = durable_proofs.get(member)
            if prior is not None and prior != proof:
                raise RuntimeError(
                    "termination proof checkpoint disagrees with journal"
                )
            durable_proofs[member] = proof
        for proof in durable_proofs.values():
            guard.record_termination(proof)
            _checkpoint_termination_proof(
                run_state,
                run_state_path,
                proof,
            )
        watches: dict[str, object] = {}
        try:
            watch_members = (
                MEMBERS
                if checkpoint == "CREATED"
                else tuple(
                    member for member in MEMBERS
                    if member not in guard._proofs
                )
                if checkpoint == "CAPTURED"
                else ()
            )
            for member in watch_members:
                watches[member] = self.backend.start_termination_watch(
                    release, namespace, old[member]
                )
            if checkpoint == "CREATED":
                captured_identities = [
                    _captured_identity_document(old[name])
                    for name in MEMBERS
                ]
                run_state["phase"] = "STOPPING"
                run_state["checkpoint"] = "CAPTURED"
                run_state["captured_identities"] = captured_identities
                run_state["captured_operation_ids"] = sorted(
                    captured_operation_ids
                )
                run_state["captured_resource_ids"] = sorted(
                    captured_resource_ids
                )
                run_state["termination_proofs"] = {}
                if fault_member is not None:
                    run_state["fault"] = {
                        "kind": "worker_crash",
                        "member": fault_member,
                        "old_instance": {
                            "pod_uid": old[fault_member].pod_uid,
                            "container_id": old[fault_member].container_id,
                            "process_generation":
                                old[fault_member].process_generation,
                        },
                        "injection_requested_at_ns": time.monotonic_ns(),
                        "expected_exit_code": 137,
                    }
                elif watchdog_member is not None:
                    run_state["fault"] = {
                        "kind": "nccl_watchdog_timeout",
                        "member": watchdog_member,
                        "old_instance": {
                            "pod_uid": old[watchdog_member].pod_uid,
                            "container_id": old[watchdog_member].container_id,
                            "process_generation":
                                old[watchdog_member].process_generation,
                        },
                        "watch_started_at_ns": time.monotonic_ns(),
                        "expected_exit_code": 70,
                        "expected_operation_ids": sorted(expected_ids),
                    }
                    _record_protocol_event(
                        run_state,
                        name="all_termination_watches_established",
                        evidence=captured_identities,
                    )
                _atomic_write_json(run_state_path, run_state)
                checkpoint = "CAPTURED"
            if checkpoint != "CAPTURED":
                return self._continue_after_termination(
                    release=release,
                    namespace=namespace,
                    chart=chart,
                    old_generation=old_generation,
                    new_generation=new_generation,
                    run_state=run_state,
                    run_state_path=run_state_path,
                    old=old,
                    guard=guard,
                    watchdog_member=watchdog_member,
                )
            injected_termination = (
                _validated_journal_proofs(
                    run_state,
                    stage="worker_crash_injection",
                ).get(fault_member)
                if fault_member is not None
                else None
            )
            if injected_termination is not None:
                _validate_worker_crash_termination(
                    injected_termination,
                    old[fault_member],
                )
            if (
                fault_member is not None
                and fault_member not in guard._proofs
                and injected_termination is None
            ):
                fault_identity = old[fault_member]
                attempt_sequence = _append_fault_injection_attempt(
                    run_state,
                    run_state_path,
                    identity=fault_identity,
                )
                try:
                    injection_result = self.backend.inject_process_fault(
                        release, namespace, fault_identity
                    )
                except Exception as exc:
                    _append_fault_injection_incomplete(
                        run_state,
                        run_state_path,
                        attempt_sequence=attempt_sequence,
                        bundle=_injection_command_bundle(error=exc),
                        error=exc,
                    )
                    _capture_ambiguous_injection_evidence(
                        self.backend,
                        run_state,
                        run_state_path,
                        release=release,
                        namespace=namespace,
                        watch=watches[fault_member],
                        expected=fault_identity,
                    )
                    raise
                try:
                    process_selection, injection_bundle = (
                        _validate_fault_injection_result(
                            injection_result,
                            fault_identity,
                        )
                    )
                except Exception as exc:
                    _append_fault_injection_incomplete(
                        run_state,
                        run_state_path,
                        attempt_sequence=attempt_sequence,
                        bundle=_injection_command_bundle(
                            injection_result
                        ),
                        error=exc,
                    )
                    _capture_ambiguous_injection_evidence(
                        self.backend,
                        run_state,
                        run_state_path,
                        release=release,
                        namespace=namespace,
                        watch=watches[fault_member],
                        expected=fault_identity,
                    )
                    raise
                _append_fault_injection_result(
                    run_state,
                    run_state_path,
                    attempt_sequence=attempt_sequence,
                    selection=process_selection,
                    bundle=injection_bundle,
                )
                run_state["fault"]["process_selection"] = {
                    **process_selection,
                    "command": injection_bundle["command"],
                    "exec_return_code":
                        injection_bundle["exec_return_code"],
                }
                _atomic_write_json(run_state_path, run_state)
                observe_injected = getattr(
                    self.backend,
                    "observe_injected_process_termination",
                    None,
                )
                if observe_injected is None:
                    observe_injected = (
                        self.backend.wait_injected_process_termination
                    )
                observe_options = _raw_observation_sink_options(
                    self.backend,
                    run_state,
                    run_state_path,
                    stage="worker_crash_injection",
                )
                injected_observation = _observe_or_persist_incomplete(
                    run_state,
                    run_state_path,
                    stage="worker_crash_injection",
                    observer=lambda: observe_injected(
                        release,
                        namespace,
                        watches[fault_member],
                        old[fault_member],
                        **observe_options,
                    ),
                )
                (
                    injected_termination,
                    injected_digest,
                ) = _persist_fault_termination_observation(
                    run_state,
                    run_state_path,
                    stage="worker_crash_injection",
                    observation=injected_observation,
                )
                _validate_and_record_semantics(
                    run_state,
                    run_state_path,
                    stage="worker_crash_injection",
                    proof_sha256=injected_digest,
                    validator=lambda: _validate_worker_crash_termination(
                        injected_termination,
                        old[fault_member],
                    ),
                )
                _record_protocol_event(
                    run_state,
                    name="injected_process_exited",
                    evidence={
                        "member": injected_termination.member,
                        "pod_uid": injected_termination.pod_uid,
                        "node_uid": injected_termination.node_uid,
                        "container_id": injected_termination.container_id,
                        "process_generation":
                            injected_termination.process_generation,
                        "exit_code": injected_termination.exit_code,
                        "termination_reason":
                            injected_termination.termination_reason,
                        "termination_message":
                            injected_termination.termination_message,
                        "finished_at": injected_termination.finished_at,
                    },
                )
                _atomic_write_json(run_state_path, run_state)
            natural_termination = (
                _validated_journal_proofs(
                    run_state,
                    stage="nccl_watchdog",
                ).get(watchdog_member)
                if watchdog_member is not None
                else None
            )
            if natural_termination is not None:
                _validate_watchdog_termination(
                    natural_termination,
                    old[watchdog_member],
                    expected_generation=old_generation,
                    expected_operation_ids=tuple(sorted(expected_ids)),
                )
            if (
                watchdog_member is not None
                and watchdog_member not in guard._proofs
                and natural_termination is None
            ):
                observe_watchdog = getattr(
                    self.backend,
                    "observe_natural_watchdog_termination",
                    None,
                )
                if observe_watchdog is None:
                    observe_watchdog = (
                        self.backend.wait_natural_watchdog_termination
                    )
                observe_options = _raw_observation_sink_options(
                    self.backend,
                    run_state,
                    run_state_path,
                    stage="nccl_watchdog",
                )
                natural_observation = _observe_or_persist_incomplete(
                    run_state,
                    run_state_path,
                    stage="nccl_watchdog",
                    observer=lambda: observe_watchdog(
                        release,
                        namespace,
                        watches[watchdog_member],
                        old[watchdog_member],
                        expected_generation=old_generation,
                        expected_operation_ids=tuple(sorted(expected_ids)),
                        **observe_options,
                    ),
                )
                (
                    natural_termination,
                    natural_digest,
                ) = _persist_fault_termination_observation(
                    run_state,
                    run_state_path,
                    stage="nccl_watchdog",
                    observation=natural_observation,
                )
                _validate_and_record_semantics(
                    run_state,
                    run_state_path,
                    stage="nccl_watchdog",
                    proof_sha256=natural_digest,
                    validator=lambda: _validate_watchdog_termination(
                        natural_termination,
                        old[watchdog_member],
                        expected_generation=old_generation,
                        expected_operation_ids=tuple(sorted(expected_ids)),
                    ),
                )
                _record_protocol_event(
                    run_state,
                    name="watchdog_process_exited",
                    evidence={
                        "member": natural_termination.member,
                        "pod_uid": natural_termination.pod_uid,
                        "node_uid": natural_termination.node_uid,
                        "container_id": natural_termination.container_id,
                        "process_generation": natural_termination.process_generation,
                        "exit_code": natural_termination.exit_code,
                        "termination_reason": natural_termination.termination_reason,
                        "termination_message": natural_termination.termination_message,
                        "finished_at": natural_termination.finished_at,
                    },
                )
                _atomic_write_json(run_state_path, run_state)
            first_member = fault_member or watchdog_member
            stop_order = (
                (first_member,) + tuple(
                    member for member in MEMBERS if member != first_member
                )
                if first_member is not None else MEMBERS
            )
            if watchdog_member is not None:
                _record_protocol_event(
                    run_state,
                    name="whole_world_scale_down_started",
                    evidence={"members": list(stop_order)},
                )
                _atomic_write_json(run_state_path, run_state)
            for member in stop_order:
                self.backend.stop_member(release, namespace, member)
            for member in MEMBERS:
                if member in guard._proofs:
                    continue
                wait_kwargs = {}
                if member == fault_member:
                    wait_kwargs["prior_termination"] = injected_termination
                elif member == watchdog_member:
                    wait_kwargs["prior_termination"] = natural_termination
                observe_termination = getattr(
                    self.backend,
                    "observe_termination",
                    None,
                )
                if observe_termination is None:
                    observe_termination = self.backend.wait_termination
                observe_options = _raw_observation_sink_options(
                    self.backend,
                    run_state,
                    run_state_path,
                    stage="whole_world_termination",
                )
                termination_observation = _observe_or_persist_incomplete(
                    run_state,
                    run_state_path,
                    stage="whole_world_termination",
                    observer=lambda member=member, wait_kwargs=wait_kwargs: (
                        observe_termination(
                            release,
                            namespace,
                            watches[member],
                            old[member],
                            **wait_kwargs,
                            **observe_options,
                        )
                    ),
                )
                observed = _as_termination_observation(
                    termination_observation
                )
                termination_digest = _append_observation(
                    run_state,
                    run_state_path,
                    stage="whole_world_termination",
                    observation=observed,
                )
                _validate_and_record_semantics(
                    run_state,
                    run_state_path,
                    stage="whole_world_termination",
                    proof_sha256=termination_digest,
                    validator=lambda proof=observed.proof: (
                        (
                            _validate_durable_termination_record(
                                run_state, proof
                            )
                            if getattr(
                                self.backend,
                                "raw_observation_sink_supported",
                                False,
                            )
                            else _termination_authority_record(proof)
                        ),
                        guard.record_termination(proof)
                    ),
                )
                _checkpoint_termination_proof(
                    run_state,
                    run_state_path,
                    observed.proof,
                )
            if watchdog_member is not None:
                _record_protocol_event(
                    run_state,
                    name="four_old_processes_terminated",
                    evidence=[
                        {
                            "member": proof.member,
                            "pod_uid": proof.pod_uid,
                            "node_uid": proof.node_uid,
                            "container_id": proof.container_id,
                            "process_generation": proof.process_generation,
                            "exit_code": proof.exit_code,
                            "container_terminated": proof.container_terminated,
                            "pod_deleted": proof.pod_deleted,
                            "termination_reason": proof.termination_reason,
                            "termination_message": proof.termination_message,
                            "finished_at": proof.finished_at,
                        }
                        for proof in (guard._proofs[name] for name in MEMBERS)
                    ],
                )
                _atomic_write_json(run_state_path, run_state)
            if (
                fault_member is not None
                and guard._proofs[fault_member].exit_code != 137
            ):
                raise RuntimeError(
                    "worker_crash proof requires the injected container to exit 137"
                )
            if not guard.has_four_termination_proofs:
                raise RuntimeError(
                    "four durable termination proofs are required before resume"
                )
            run_state["checkpoint"] = "TERMINATED"
            _atomic_write_json(run_state_path, run_state)
            checkpoint = "TERMINATED"
        finally:
            close_watch = getattr(self.backend, "close_termination_watch", None)
            if close_watch is not None:
                for watch in watches.values():
                    close_watch(watch)
        return self._continue_after_termination(
            release=release,
            namespace=namespace,
            chart=chart,
            old_generation=old_generation,
            new_generation=new_generation,
            run_state=run_state,
            run_state_path=run_state_path,
            old=old,
            guard=guard,
            watchdog_member=watchdog_member,
        )

    def _continue_after_termination(
        self,
        *,
        release: str,
        namespace: str,
        chart: str,
        old_generation: str,
        new_generation: str,
        run_state: dict[str, object],
        run_state_path: Path,
        old: dict[str, ProcessIdentity],
        guard: WorldRestartGuard,
        watchdog_member: str | None,
    ) -> dict[str, object]:
        checkpoint = _restart_checkpoint(run_state)
        entry_checkpoint = checkpoint
        if checkpoint not in {
            "TERMINATED",
            "GENERATION_PATCHED",
            "STARTED",
            "PERMIT_READY",
            "PERMIT_PUBLISHED",
        }:
            raise RuntimeError(
                "restart continuation checkpoint is not resumable"
            )
        if not guard.has_four_termination_proofs:
            raise RuntimeError(
                "four durable termination proofs are required before scale-up"
            )

        if checkpoint == "TERMINATED":
            self.backend.patch_generation(
                release, namespace, chart, new_generation
            )
            run_state["checkpoint"] = "GENERATION_PATCHED"
            _atomic_write_json(run_state_path, run_state)
            checkpoint = "GENERATION_PATCHED"

        if checkpoint == "GENERATION_PATCHED":
            for member in MEMBERS:
                self.backend.start_member(release, namespace, member)
            run_state["checkpoint"] = "STARTED"
            _atomic_write_json(run_state_path, run_state)
            checkpoint = "STARTED"

        if checkpoint == "STARTED":
            fresh_pod_uids = self.backend.wait_fresh_pod_uids(
                release,
                namespace,
                generation=new_generation,
                old_pod_uids={
                    member: old[member].pod_uid for member in MEMBERS
                },
            )
            startup_permit = build_startup_permit(
                topology_generation=new_generation,
                members=fresh_pod_uids,
                issuance_mode="RESTART",
            )
            run_state["fresh_pod_uids"] = dict(fresh_pod_uids)
            run_state["startup_permit"] = startup_permit
            run_state["checkpoint"] = "PERMIT_READY"
            _atomic_write_json(run_state_path, run_state)
            checkpoint = "PERMIT_READY"
        else:
            fresh_pod_uids = run_state.get("fresh_pod_uids")
            if (
                not isinstance(fresh_pod_uids, dict)
                or set(fresh_pod_uids) != set(MEMBERS)
                or any(
                    not isinstance(value, str) or not value
                    for value in fresh_pod_uids.values()
                )
                or len(set(fresh_pod_uids.values())) != len(MEMBERS)
            ):
                raise RuntimeError(
                    "durable fresh Pod UID checkpoint is invalid"
                )
            startup_permit = validate_startup_permit(
                run_state.get("startup_permit"),
                expected_generation=new_generation,
                expected_mode="RESTART",
            )
            if startup_permit["members"] != fresh_pod_uids:
                raise RuntimeError(
                    "startup permit changed fresh Pod identity checkpoint"
                )

        if checkpoint == "PERMIT_READY":
            if entry_checkpoint == "PERMIT_READY":
                observed_fresh_pod_uids = self.backend.wait_fresh_pod_uids(
                    release,
                    namespace,
                    generation=new_generation,
                    old_pod_uids={
                        member: old[member].pod_uid for member in MEMBERS
                    },
                )
                if observed_fresh_pod_uids != fresh_pod_uids:
                    raise RuntimeError(
                        "fresh Pod UIDs changed after startup permit intent"
                    )
            published_permit = self.backend.publish_startup_permit(
                release, namespace, startup_permit
            )
            if validate_startup_permit(
                published_permit,
                expected_generation=new_generation,
                expected_mode="RESTART",
            ) != startup_permit:
                raise RuntimeError(
                    "published startup permit changed exact payload"
                )
            run_state["checkpoint"] = "PERMIT_PUBLISHED"
            _atomic_write_json(run_state_path, run_state)
            checkpoint = "PERMIT_PUBLISHED"

        evidence = self.backend.wait_acceptance_evidence(
            old_generation=old_generation,
            new_generation=new_generation,
            restart_run_id=str(run_state["restart_run_id"]),
            termination_proofs=tuple(
                guard._proofs[name] for name in MEMBERS
            ),
        )
        canonical_terminations = [
            _termination_authority_record(guard._proofs[name])
            for name in MEMBERS
        ]
        returned_terminations = evidence.get("termination_records")
        if (
            returned_terminations is not None
            and returned_terminations != canonical_terminations
        ):
            raise RuntimeError(
                "acceptance evidence changed canonical termination records"
            )
        evidence["termination_records"] = canonical_terminations
        validate_termination_records(
            canonical_terminations,
            expected_generation=old_generation,
            expected_members=MEMBERS,
        )
        if evidence.get("new_topology_generation") != new_generation:
            raise RuntimeError(
                "fresh evidence does not bind requested generation"
            )
        if evidence.get("restart_run_id") != run_state["restart_run_id"]:
            raise RuntimeError(
                "fresh evidence changed the durable restart run id"
            )
        _validate_permitted_identities(evidence, startup_permit)
        if watchdog_member is not None:
            _record_protocol_event(
                run_state,
                name="four_fresh_reports_observed",
                evidence=evidence.get("resource_reports"),
            )
            _record_protocol_event(
                run_state,
                name="topology_accept_started",
                evidence=evidence,
            )
        run_state["phase"] = "EVIDENCE_READY"
        run_state["checkpoint"] = "EVIDENCE_READY"
        run_state["evidence"] = evidence
        _atomic_write_json(run_state_path, run_state)
        accepted = self.backend.accept_topology(evidence)
        if accepted.get("accepted") is not True:
            raise RuntimeError("Gateway rejected replacement topology")
        if watchdog_member is not None:
            _record_protocol_event(
                run_state,
                name="topology_accept_succeeded",
                evidence={"request": evidence, "response": accepted},
            )
        persist = getattr(self.backend, "persist_started_world", None)
        if persist is not None:
            persist(
                release,
                namespace,
                chart,
                new_generation,
                startup_permit,
            )
        run_state["phase"] = "ACCEPTED"
        run_state["checkpoint"] = "ACCEPTED"
        run_state["accepted_response"] = accepted
        _atomic_write_json(run_state_path, run_state)
        return accepted


class KubectlGatewayBackend:
    """Real kubectl/Helm/Gateway backend; it never force-deletes Pods."""

    raw_observation_sink_supported = True

    def __init__(
        self, gateway_url: str, *, timeout_s: float = 300.0,
        context: str = "", gateway_read_timeout_s: float | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("pd-worldctl timeout must be positive")
        if gateway_read_timeout_s is not None and (
            gateway_read_timeout_s <= 0 or gateway_read_timeout_s > timeout_s
        ):
            raise ValueError(
                "Gateway read timeout must be positive and no greater than "
                "the pd-worldctl timeout"
            )
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout_s = timeout_s
        self.gateway_read_timeout_s = (
            timeout_s if gateway_read_timeout_s is None
            else gateway_read_timeout_s
        )
        self.context = context
        self._old_operation_ids: set[str] = set()
        self._old_resource_ids: set[str] = set()

    def _kubectl_args(self, *args: str) -> list[str]:
        value = ["kubectl"]
        if self.context:
            value.extend(["--context", self.context])
        return [*value, *args]

    def _helm_args(self, *args: str) -> list[str]:
        value = ["helm"]
        if self.context:
            value.extend(["--kube-context", self.context])
        return [*value, *args]

    def _run_mutation(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Keep operational command output off the CLI JSON stdout channel."""

        kwargs: dict[str, object] = {
            "check": True,
            "stdout": sys.stderr,
        }
        if input_text is not None:
            kwargs.update({"input": input_text, "text": True})
        return subprocess.run(command, **kwargs)

    def _helm_json(self, *args: str) -> object:
        completed = subprocess.run(
            self._helm_args(*args),
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def _helm_history(
        self, release: str, namespace: str,
    ) -> list[dict[str, object]]:
        value = self._helm_json(
            "history", release, "--namespace", namespace, "--output", "json",
        )
        if not isinstance(value, list) or not value:
            raise RuntimeError("Helm release history is unavailable")
        history: list[dict[str, object]] = []
        seen_revisions: set[int] = set()
        for raw in value:
            if not isinstance(raw, dict):
                raise RuntimeError("Helm release history entry is invalid")
            try:
                revision = int(raw.get("revision"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Helm release history revision is invalid"
                ) from exc
            status = raw.get("status")
            if (
                revision <= 0
                or revision in seen_revisions
                or not isinstance(status, str)
                or not status
            ):
                raise RuntimeError("Helm release history entry is invalid")
            seen_revisions.add(revision)
            history.append({"revision": revision, "status": status.lower()})
        history.sort(key=lambda item: int(item["revision"]))
        return history

    def _helm_release_values(
        self, release: str, namespace: str, revision: int,
    ) -> dict[str, object]:
        value = self._helm_json(
            "get", "values", release,
            "--namespace", namespace,
            "--revision", str(revision),
            "--output", "json",
        )
        if not isinstance(value, dict):
            raise RuntimeError("Helm release values are invalid")
        return value

    @staticmethod
    def _apply_helm_overrides(
        base: dict[str, object],
        overrides: tuple[tuple[tuple[str, ...], object], ...],
    ) -> dict[str, object]:
        value = copy.deepcopy(base)
        for path, replacement in overrides:
            if not path:
                raise RuntimeError("Helm override path is empty")
            cursor = value
            for field in path[:-1]:
                child = cursor.get(field)
                if child is None:
                    child = {}
                    cursor[field] = child
                if not isinstance(child, dict):
                    raise RuntimeError(
                        "Helm override path conflicts with existing values"
                    )
                cursor = child
            cursor[path[-1]] = copy.deepcopy(replacement)
        return value

    def _helm_deployed_snapshot(
        self, release: str, namespace: str,
    ) -> _HelmReleaseSnapshot:
        history = self._helm_history(release, namespace)
        latest = history[-1]
        revision = int(latest["revision"])
        status = str(latest["status"])
        if status != "deployed":
            raise RuntimeError(
                "Helm upgrade requires an exact deployed pre-attempt revision; "
                f"observed revision={revision} status={status}"
            )
        return _HelmReleaseSnapshot(
            revision=revision,
            values=self._helm_release_values(release, namespace, revision),
        )

    @staticmethod
    def _helm_values_equal(expected: object, observed: object) -> bool:

        return _canonical_json(expected) == _canonical_json(observed)

    @staticmethod
    def _helm_value_diff_paths(
        expected: object,
        observed: object,
        *,
        limit: int = 64,
    ) -> tuple[list[str], bool]:

        paths: list[str] = []
        truncated = False
        missing = object()

        def append(path: str) -> None:
            nonlocal truncated
            if len(paths) >= limit:
                truncated = True
                return
            paths.append(path or "/")

        def child(path: str, field: object) -> str:
            escaped = str(field).replace("~", "~0").replace("/", "~1")
            return f"{path}/{escaped}"

        def walk(left: object, right: object, path: str) -> None:
            nonlocal truncated
            if len(paths) >= limit:
                truncated = True
                return
            if isinstance(left, dict) and isinstance(right, dict):
                keys = sorted(set(left) | set(right), key=str)
                for key in keys:
                    left_value = left.get(key, missing)
                    right_value = right.get(key, missing)
                    next_path = child(path, key)
                    if left_value is missing or right_value is missing:
                        append(next_path)
                    else:
                        walk(left_value, right_value, next_path)
                return
            if isinstance(left, list) and isinstance(right, list):
                for index in range(max(len(left), len(right))):
                    next_path = child(path, index)
                    if index >= len(left) or index >= len(right):
                        append(next_path)
                    else:
                        walk(left[index], right[index], next_path)
                return
            if type(left) is type(right) and left == right:
                return
            append(path)

        walk(expected, observed, "")
        return paths, truncated

    def _emit_helm_reconcile_diagnostic(
        self,
        *,
        release: str,
        namespace: str,
        phase: str,
        outcome: str,
        snapshot: _HelmReleaseSnapshot,
        expected: dict[str, object],
        history: list[dict[str, object]],
        comparison: dict[str, object] | None,
        observed: dict[str, object] | None,
        predecessor_observed: dict[str, object] | None,
        error: Exception | None,
    ) -> None:
        latest = history[-1]
        event: dict[str, object] = {
            "event": "pd_worldctl.helm_upgrade_reconcile",
            "expected_values_sha256": _canonical_sha256(expected),
            "history_tail": history[-3:],
            "latest_revision": int(latest["revision"]),
            "latest_status": str(latest["status"]),
            "namespace": namespace,
            "outcome": outcome,
            "phase": phase,
            "pre_revision": snapshot.revision,
            "pre_values_sha256": _canonical_sha256(snapshot.values),
            "release": release,
        }
        if error is not None:
            event["error_type"] = type(error).__name__
            returncode = getattr(error, "returncode", None)
            if isinstance(returncode, int):
                event["error_returncode"] = returncode
        if predecessor_observed is not None:
            predecessor_paths, predecessor_truncated = (
                self._helm_value_diff_paths(
                    snapshot.values, predecessor_observed,
                )
            )
            event.update({
                "predecessor_diff_paths": predecessor_paths,
                "predecessor_diff_truncated": predecessor_truncated,
                "predecessor_values_sha256": _canonical_sha256(
                    predecessor_observed
                ),
            })
        if comparison is not None and observed is not None:
            paths, truncated = self._helm_value_diff_paths(
                comparison, observed,
            )
            event.update({
                "observed_values_sha256": _canonical_sha256(observed),
                "value_diff_paths": paths,
                "value_diff_truncated": truncated,
            })
        print(_canonical_json(event), file=sys.stderr, flush=True)

    def _inspect_helm_upgrade_result(
        self,
        *,
        release: str,
        namespace: str,
        phase: str,
        snapshot: _HelmReleaseSnapshot,
        expected: dict[str, object],
        error: Exception | None,
    ) -> str:
        history = self._helm_history(release, namespace)
        latest = history[-1]
        latest_revision = int(latest["revision"])
        latest_status = str(latest["status"])

        if latest_revision == snapshot.revision:
            observed = self._helm_release_values(
                release, namespace, latest_revision,
            )
            unchanged = (
                latest_status == "deployed"
                and self._helm_values_equal(observed, snapshot.values)
            )
            self._emit_helm_reconcile_diagnostic(
                release=release,
                namespace=namespace,
                phase=phase,
                outcome="history_unchanged" if unchanged else "invalid",
                snapshot=snapshot,
                expected=expected,
                history=history,
                comparison=snapshot.values,
                observed=observed,
                predecessor_observed=None,
                error=error,
            )
            if unchanged:
                return "unchanged"
            raise RuntimeError(
                "pre-attempt Helm revision status/values changed"
            ) from error

        exact_new_revision = latest_revision == snapshot.revision + 1
        exact_predecessor = (
            len(history) >= 2
            and int(history[-2]["revision"]) == snapshot.revision
        )
        if not exact_new_revision or not exact_predecessor:
            self._emit_helm_reconcile_diagnostic(
                release=release,
                namespace=namespace,
                phase=phase,
                outcome="invalid",
                snapshot=snapshot,
                expected=expected,
                history=history,
                comparison=None,
                observed=None,
                predecessor_observed=None,
                error=error,
            )
            raise RuntimeError(
                "Helm response loss did not produce the exact next revision"
            ) from error

        predecessor = history[-2]
        predecessor_status = str(predecessor["status"])
        predecessor_observed = self._helm_release_values(
            release, namespace, snapshot.revision,
        )
        observed = self._helm_release_values(
            release, namespace, latest_revision,
        )
        if latest_status == "deployed":
            valid = (
                predecessor_status == "superseded"
                and self._helm_values_equal(
                    predecessor_observed, snapshot.values,
                )
                and self._helm_values_equal(observed, expected)
            )
            outcome = "new_deployed" if valid else "invalid"
        elif latest_status in {"pending-upgrade", "failed"}:
            valid = (
                predecessor_status == "deployed"
                and self._helm_values_equal(
                    predecessor_observed, snapshot.values,
                )
                and self._helm_values_equal(observed, expected)
            )
            if not valid:
                outcome = "invalid"
            elif latest_status == "pending-upgrade":
                outcome = "new_pending"
            else:
                outcome = "new_failed"
        else:
            valid = False
            outcome = "invalid"
        self._emit_helm_reconcile_diagnostic(
            release=release,
            namespace=namespace,
            phase=phase,
            outcome=outcome,
            snapshot=snapshot,
            expected=expected,
            history=history,
            comparison=expected,
            observed=observed,
            predecessor_observed=predecessor_observed,
            error=error,
        )
        if not valid:
            raise RuntimeError(
                "new Helm revision status/predecessor/values are invalid"
            ) from error
        if latest_status == "deployed":
            return "deployed"
        if latest_status == "pending-upgrade":
            return "pending"
        return "failed"

    def _delete_exact_helm_revision(
        self,
        *,
        release: str,
        namespace: str,
        revision: int,
        expected_status: str,
    ) -> None:
        if expected_status not in {"pending-upgrade", "failed"}:
            raise RuntimeError("Helm recovery status is not deletable")
        name = f"sh.helm.release.v1.{release}.v{revision}"
        secret = self._kubectl_json(namespace, "get", "secret", name)
        metadata = secret.get("metadata") if isinstance(secret, dict) else None
        if not isinstance(metadata, dict):
            raise RuntimeError("recoverable Helm metadata Secret is invalid")
        labels = metadata.get("labels")
        uid = metadata.get("uid")
        resource_version = metadata.get("resourceVersion")
        if (
            metadata.get("name") != name
            or not isinstance(labels, dict)
            or labels.get("owner") != "helm"
            or labels.get("name") != release
            or labels.get("version") != str(revision)
            or labels.get("status") != expected_status
            or not isinstance(uid, str)
            or not uid
            or not isinstance(resource_version, str)
            or not resource_version
        ):
            raise RuntimeError(
                "recoverable Helm metadata Secret identity changed"
            )
        delete_options = _canonical_json({
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {
                "uid": uid,
                "resourceVersion": resource_version,
            },
        })
        uri = (
            f"/api/v1/namespaces/{quote(namespace, safe='')}/secrets/"
            f"{quote(name, safe='')}"
        )
        event = (
            "pd_worldctl.helm_pending_revision_delete"
            if expected_status == "pending-upgrade"
            else "pd_worldctl.helm_failed_revision_delete"
        )
        print(_canonical_json({
            "event": event,
            "namespace": namespace,
            "release": release,
            "resourceVersion": resource_version,
            "revision": revision,
            "uid": uid,
        }), file=sys.stderr, flush=True)
        self._run_mutation(
            self._kubectl_args("delete", "--raw", uri, "-f", "-"),
            input_text=delete_options,
        )

    def _delete_exact_pending_helm_revision(
        self,
        *,
        release: str,
        namespace: str,
        revision: int,
    ) -> None:
        self._delete_exact_helm_revision(
            release=release,
            namespace=namespace,
            revision=revision,
            expected_status="pending-upgrade",
        )

    def _delete_exact_failed_helm_revision(
        self,
        *,
        release: str,
        namespace: str,
        revision: int,
    ) -> None:
        self._delete_exact_helm_revision(
            release=release,
            namespace=namespace,
            revision=revision,
            expected_status="failed",
        )

    def _reconcile_helm_upgrade_response_loss(
        self,
        *,
        release: str,
        namespace: str,
        command: list[str],
        snapshot: _HelmReleaseSnapshot,
        expected: dict[str, object],
        original_error: Exception,
    ) -> None:
        outcome = self._inspect_helm_upgrade_result(
            release=release,
            namespace=namespace,
            phase="after_initial_failure",
            snapshot=snapshot,
            expected=expected,
            error=original_error,
        )
        if outcome == "deployed":
            return
        if outcome == "pending":
            self._delete_exact_pending_helm_revision(
                release=release,
                namespace=namespace,
                revision=snapshot.revision + 1,
            )
        elif outcome == "failed":
            self._delete_exact_failed_helm_revision(
                release=release,
                namespace=namespace,
                revision=snapshot.revision + 1,
            )

        retry_error: Exception | None = None
        try:
            self._run_mutation(command)
        except Exception as exc:
            retry_error = exc
        retry_outcome = self._inspect_helm_upgrade_result(
            release=release,
            namespace=namespace,
            phase="after_bounded_retry",
            snapshot=snapshot,
            expected=expected,
            error=retry_error,
        )
        if retry_outcome != "deployed":
            raise RuntimeError(
                "bounded Helm retry did not reach the exact deployed revision"
            ) from (retry_error or original_error)

    def _run_helm_upgrade(
        self,
        *,
        release: str,
        namespace: str,
        command: list[str],
        overrides: tuple[tuple[tuple[str, ...], object], ...],
    ) -> None:
        snapshot = self._helm_deployed_snapshot(release, namespace)
        expected = self._apply_helm_overrides(snapshot.values, overrides)
        try:
            self._run_mutation(command)
        except Exception as exc:
            self._reconcile_helm_upgrade_response_loss(
                release=release,
                namespace=namespace,
                command=command,
                snapshot=snapshot,
                expected=expected,
                original_error=exc,
            )
            return
        outcome = self._inspect_helm_upgrade_result(
            release=release,
            namespace=namespace,
            phase="after_success",
            snapshot=snapshot,
            expected=expected,
            error=None,
        )
        if outcome != "deployed":
            raise RuntimeError(
                "successful Helm command did not produce the exact deployed "
                "revision"
            )

    @staticmethod
    def _json_url_with_timeout(
        url: str, *, body: dict[str, object] | None = None,
        timeout: float,
    ):
        data = None if body is None else json.dumps(body).encode()
        request = Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())

    @staticmethod
    def _json_url(url: str, *, body: dict[str, object] | None = None):
        return KubectlGatewayBackend._json_url_with_timeout(
            url, body=body, timeout=30,
        )

    @staticmethod
    def _gateway_read_retryable(error: BaseException) -> bool:
        if isinstance(error, HTTPError):
            return error.code in {408, 425, 429, 500, 502, 503, 504}
        return isinstance(error, (
            ConnectionError,
            TimeoutError,
            URLError,
            HTTPException,
            json.JSONDecodeError,
            UnicodeError,
        ))

    def _read_gateway_json(self, url: str) -> dict[str, object]:
        """Retry only a pre-mutation Gateway GET within the reset budget."""

        deadline = time.monotonic() + self.gateway_read_timeout_s
        attempts = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "timed out reading Gateway topology before world mutation"
                )
            try:
                value = self._json_url_with_timeout(
                    url, timeout=min(30.0, remaining)
                )
                if not isinstance(value, dict):
                    raise RuntimeError("Gateway read did not return a JSON object")
                if attempts:
                    print(_canonical_json({
                        "event": "pd_worldctl.gateway_read_recovered",
                        "retry_attempts": attempts,
                    }), file=sys.stderr, flush=True)
                return value
            except Exception as exc:
                if not self._gateway_read_retryable(exc):
                    raise
                attempts += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out reading Gateway topology before world mutation"
                    ) from exc
                if attempts == 1:
                    print(_canonical_json({
                        "event": "pd_worldctl.gateway_read_retry",
                        "error_type": type(exc).__name__,
                    }), file=sys.stderr, flush=True)
                time.sleep(min(0.5, remaining))

    def _kubectl_json(self, namespace: str, *args: str):
        completed = subprocess.run(
            self._kubectl_args("-n", namespace, *args, "-o", "json"),
            check=True, capture_output=True, text=True,
        )
        return json.loads(completed.stdout)

    def read_initialization_snapshot(
        self, release: str, namespace: str
    ) -> InitializationSnapshot:
        config_map = self._kubectl_json(
            namespace,
            "get",
            "configmap",
            f"{release}-prism-serve-worker-topology",
        )
        metadata = config_map.get("metadata")
        data = config_map.get("data")
        if not isinstance(metadata, dict) or not isinstance(data, dict):
            raise RuntimeError(
                "worker topology ConfigMap metadata/data is unavailable"
            )
        configmap_uid = metadata.get("uid")
        resource_version = metadata.get("resourceVersion")
        if (
            not isinstance(configmap_uid, str)
            or not configmap_uid
            or not isinstance(resource_version, str)
            or not resource_version
        ):
            raise RuntimeError(
                "worker topology ConfigMap UID/resourceVersion is unavailable"
            )
        state_raw = data.get(INITIALIZATION_STATE_KEY)
        if not isinstance(state_raw, str):
            raise RuntimeError(
                f"{INITIALIZATION_STATE_KEY} is absent from ConfigMap"
            )
        try:
            state = validate_initialization_state(json.loads(state_raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                "worker topology initialization state is invalid"
            ) from exc
        permit_raw = data.get(STARTUP_PERMIT_KEY)
        permit = None
        if permit_raw is not None:
            if not isinstance(permit_raw, str):
                raise RuntimeError("startup permit ConfigMap value is invalid")
            try:
                permit = validate_startup_permit(json.loads(permit_raw))
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    "startup permit ConfigMap JSON is invalid"
                ) from exc
        return InitializationSnapshot(
            configmap_uid=configmap_uid,
            resource_version=resource_version,
            state_raw=state_raw,
            state=state,
            startup_permit_raw=permit_raw,
            startup_permit=permit,
        )

    def _patch_initialization_configmap(
        self,
        release: str,
        namespace: str,
        snapshot: InitializationSnapshot,
        operations: list[dict[str, object]],
    ) -> None:
        patch = [
            {
                "op": "test",
                "path": "/metadata/uid",
                "value": snapshot.configmap_uid,
            },
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": snapshot.resource_version,
            },
            {
                "op": "test",
                "path": f"/data/{INITIALIZATION_STATE_KEY}",
                "value": snapshot.state_raw,
            },
            *operations,
        ]
        self._run_mutation(
            self._kubectl_args(
                "-n",
                namespace,
                "patch",
                "configmap",
                f"{release}-prism-serve-worker-topology",
                "--type",
                "json",
                "--patch",
                _canonical_json(patch),
            ),
        )

    def claim_initialization(
        self,
        release: str,
        namespace: str,
        snapshot: InitializationSnapshot,
        *,
        initialize_run_id: str,
        permit_id: str,
        generation: str,
    ) -> InitializationSnapshot:
        state = validate_initialization_state(snapshot.state)
        if state["phase"] != "UNINITIALIZED":
            raise RuntimeError("initialization claim requires UNINITIALIZED")
        if snapshot.startup_permit is not None:
            raise RuntimeError(
                "UNINITIALIZED topology already contains a startup permit"
            )
        desired = _initializing_state(
            initialize_run_id=initialize_run_id,
            permit_id=permit_id,
            topology_generation=generation,
        )
        desired_raw = _canonical_json(desired)
        try:
            self._patch_initialization_configmap(
                release,
                namespace,
                snapshot,
                [{
                    "op": "replace",
                    "path": f"/data/{INITIALIZATION_STATE_KEY}",
                    "value": desired_raw,
                }],
            )
        except Exception as exc:
            observed = self.read_initialization_snapshot(release, namespace)
            if (
                observed.configmap_uid != snapshot.configmap_uid
                or observed.state_raw != desired_raw
                or observed.state != desired
                or observed.startup_permit is not None
            ):
                raise RuntimeError(
                    "UNINITIALIZED to INITIALIZING CAS was not committed"
                ) from exc
            return observed
        observed = self.read_initialization_snapshot(release, namespace)
        if (
            observed.configmap_uid != snapshot.configmap_uid
            or observed.state_raw != desired_raw
            or observed.state != desired
            or observed.startup_permit is not None
        ):
            raise RuntimeError(
                "initialization claim read-back changed exact identity"
            )
        return observed

    def publish_initialization_permit(
        self,
        release: str,
        namespace: str,
        snapshot: InitializationSnapshot,
        *,
        startup_permit: dict[str, object],
        initialize_run_id: str,
        permit_id: str,
        generation: str,
    ) -> InitializationSnapshot:
        _require_initialization_identity(
            snapshot.state,
            initialize_run_id=initialize_run_id,
            permit_id=permit_id,
            topology_generation=generation,
            phases=("INITIALIZING",),
        )
        permit = validate_startup_permit(
            startup_permit,
            expected_generation=generation,
            expected_mode="INIT",
        )
        if permit["permit_id"] != permit_id:
            raise RuntimeError("INIT permit does not bind initialization ID")
        canonical = _canonical_json(permit)
        if snapshot.startup_permit is not None:
            if (
                snapshot.startup_permit != permit
                or snapshot.startup_permit_raw != canonical
            ):
                raise RuntimeError(
                    "initialization contains a foreign startup permit"
                )
            return snapshot
        try:
            self._patch_initialization_configmap(
                release,
                namespace,
                snapshot,
                [{
                    "op": "add",
                    "path": f"/data/{STARTUP_PERMIT_KEY}",
                    "value": canonical,
                }],
            )
        except Exception as exc:
            observed = self.read_initialization_snapshot(release, namespace)
            if (
                observed.configmap_uid != snapshot.configmap_uid
                or observed.state != snapshot.state
                or observed.startup_permit_raw != canonical
                or observed.startup_permit != permit
            ):
                raise RuntimeError(
                    "response loss did not read back the exact INIT permit"
                ) from exc
            return observed
        observed = self.read_initialization_snapshot(release, namespace)
        if (
            observed.configmap_uid != snapshot.configmap_uid
            or observed.state != snapshot.state
            or observed.startup_permit_raw != canonical
            or observed.startup_permit != permit
        ):
            raise RuntimeError("published INIT permit changed exact payload")
        return observed

    def accept_initialization(
        self,
        release: str,
        namespace: str,
        snapshot: InitializationSnapshot,
        *,
        initialize_run_id: str,
        permit_id: str,
        generation: str,
    ) -> dict[str, object]:
        initializing = _require_initialization_identity(
            snapshot.state,
            initialize_run_id=initialize_run_id,
            permit_id=permit_id,
            topology_generation=generation,
            phases=("INITIALIZING",),
        )
        permit = validate_startup_permit(
            snapshot.startup_permit,
            expected_generation=generation,
            expected_mode="INIT",
        )
        if permit["permit_id"] != permit_id:
            raise RuntimeError("INIT permit changed before acceptance")
        accepted = _accepted_initialization_state(initializing)
        accepted_raw = _canonical_json(accepted)
        operations: list[dict[str, object]] = []
        if snapshot.startup_permit_raw is not None:
            operations.append({
                "op": "test",
                "path": f"/data/{STARTUP_PERMIT_KEY}",
                "value": snapshot.startup_permit_raw,
            })
        operations.append({
            "op": "replace",
            "path": f"/data/{INITIALIZATION_STATE_KEY}",
            "value": accepted_raw,
        })
        try:
            self._patch_initialization_configmap(
                release, namespace, snapshot, operations
            )
        except Exception as exc:
            observed = self.read_initialization_snapshot(release, namespace)
            if (
                observed.configmap_uid != snapshot.configmap_uid
                or observed.state_raw != accepted_raw
                or observed.state != accepted
                or observed.startup_permit != permit
            ):
                raise RuntimeError(
                    "INITIALIZING to ACCEPTED CAS was not committed"
                ) from exc
            return accepted
        observed = self.read_initialization_snapshot(release, namespace)
        if (
            observed.configmap_uid != snapshot.configmap_uid
            or observed.state_raw != accepted_raw
            or observed.state != accepted
            or observed.startup_permit != permit
        ):
            raise RuntimeError(
                "accepted initialization read-back changed exact identity"
            )
        return accepted

    def verify_initial_worker_templates(
        self,
        release: str,
        namespace: str,
        *,
        generation: str,
    ) -> None:
        for member in MEMBERS:
            statefulset = self._kubectl_json(
                namespace,
                "get",
                "statefulset",
                f"{release}-prism-serve-{member}",
            )
            template = (
                statefulset.get("spec", {}).get("template", {})
                if isinstance(statefulset, dict) else {}
            )
            labels = template.get("metadata", {}).get("labels", {})
            containers = template.get("spec", {}).get("containers", ())
            workers = [
                item for item in containers
                if isinstance(item, dict) and item.get("name") == "worker"
            ]
            if (
                labels.get("prism.sparksnail.ai/member") != member
                or labels.get("prism.sparksnail.ai/topology-generation")
                != generation
                or len(workers) != 1
            ):
                raise RuntimeError(
                    f"initial worker template mismatch for {member}"
                )
            environment = {
                item.get("name"): item.get("value")
                for item in workers[0].get("env", ())
                if isinstance(item, dict)
            }
            if (
                environment.get("PRISM_INSTANCE_ID") != member
                or environment.get("PRISM_TOPOLOGY_GENERATION") != generation
            ):
                raise RuntimeError(
                    f"initial worker environment mismatch for {member}"
                )

    @staticmethod
    def _worker_container_status(pod: dict[str, object]) -> dict[str, object]:
        statuses = [
            value for value in pod.get("status", {}).get("containerStatuses", ())
            if value.get("name") == "worker"
        ]
        if len(statuses) != 1:
            raise RuntimeError("exact worker container status is unavailable")
        return statuses[0]

    @staticmethod
    def _node_is_ready(node: dict[str, object]) -> bool:
        ready = [
            value for value in node.get("status", {}).get("conditions", ())
            if value.get("type") == "Ready"
        ]
        return len(ready) == 1 and ready[0].get("status") == "True"

    def _assert_node_available(
        self, namespace: str, expected: ProcessIdentity
    ) -> None:
        node = self._kubectl_json(namespace, "get", "node", expected.node_name)
        if str(node.get("metadata", {}).get("uid") or "") != expected.node_uid:
            raise RuntimeError(f"node uid drifted for {expected.member}")
        if not self._node_is_ready(node):
            raise RuntimeError(f"node is not Ready for {expected.member}")

    def capture_world(
        self, release: str, namespace: str, *, expected_generation: str
    ):
        topology = self._read_gateway_json(
            f"{self.gateway_url}/admin/topology"
        )
        if topology.get("topology_generation") != expected_generation:
            raise RuntimeError(
                "Gateway topology generation does not match --old-generation"
            )
        identities = {
            value["instance_id"]: value for value in topology.get("identities", ())
        }
        reports = topology.get("resource_reports", {})
        result = {}
        for member in MEMBERS:
            items = self._kubectl_json(
                namespace, "get", "pods", "-l",
                f"app.kubernetes.io/instance={release},prism.sparksnail.ai/member={member}",
            ).get("items", ())
            if len(items) != 1 or member not in identities:
                raise RuntimeError(f"cannot capture exact old process for {member}")
            pod = items[0]
            identity = identities[member]
            if identity.get("topology_generation") != expected_generation:
                raise RuntimeError(f"Gateway identity generation mismatch for {member}")
            if pod["metadata"]["uid"] != identity["pod_uid"]:
                raise RuntimeError(f"Gateway/Kubernetes pod uid mismatch for {member}")
            node = self._kubectl_json(
                namespace, "get", "node", pod["spec"]["nodeName"]
            )
            if not self._node_is_ready(node):
                raise RuntimeError(f"node is not Ready for {member}")
            status = self._worker_container_status(pod)
            if not status.get("containerID"):
                raise RuntimeError(f"container identity unavailable for {member}")
            restart_count = status.get("restartCount")
            if (
                isinstance(restart_count, bool)
                or not isinstance(restart_count, int)
                or restart_count < 0
            ):
                raise RuntimeError(
                    f"restartCount unavailable for {member}"
                )
            result[member] = ProcessIdentity(
                member=member, pod_uid=identity["pod_uid"],
                node_uid=node["metadata"]["uid"],
                container_id=status["containerID"],
                process_generation=identity["process_generation"],
                pod_name=pod["metadata"]["name"],
                node_name=pod["spec"]["nodeName"],
                resource_version=pod["metadata"]["resourceVersion"],
                topology_generation=expected_generation,
                container_name="worker",
                restart_count=restart_count,
            )
            report = reports.get(member, {})
            self._old_operation_ids.update(map(str, report.get("operation_ids", ())))
            self._old_resource_ids.update(map(str, report.get("resource_ids", ())))
        return result

    @property
    def captured_operation_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._old_operation_ids))

    @property
    def captured_resource_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._old_resource_ids))

    def include_required_old_operation_ids(
        self, operation_ids: tuple[str, ...]
    ) -> None:
        self._old_operation_ids.update(operation_ids)

    def include_required_old_resource_ids(
        self, resource_ids: tuple[str, ...]
    ) -> None:
        self._old_resource_ids.update(resource_ids)

    def start_termination_watch(
        self, release: str, namespace: str, expected: ProcessIdentity
    ) -> PodTerminationWatch:
        del release
        if (
            not expected.resource_version
            or not expected.pod_name
            or not expected.node_name
        ):
            raise RuntimeError(f"watch cursor unavailable for {expected.member}")
        self._assert_node_available(namespace, expected)
        query = urlencode((
            ("watch", "1"),
            ("resourceVersion", expected.resource_version),
            ("fieldSelector", f"metadata.name={expected.pod_name}"),
        ))
        watch_uri = (
            f"/api/v1/namespaces/{quote(namespace, safe='')}/pods?{query}"
        )
        probe_uri = f"{watch_uri}&timeoutSeconds=1"
        try:
            probe = subprocess.run(
                self._kubectl_args("get", "--raw", probe_uri),
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"could not establish exact Pod watch for {expected.member}: "
                "probe timed out"
            ) from exc
        if probe.returncode != 0:
            detail = probe.stderr.strip() or probe.stdout.strip()
            raise RuntimeError(
                f"could not establish exact Pod watch for {expected.member}: "
                f"{detail or f'probe exited {probe.returncode}'}"
            )
        try:
            for line in (probe.stdout or "").splitlines():
                if not line.strip():
                    continue
                fields = self._parse_watch_event(
                    self._normalize_raw_watch_event(line, expected.member),
                    expected.member,
                )
                if fields["event_type"] == "ERROR":
                    raise RuntimeError("Kubernetes watch probe returned ERROR")
        except RuntimeError as exc:
            raise RuntimeError(
                f"could not establish exact Pod watch for {expected.member}: {exc}"
            ) from exc


        process = subprocess.Popen(
            self._kubectl_args("get", "--raw", watch_uri),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        events: Queue[str | KubernetesWatchEvent | BaseException] = Queue()

        def read_events() -> None:
            try:
                if process.stdout is None:
                    raise RuntimeError("kubectl watch stdout is unavailable")
                for line in process.stdout:
                    value = line.rstrip("\r\n")
                    if value:
                        events.put(self._decode_raw_watch_event(
                            value, expected.member
                        ))
            except BaseException as exc:  # delivered to the controlling thread
                events.put(exc)

        reader = Thread(
            target=read_events,
            name=f"pd-worldctl-watch-{expected.member}",
            daemon=True,
        )
        reader.start()
        return PodTerminationWatch(expected, process, events, reader)

    @staticmethod
    def _decode_raw_watch_event(
        value: str, member: str,
    ) -> KubernetesWatchEvent:
        del member
        raw_digest = (
            "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        return KubernetesWatchEvent(
            raw_text=value,
            raw_sha256=raw_digest,
        )

    @classmethod
    def _decode_observed_watch_event(
        cls,
        event: KubernetesWatchEvent,
        member: str,
        *,
        raw_observation_sequence: int | None,
    ) -> DecodedKubernetesWatchEvent:
        try:
            decoded = json.loads(event.raw_text)
        except json.JSONDecodeError as exc:
            raise TerminationObservationError(
                f"malformed Pod watch event for {member}",
                raw_watch_event=None,
                raw_watch_event_text=event.raw_text,
                raw_watch_event_sha256=event.raw_sha256,
                resource_version="",
                raw_observation_sequence=raw_observation_sequence,
            ) from exc
        if not isinstance(decoded, dict):
            raise TerminationObservationError(
                f"malformed Pod watch event for {member}",
                raw_watch_event=decoded,
                raw_watch_event_text=event.raw_text,
                raw_watch_event_sha256=event.raw_sha256,
                resource_version="",
                raw_observation_sequence=raw_observation_sequence,
            )
        pod = decoded.get("object")
        metadata = pod.get("metadata") if isinstance(pod, dict) else None
        resource_version = (
            str(metadata.get("resourceVersion") or "")
            if isinstance(metadata, dict) else ""
        )
        try:
            normalized = cls._normalize_raw_watch_event(
                event.raw_text, member
            )
        except RuntimeError as exc:
            raise TerminationObservationError(
                str(exc),
                raw_watch_event=decoded,
                raw_watch_event_text=event.raw_text,
                raw_watch_event_sha256=event.raw_sha256,
                resource_version=resource_version,
                raw_observation_sequence=raw_observation_sequence,
            ) from exc
        return DecodedKubernetesWatchEvent(
            normalized=normalized,
            raw_object=decoded,
            raw_text=event.raw_text,
            raw_sha256=event.raw_sha256,
            resource_version=resource_version,
            raw_observation_sequence=raw_observation_sequence,
        )

    @staticmethod
    def _normalize_raw_watch_event(value: str, member: str) -> str:
        try:
            event = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"malformed Pod watch event for {member}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise RuntimeError(f"malformed Pod watch event for {member}")
        pod = event.get("object")
        if not isinstance(pod, dict):
            raise RuntimeError(f"malformed Pod watch event for {member}")
        if event["type"] == "ERROR":
            return "\t".join(("ERROR", *("" for _ in range(19))))

        metadata = pod.get("metadata", {})
        spec = pod.get("spec", {})
        status = pod.get("status", {})
        if not all(isinstance(item, dict) for item in (metadata, spec, status)):
            raise RuntimeError(f"malformed Pod watch event for {member}")
        container_statuses = status.get("containerStatuses", ())
        if not isinstance(container_statuses, (list, tuple)):
            raise RuntimeError(f"malformed Pod watch event for {member}")
        workers = [
            item for item in container_statuses
            if isinstance(item, dict) and item.get("name") == "worker"
        ]
        if len(workers) > 1:
            raise RuntimeError(f"ambiguous worker status for {member}")
        worker = workers[0] if workers else {}
        current_state = worker.get("state", {})
        last_state = worker.get("lastState", {})
        if not isinstance(current_state, dict) or not isinstance(last_state, dict):
            raise RuntimeError(f"malformed Pod watch event for {member}")
        current = current_state.get("terminated", {})
        previous = last_state.get("terminated", {})
        if not isinstance(current, dict) or not isinstance(previous, dict):
            raise RuntimeError(f"malformed Pod watch event for {member}")

        def field(source: dict[str, object], name: str) -> str:
            raw = source.get(name)
            return "" if raw is None else str(raw)

        return "\t".join((
            event["type"],
            field(metadata, "uid"),
            field(spec, "nodeName"),
            field(worker, "containerID"),
            field(worker, "restartCount"),
            field(current, "containerID"),
            field(current, "exitCode"),
            field(current, "reason"),
            field(current, "signal"),
            field(current, "message"),
            field(current, "startedAt"),
            field(current, "finishedAt"),
            field(previous, "containerID"),
            field(previous, "exitCode"),
            field(previous, "reason"),
            field(previous, "signal"),
            field(previous, "message"),
            field(previous, "startedAt"),
            field(previous, "finishedAt"),
            field(metadata, "resourceVersion"),
        ))

    def inject_process_fault(
        self,
        release: str,
        namespace: str,
        expected: ProcessIdentity,
    ) -> dict[str, object]:
        del release
        command = self._kubectl_args(
            "-n", namespace, "exec", expected.pod_name,
            "-c", "worker", "--",
            "python", "-m", "prism_infer.server.process_identity", "kill",
            "--path", WORKER_PROCESS_IDENTITY_PATH,
            "--expected-component", "worker",
            "--expected-instance-id", expected.member,
            "--expected-pod-uid", expected.pod_uid,
            "--expected-process-generation", expected.process_generation,
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_s,
        )
        result: dict[str, object] = {
            "command": command,
            "exec_return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        try:
            selection = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return result
        if isinstance(selection, dict):
            result.update(selection)
        return result

    def wait_injected_process_termination(
        self,
        release: str,
        namespace: str,
        watch: PodTerminationWatch,
        expected: ProcessIdentity,
        *,
        raw_observation_sink: Callable[
            [KubernetesWatchEvent], tuple[int, str]
        ] | None = None,
    ) -> TerminationProof:
        """Prove the injected SIGKILL before controlled scale-to-zero."""
        observation = self.observe_injected_process_termination(
            release,
            namespace,
            watch,
            expected,
            raw_observation_sink=raw_observation_sink,
        )
        _validate_worker_crash_termination(observation.proof, expected)
        return observation.proof

    def observe_injected_process_termination(
        self,
        release: str,
        namespace: str,
        watch: PodTerminationWatch,
        expected: ProcessIdentity,
        *,
        raw_observation_sink: Callable[
            [KubernetesWatchEvent], tuple[int, str]
        ] | None = None,
    ) -> TerminationObservation:
        del release
        return self._consume_termination_observation(
            namespace,
            watch,
            expected,
            prior_termination=None,
            require_deleted=False,
            raw_observation_sink=raw_observation_sink,
        )

    def stop_member(self, release: str, namespace: str, member: str) -> None:
        command = WorldRestartGuard.scale_down_commands(
            release, namespace
        )[MEMBERS.index(member)]
        self._run_mutation(self._kubectl_args(*command[1:]))

    @staticmethod
    def _parse_watch_event(value: str, member: str) -> dict[str, str]:
        fields = value.split("\t")
        if len(fields) == 6:
            # Kept for old dry-run fixtures; the real watch always emits the
            # full current/last-state shape above.
            event_type, pod_uid, node_name, container_id, raw_exit, resource = fields
            return {
                "event_type": event_type,
                "pod_uid": pod_uid,
                "node_name": node_name,
                "current_container_id": container_id,
                "restart_count": "",
                "current_terminated_container_id": container_id if raw_exit else "",
                "current_exit": raw_exit,
                "current_reason": "",
                "current_signal": "",
                "current_message": "",
                "current_started_at": "",
                "current_finished_at": "",
                "last_terminated_container_id": "",
                "last_exit": "",
                "last_reason": "",
                "last_signal": "",
                "last_message": "",
                "last_started_at": "",
                "last_finished_at": "",
                "resource_version": resource,
            }
        if len(fields) == 15:
            names = (
                "event_type", "pod_uid", "node_name",
                "current_container_id",
                "current_terminated_container_id", "current_exit",
                "current_reason", "current_message", "current_finished_at",
                "last_terminated_container_id", "last_exit", "last_reason",
                "last_message", "last_finished_at", "resource_version",
            )
            parsed = dict(zip(names, fields, strict=True))
            parsed.update({
                "restart_count": "",
                "current_signal": "",
                "current_started_at": "",
                "last_signal": "",
                "last_started_at": "",
            })
            return parsed
        if len(fields) != 20:
            raise RuntimeError(f"malformed Pod watch event for {member}")
        names = (
            "event_type", "pod_uid", "node_name", "current_container_id",
            "restart_count", "current_terminated_container_id",
            "current_exit", "current_reason", "current_signal",
            "current_message", "current_started_at",
            "current_finished_at", "last_terminated_container_id",
            "last_exit", "last_reason", "last_signal", "last_message",
            "last_started_at", "last_finished_at", "resource_version",
        )
        return dict(zip(names, fields, strict=True))

    @staticmethod
    def _event_termination(
        fields: dict[str, str], expected: ProcessIdentity,
    ) -> TerminationProof | None:
        current_matches = (
            fields["current_terminated_container_id"]
            == expected.container_id
            and bool(fields["current_exit"])
        )
        last_matches = (
            fields["last_terminated_container_id"]
            == expected.container_id
            and bool(fields["last_exit"])
        )
        if current_matches and last_matches:
            raise RuntimeError(
                f"ambiguous termination state for {expected.member}"
            )
        if not current_matches and not last_matches:
            return None

        source = (
            "state.terminated"
            if current_matches else "lastState.terminated"
        )
        prefix = "current" if current_matches else "last"
        raw_exit = fields[f"{prefix}_exit"]
        try:
            exit_code = int(raw_exit)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid termination exit code for {expected.member}"
            ) from exc
        raw_signal = fields[f"{prefix}_signal"]
        try:
            signal = int(raw_signal) if raw_signal else None
        except ValueError as exc:
            raise RuntimeError(
                f"invalid termination signal for {expected.member}"
            ) from exc
        raw_restart_count = fields["restart_count"]
        if raw_restart_count:
            try:
                restart_count = int(raw_restart_count)
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid restartCount for {expected.member}"
                ) from exc
            if restart_count < 0:
                raise RuntimeError(
                    f"invalid restartCount for {expected.member}"
                )
        else:


            restart_count = expected.restart_count + (
                1 if last_matches else 0
            )

        adjacent: str | None = None
        if current_matches:
            if fields["current_container_id"] != expected.container_id:
                raise RuntimeError(
                    "state.terminated does not bind the captured container"
                )
            if restart_count != expected.restart_count:
                raise RuntimeError(
                    "state.terminated restartCount does not match capture"
                )
        else:
            adjacent = fields["current_container_id"] or None
            if restart_count != expected.restart_count + 1:
                raise RuntimeError(
                    "lastState.terminated restartCount did not increase "
                    "exactly once"
                )
            if (
                adjacent is None
                or adjacent == expected.container_id
            ):
                raise RuntimeError(
                    "lastState.terminated lacks a distinct adjacent current "
                    "container"
                )

        return TerminationProof(
            member=expected.member,
            pod_uid=expected.pod_uid,
            node_uid=expected.node_uid,
            container_id=expected.container_id,
            process_generation=expected.process_generation,
            exit_code=exit_code,
            container_terminated=True,
            pod_deleted=False,
            termination_reason=fields[f"{prefix}_reason"],
            termination_message=fields[f"{prefix}_message"],
            finished_at=fields[f"{prefix}_finished_at"],
            topology_generation=expected.topology_generation,
            container_name=expected.container_name,
            watch_start_resource_version=expected.resource_version,
            observed_resource_version=fields["resource_version"],
            restart_count_before=expected.restart_count,
            restart_count_observed=restart_count,
            termination_source=source,
            termination_event_type=fields["event_type"],
            signal=signal,
            started_at=fields[f"{prefix}_started_at"],
            adjacent_current_container_id=adjacent,
        )

    @staticmethod
    def _matches_frozen_termination(
        candidate: TerminationProof,
        frozen: TerminationProof,
    ) -> bool:

        candidate_projection = (
            candidate.termination_source,
            candidate.restart_count_observed,
            candidate.adjacent_current_container_id,
        )
        frozen_projection = (
            frozen.termination_source,
            frozen.restart_count_observed,
            frozen.adjacent_current_container_id,
        )
        if candidate_projection != frozen_projection:
            advanced_to_last_state = (
                frozen.termination_source == "state.terminated"
                and frozen.restart_count_observed
                == frozen.restart_count_before
                and frozen.adjacent_current_container_id is None
                and candidate.termination_source == "lastState.terminated"
                and candidate.restart_count_observed
                == frozen.restart_count_before + 1
                and candidate.adjacent_current_container_id is not None
                and candidate.adjacent_current_container_id
                != frozen.container_id
            )
            if not advanced_to_last_state:
                return False
        comparable = replace(
            candidate,
            observed_resource_version=frozen.observed_resource_version,
            restart_count_observed=frozen.restart_count_observed,
            termination_source=frozen.termination_source,
            termination_event_type=frozen.termination_event_type,
            adjacent_current_container_id=(
                frozen.adjacent_current_container_id
            ),
            termination_raw_pod_json_sha256=(
                frozen.termination_raw_pod_json_sha256
            ),
            termination_raw_observation_sequence=(
                frozen.termination_raw_observation_sequence
            ),
        )
        return comparable == frozen

    @staticmethod
    def _termination_observation(
        proof: TerminationProof,
        event: DecodedKubernetesWatchEvent | None,
    ) -> TerminationObservation:
        if event is None:
            return TerminationObservation(proof=proof)
        return TerminationObservation(
            proof=proof,
            raw_watch_event=event.raw_object,
            raw_watch_event_text=event.raw_text,
            raw_watch_event_sha256=event.raw_sha256,
            resource_version=event.resource_version,
            raw_observation_sequence=event.raw_observation_sequence,
        )

    @classmethod
    def _observation_error(
        cls,
        message: str,
        event: DecodedKubernetesWatchEvent | None,
        partial_observation: TerminationObservation | None,
    ) -> TerminationObservationError:
        return TerminationObservationError(
            message,
            raw_watch_event=(
                event.raw_object if event is not None else None
            ),
            raw_watch_event_text=(
                event.raw_text if event is not None else ""
            ),
            raw_watch_event_sha256=(
                event.raw_sha256 if event is not None else ""
            ),
            resource_version=(
                event.resource_version if event is not None else ""
            ),
            raw_observation_sequence=(
                event.raw_observation_sequence
                if event is not None else None
            ),
            partial_observation=partial_observation,
        )

    def _consume_termination_observation(
        self,
        namespace: str,
        watch: PodTerminationWatch,
        expected: ProcessIdentity,
        *,
        prior_termination: TerminationProof | None,
        require_deleted: bool,
        raw_observation_sink: Callable[
            [KubernetesWatchEvent], tuple[int, str]
        ] | None = None,
    ) -> TerminationObservation:
        if watch.expected != expected:
            raise RuntimeError("termination watch identity changed")
        observed = prior_termination
        observed_event: DecodedKubernetesWatchEvent | None = None
        last_raw_event: DecodedKubernetesWatchEvent | None = None

        def partial_observation() -> TerminationObservation | None:
            if observed is None:
                return None
            return self._termination_observation(
                observed, observed_event,
            )

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                self._assert_node_available(namespace, expected)
            except Exception as exc:
                raise self._observation_error(
                    str(exc), last_raw_event, partial_observation(),
                ) from exc
            try:
                value = watch.events.get(timeout=0.25)
            except Empty:
                if watch.process.poll() is not None:
                    stderr = (
                        watch.process.stderr.read()
                        if watch.process.stderr is not None else ""
                    )
                    raise self._observation_error(
                        (
                            "exact Pod watch ended before proof for "
                            f"{expected.member}: {stderr.strip()}"
                        ),
                        last_raw_event,
                        partial_observation(),
                    )
                continue
            if isinstance(value, BaseException):
                if isinstance(value, TerminationObservationError):
                    if (
                        value.partial_observation is None
                        and partial_observation() is not None
                    ):
                        raise TerminationObservationError(
                            str(value),
                            raw_watch_event=value.raw_watch_event,
                            raw_watch_event_text=(
                                value.raw_watch_event_text
                            ),
                            raw_watch_event_sha256=(
                                value.raw_watch_event_sha256
                            ),
                            resource_version=value.resource_version,
                            raw_observation_sequence=(
                                value.raw_observation_sequence
                            ),
                            partial_observation=partial_observation(),
                        ) from value
                    raise value
                raise self._observation_error(
                    f"exact Pod watch failed for {expected.member}",
                    last_raw_event,
                    partial_observation(),
                ) from value
            raw_event: DecodedKubernetesWatchEvent | None = None
            if isinstance(value, KubernetesWatchEvent):
                raw_sequence = None
                if raw_observation_sink is not None:
                    raw_sequence, raw_digest = raw_observation_sink(value)
                    if raw_digest != value.raw_sha256:
                        raise RuntimeError(
                            "raw observation sink digest mismatch"
                        )
                try:
                    raw_event = self._decode_observed_watch_event(
                        value,
                        expected.member,
                        raw_observation_sequence=raw_sequence,
                    )
                except TerminationObservationError as exc:
                    if partial_observation() is None:
                        raise
                    raise TerminationObservationError(
                        str(exc),
                        raw_watch_event=exc.raw_watch_event,
                        raw_watch_event_text=exc.raw_watch_event_text,
                        raw_watch_event_sha256=(
                            exc.raw_watch_event_sha256
                        ),
                        resource_version=exc.resource_version,
                        raw_observation_sequence=(
                            exc.raw_observation_sequence
                        ),
                        partial_observation=partial_observation(),
                    ) from exc
                normalized = raw_event.normalized
                last_raw_event = raw_event
            else:
                normalized = value
            try:
                fields = self._parse_watch_event(
                    normalized, expected.member
                )
            except RuntimeError as exc:
                raise self._observation_error(
                    str(exc), raw_event, partial_observation(),
                ) from exc
            if fields["event_type"] == "ERROR":
                raise self._observation_error(
                    f"Kubernetes watch error for {expected.member}",
                    raw_event,
                    partial_observation(),
                )
            if raw_event is not None and not fields["resource_version"]:
                raise self._observation_error(
                    (
                        "Kubernetes watch event lacks resourceVersion for "
                        f"{expected.member}"
                    ),
                    raw_event,
                    partial_observation(),
                )
            if fields["pod_uid"] != expected.pod_uid:
                raise self._observation_error(
                    f"Pod uid drifted for {expected.member}",
                    raw_event,
                    partial_observation(),
                )
            if fields["node_name"] != expected.node_name:
                raise self._observation_error(
                    f"node assignment drifted for {expected.member}",
                    raw_event,
                    partial_observation(),
                )
            try:
                candidate = self._event_termination(fields, expected)
            except RuntimeError as exc:
                raise self._observation_error(
                    str(exc), raw_event, partial_observation(),
                ) from exc
            if candidate is not None:
                if fields["event_type"] == "DELETED" and observed is not None:
                    if not self._matches_frozen_termination(
                        candidate, observed,
                    ):
                        raise self._observation_error(
                            (
                                f"termination proof changed for "
                                f"{expected.member} at deletion"
                            ),
                            raw_event,
                            partial_observation(),
                        )
                    candidate = None
            if candidate is not None:
                if fields["event_type"] != "MODIFIED":
                    raise self._observation_error(
                        (
                            f"termination for {expected.member} was not "
                            "frozen in a MODIFIED transition before deletion"
                        ),
                        raw_event,
                        partial_observation(),
                    )
                if raw_event is not None:
                    raw_pod = raw_event.raw_object.get("object")
                    if not isinstance(raw_pod, dict):
                        raise self._observation_error(
                            "termination raw event lacks Pod object",
                            raw_event,
                            partial_observation(),
                        )
                    if raw_event.raw_observation_sequence is None:
                        raise self._observation_error(
                            "termination raw event was not durably journaled",
                            raw_event,
                            partial_observation(),
                        )
                    candidate = replace(
                        candidate,
                        termination_raw_pod_json_sha256=(
                            _canonical_sha256(raw_pod)
                        ),
                        termination_raw_observation_sequence=(
                            raw_event.raw_observation_sequence
                        ),
                    )
                if observed is not None:





                    if not self._matches_frozen_termination(
                        candidate, observed,
                    ):
                        raise self._observation_error(
                            f"termination proof changed for {expected.member}",
                            raw_event,
                            partial_observation(),
                        )
                    candidate = None
                if candidate is not None:
                    observed = candidate
                    observed_event = raw_event
            current_container = fields["current_container_id"]
            if (
                current_container
                and current_container != expected.container_id
                and observed is None
            ):
                raise self._observation_error(
                    f"worker container id drifted for {expected.member}",
                    raw_event,
                    partial_observation(),
                )
            if observed is not None and not require_deleted:
                return self._termination_observation(
                    observed, observed_event,
                )
            if fields["event_type"] == "DELETED":
                if observed is None:
                    raise self._observation_error(
                        (
                            f"old Pod {expected.member} was deleted without "
                            "exact container state.terminated"
                        ),
                        raw_event,
                        partial_observation(),
                    )
                if (
                    raw_event is None
                    or raw_event.raw_observation_sequence is None
                    or not isinstance(
                        raw_event.raw_object.get("object"), dict
                    )
                ):
                    raise self._observation_error(
                        (
                            f"old Pod {expected.member} deletion raw "
                            "transition was not durably journaled"
                        ),
                        raw_event,
                        partial_observation(),
                    )
                if (
                    observed.termination_event_type != "MODIFIED"
                    or observed.termination_raw_observation_sequence is None
                ):
                    raise self._observation_error(
                        (
                            f"old Pod {expected.member} deletion is not "
                            "linked to a frozen MODIFIED termination"
                        ),
                        raw_event,
                        partial_observation(),
                    )
                deletion_pod = raw_event.raw_object["object"]
                return self._termination_observation(
                    replace(
                        observed,
                        pod_deleted=True,
                        deletion_resource_version=fields[
                            "resource_version"
                        ],
                        deletion_event_type="DELETED",
                        deletion_raw_pod_json_sha256=(
                            _canonical_sha256(deletion_pod)
                        ),
                        deletion_raw_observation_sequence=(
                            raw_event.raw_observation_sequence
                        ),
                    ),
                    raw_event,
                )
        raise self._observation_error(
            f"timed out proving termination for {expected.member}",
            last_raw_event,
            partial_observation(),
        )

    def _consume_termination_watch(
        self,
        namespace: str,
        watch: PodTerminationWatch,
        expected: ProcessIdentity,
        *,
        prior_termination: TerminationProof | None,
        require_deleted: bool,
        raw_observation_sink: Callable[
            [KubernetesWatchEvent], tuple[int, str]
        ] | None = None,
    ) -> TerminationProof:
        return self._consume_termination_observation(
            namespace,
            watch,
            expected,
            prior_termination=prior_termination,
            require_deleted=require_deleted,
            raw_observation_sink=raw_observation_sink,
        ).proof

    def wait_natural_watchdog_termination(
        self,
        release: str,
        namespace: str,
        watch: PodTerminationWatch,
        expected: ProcessIdentity,
        *,
        expected_generation: str,
        expected_operation_ids: tuple[str, ...],
        raw_observation_sink: Callable[
            [KubernetesWatchEvent], tuple[int, str]
        ] | None = None,
    ) -> TerminationProof:
        """Wait for the captured worker to exit itself with an exact watchdog message."""
        observation = self.observe_natural_watchdog_termination(
            release,
            namespace,
            watch,
            expected,
            expected_generation=expected_generation,
            expected_operation_ids=expected_operation_ids,
            raw_observation_sink=raw_observation_sink,
        )
        _validate_watchdog_termination(
            observation.proof,
            expected,
            expected_generation=expected_generation,
            expected_operation_ids=expected_operation_ids,
        )
        return observation.proof

    def observe_natural_watchdog_termination(
        self,
        release: str,
        namespace: str,
        watch: PodTerminationWatch,
        expected: ProcessIdentity,
        *,
        expected_generation: str,
        expected_operation_ids: tuple[str, ...],
        raw_observation_sink: Callable[
            [KubernetesWatchEvent], tuple[int, str]
        ] | None = None,
    ) -> TerminationObservation:
        del release, expected_generation, expected_operation_ids
        return self._consume_termination_observation(
            namespace,
            watch,
            expected,
            prior_termination=None,
            require_deleted=False,
            raw_observation_sink=raw_observation_sink,
        )

    def wait_termination(
        self,
        release: str,
        namespace: str,
        watch: PodTerminationWatch,
        expected: ProcessIdentity,
        *,
        prior_termination: TerminationProof | None = None,
        raw_observation_sink: Callable[
            [KubernetesWatchEvent], tuple[int, str]
        ] | None = None,
    ) -> TerminationProof:
        return self.observe_termination(
            release,
            namespace,
            watch,
            expected,
            prior_termination=prior_termination,
            raw_observation_sink=raw_observation_sink,
        ).proof

    def observe_termination(
        self,
        release: str,
        namespace: str,
        watch: PodTerminationWatch,
        expected: ProcessIdentity,
        *,
        prior_termination: TerminationProof | None = None,
        raw_observation_sink: Callable[
            [KubernetesWatchEvent], tuple[int, str]
        ] | None = None,
    ) -> TerminationObservation:
        del release
        return self._consume_termination_observation(
            namespace,
            watch,
            expected,
            prior_termination=prior_termination,
            require_deleted=True,
            raw_observation_sink=raw_observation_sink,
        )

    @staticmethod
    def close_termination_watch(watch: PodTerminationWatch) -> None:
        if watch.process.poll() is None:
            watch.process.terminate()
            try:
                watch.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                watch.process.kill()
                watch.process.wait(timeout=2)

    def patch_generation(
        self, release: str, namespace: str, chart: str, generation: str
    ) -> None:
        command = self._helm_args(
            "upgrade", release, chart, "--namespace", namespace,
            "--reuse-values", "--set-string",
            f"worker.topologyGeneration={generation}",
            "--set", "worker.replicas=0",
        )
        self._run_helm_upgrade(
            release=release,
            namespace=namespace,
            command=command,
            overrides=(
                (("worker", "topologyGeneration"), generation),
                (("worker", "replicas"), 0),
            ),
        )

    def start_member(self, release: str, namespace: str, member: str) -> None:
        self._run_mutation(self._kubectl_args(
            "-n", namespace, "scale", "statefulset",
            f"{release}-prism-serve-{member}", "--replicas=1",
        ))

    def wait_fresh_pod_uids(
        self,
        release: str,
        namespace: str,
        *,
        generation: str,
        old_pod_uids: dict[str, str],
    ) -> dict[str, str]:

        if set(old_pod_uids) != set(MEMBERS):
            raise ValueError("old Pod UID map must contain exact four members")
        return self._wait_worker_pod_uids(
            release,
            namespace,
            generation=generation,
            forbidden_uids=set(old_pod_uids.values()),
        )

    def wait_initial_pod_uids(
        self,
        release: str,
        namespace: str,
        *,
        generation: str,
    ) -> dict[str, str]:

        return self._wait_worker_pod_uids(
            release,
            namespace,
            generation=generation,
            forbidden_uids=set(),
        )

    def _wait_worker_pod_uids(
        self,
        release: str,
        namespace: str,
        *,
        generation: str,
        forbidden_uids: set[str],
    ) -> dict[str, str]:
        deadline = time.monotonic() + self.timeout_s
        selector = f"app.kubernetes.io/instance={release}"
        while time.monotonic() < deadline:
            try:
                items = self._kubectl_json(
                    namespace, "get", "pods", "-l", selector
                ).get("items")
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                time.sleep(0.5)
                continue
            if not isinstance(items, list):
                raise RuntimeError("Kubernetes Pod list is unavailable")

            observed: dict[str, str] = {}
            for pod in items:
                if not isinstance(pod, dict):
                    raise RuntimeError("Kubernetes Pod entry is invalid")
                metadata = pod.get("metadata")
                if not isinstance(metadata, dict):
                    raise RuntimeError("Kubernetes Pod metadata is invalid")
                labels = metadata.get("labels")
                if not isinstance(labels, dict):
                    continue
                member = labels.get("prism.sparksnail.ai/member")
                if member is None:
                    continue
                if member not in MEMBERS:
                    raise RuntimeError("unexpected fixed-topology worker member")
                if metadata.get("deletionTimestamp"):
                    continue
                if (
                    labels.get("prism.sparksnail.ai/topology-generation")
                    != generation
                ):
                    raise RuntimeError(
                        f"fresh Pod generation mismatch for {member}"
                    )
                pod_uid = metadata.get("uid")
                if not isinstance(pod_uid, str) or not pod_uid:
                    raise RuntimeError(f"fresh Pod UID unavailable for {member}")
                if member in observed:
                    raise RuntimeError(
                        f"multiple live fresh Pods observed for {member}"
                    )
                observed[str(member)] = pod_uid

            if set(observed) == set(MEMBERS):
                if len(set(observed.values())) != len(MEMBERS):
                    raise RuntimeError("fresh Pod UIDs must be unique")
                if set(observed.values()) & forbidden_uids:
                    raise RuntimeError("fresh Pod UID reuses an old-world Pod")
                return {member: observed[member] for member in MEMBERS}
            time.sleep(0.5)
        raise TimeoutError("timed out waiting for exact four fresh worker Pods")

    def _read_startup_permit(
        self, release: str, namespace: str
    ) -> tuple[str, dict[str, object]]:
        config_map = self._kubectl_json(
            namespace,
            "get",
            "configmap",
            f"{release}-prism-serve-worker-topology",
        )
        data = config_map.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("worker topology ConfigMap data is unavailable")
        raw = data.get("startup-permit.json")
        if not isinstance(raw, str):
            raise RuntimeError("startup permit is absent from ConfigMap")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("startup permit ConfigMap JSON is invalid") from exc
        return raw, validate_startup_permit(parsed)

    def publish_startup_permit(
        self,
        release: str,
        namespace: str,
        startup_permit: dict[str, object],
    ) -> dict[str, object]:

        permit = validate_startup_permit(
            startup_permit, expected_mode="RESTART"
        )
        canonical = _canonical_json(permit)
        patch = _canonical_json({
            "data": {"startup-permit.json": canonical}
        })
        command = self._kubectl_args(
            "-n",
            namespace,
            "patch",
            "configmap",
            f"{release}-prism-serve-worker-topology",
            "--type",
            "merge",
            "--patch",
            patch,
        )
        try:
            self._run_mutation(command)
        except Exception as exc:
            try:
                observed_raw, observed = self._read_startup_permit(
                    release, namespace
                )
            except Exception as read_exc:
                raise RuntimeError(
                    "response loss did not read back the exact startup permit"
                ) from read_exc
            if observed_raw != canonical or observed != permit:
                raise RuntimeError(
                    "response loss did not read back the exact startup permit"
                ) from exc
        return permit

    def persist_started_world(
        self,
        release: str,
        namespace: str,
        chart: str,
        generation: str,
        startup_permit: dict[str, object],
    ) -> None:
        permit = validate_startup_permit(
            startup_permit,
            expected_generation=generation,
            expected_mode="RESTART",
        )
        command = self._helm_args(
            "upgrade", release, chart, "--namespace", namespace,
            "--reuse-values", "--set-string",
            f"worker.topologyGeneration={generation}",
            "--set-string", f"gateway.topologyGeneration={generation}",
            "--set", "worker.replicas=1",
            "--set-json",
            f"worker.startupPermitJson={_canonical_json(permit)}",
        )
        self._run_helm_upgrade(
            release=release,
            namespace=namespace,
            command=command,
            overrides=(
                (("worker", "topologyGeneration"), generation),
                (("gateway", "topologyGeneration"), generation),
                (("worker", "replicas"), 1),
                (("worker", "startupPermitJson"), permit),
            ),
        )

    def wait_initialization_evidence(
        self, *, generation: str
    ) -> dict[str, object]:
        deadline = time.monotonic() + self.timeout_s
        query = urlencode({"generation": generation})
        while time.monotonic() < deadline:
            try:
                body = self._json_url(
                    f"{self.gateway_url}/admin/topology/evidence?{query}"
                )
                if body.get("ready") is True:
                    return body
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError(
            "timed out waiting for fresh initialization probes/reports"
        )

    def wait_gateway_ready(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                body = self._json_url(f"{self.gateway_url}/readyz")
                if body.get("status") == "ready":
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError("timed out waiting for Gateway admission readiness")

    def persist_initialized_world(
        self,
        release: str,
        namespace: str,
        chart: str,
        generation: str,
        startup_permit: dict[str, object],
        accepted_state: dict[str, object],
    ) -> None:
        permit = validate_startup_permit(
            startup_permit,
            expected_generation=generation,
            expected_mode="INIT",
        )
        state = _require_initialization_identity(
            accepted_state,
            initialize_run_id=str(accepted_state.get("initialize_run_id", "")),
            permit_id=str(accepted_state.get("permit_id", "")),
            topology_generation=generation,
            phases=("ACCEPTED",),
        )
        if permit["permit_id"] != state["permit_id"]:
            raise RuntimeError(
                "persisted initialization permit/state identity mismatch"
            )
        command = self._helm_args(
            "upgrade",
            release,
            chart,
            "--namespace",
            namespace,
            "--reuse-values",
            "--set-string",
            f"worker.topologyGeneration={generation}",
            "--set-string",
            f"gateway.topologyGeneration={generation}",
            "--set",
            "worker.replicas=1",
            "--set-json",
            f"worker.startupPermitJson={_canonical_json(permit)}",
            "--set-json",
            f"worker.initializationStateJson={_canonical_json(state)}",
        )
        self._run_helm_upgrade(
            release=release,
            namespace=namespace,
            command=command,
            overrides=(
                (("worker", "topologyGeneration"), generation),
                (("gateway", "topologyGeneration"), generation),
                (("worker", "replicas"), 1),
                (("worker", "startupPermitJson"), permit),
                (("worker", "initializationStateJson"), state),
            ),
        )

    def wait_acceptance_evidence(
        self, *, old_generation: str, new_generation: str,
        restart_run_id: str,
        termination_proofs: tuple[TerminationProof, ...],
    ) -> dict[str, object]:
        deadline = time.monotonic() + self.timeout_s
        query = urlencode({"generation": new_generation})
        while time.monotonic() < deadline:
            try:
                body = self._json_url(
                    f"{self.gateway_url}/admin/topology/evidence?{query}"
                )
                if body.get("ready") is True:
                    for report in body.get("resource_reports", {}).values():
                        if set(map(str, report.get("operation_ids", ()))) & self._old_operation_ids:
                            raise RuntimeError("fresh report still contains an old operation")
                        if set(map(str, report.get("resource_ids", ()))) & self._old_resource_ids:
                            raise RuntimeError("fresh report still contains an old resource")
                        report["excluded_operation_ids"] = sorted(self._old_operation_ids)
                        report["excluded_resource_ids"] = sorted(self._old_resource_ids)
                    body.update({
                        "restart_run_id": restart_run_id,
                        "old_topology_generation": old_generation,
                        "new_topology_generation": new_generation,
                        "termination_records": [
                            _termination_authority_record(proof)
                            for proof in termination_proofs
                        ],
                        "old_operation_ids": sorted(self._old_operation_ids),
                        "old_resource_ids": sorted(self._old_resource_ids),
                    })
                    validate_termination_records(
                        body["termination_records"],
                        expected_generation=old_generation,
                        expected_members=MEMBERS,
                    )
                    return body
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError("timed out waiting for fresh replacement probes/reports")

    def accept_topology(self, evidence: dict[str, object]):
        return self._json_url(
            f"{self.gateway_url}/admin/topology/accept", body=evidence
        )


class WorldRestartGuard:
    def __init__(
        self,
        old_generation: str,
        new_generation: str,
        expected_old_processes: dict[str, ProcessIdentity],
        *,
        require_canonical_authority: bool = False,
    ) -> None:
        if not old_generation or not new_generation or old_generation == new_generation:
            raise ValueError("whole-world restart requires a fresh generation")
        self.old_generation = old_generation
        self.new_generation = new_generation
        self.require_canonical_authority = require_canonical_authority
        if set(expected_old_processes) != set(MEMBERS):
            raise ValueError("expected old world must contain four members")
        self.expected_old_processes = dict(expected_old_processes)
        for member, identity in self.expected_old_processes.items():
            if identity.member != member or any(
                not value for value in (
                    identity.pod_uid,
                    identity.node_uid,
                    identity.container_id,
                    identity.process_generation,
                )
            ):
                raise ValueError("expected old process physical identity is incomplete")
        if len({value.pod_uid for value in expected_old_processes.values()}) != 4:
            raise ValueError("expected old Pod UIDs must be unique")
        if len({value.container_id for value in expected_old_processes.values()}) != 4:
            raise ValueError("expected old container IDs must be unique")
        self._proofs: dict[str, TerminationProof] = {}

    def record_termination(self, proof: TerminationProof) -> None:
        if proof.member not in MEMBERS:
            raise ValueError(f"unknown fixed-topology member: {proof.member}")
        if not proof.pod_uid or not proof.node_uid or not proof.container_id:
            raise ValueError("termination identity fields are required")
        if not proof.container_terminated or not proof.pod_deleted:
            raise ValueError("termination proof is not complete")
        if self.require_canonical_authority:
            _termination_authority_record(proof)
        expected = self.expected_old_processes[proof.member]
        if (
            proof.pod_uid,
            proof.node_uid,
            proof.container_id,
            proof.process_generation,
        ) != (
            expected.pod_uid,
            expected.node_uid,
            expected.container_id,
            expected.process_generation,
        ):
            raise ValueError("termination proof does not bind expected old process")
        prior = self._proofs.get(proof.member)
        if prior is not None and prior != proof:
            raise ValueError("termination proof changed for member")
        self._proofs[proof.member] = proof

    @property
    def has_four_termination_proofs(self) -> bool:
        return set(self._proofs) == set(MEMBERS) and len(
            {proof.pod_uid for proof in self._proofs.values()}
        ) == 4 and len({
            proof.container_id for proof in self._proofs.values()
        }) == 4

    @staticmethod
    def scale_down_commands(release: str, namespace: str) -> list[list[str]]:
        return [
            [
                "kubectl", "-n", namespace, "scale", "statefulset",
                f"{release}-prism-serve-{member}", "--replicas=0",
            ]
            for member in MEMBERS
        ]

    def patch_generation_commands(
        self, release: str, namespace: str, chart: str
    ) -> list[list[str]]:
        if not self.has_four_termination_proofs:
            raise ValueError("four termination proofs are required before patch")
        # Helm is the deployment authority: this atomically updates the worker
        # templates and Gateway topology ConfigMap from one immutable value.
        return [[
            "helm", "upgrade", release, chart, "--namespace", namespace,
            "--reuse-values", "--set-string",
            f"worker.topologyGeneration={self.new_generation}",
            "--set", "worker.replicas=0",
        ]]

    def scale_up_commands(self, release: str, namespace: str) -> list[list[str]]:
        if not self.has_four_termination_proofs:
            raise ValueError("four termination proofs are required before scale up")
        return [
            [
                "kubectl", "-n", namespace, "scale", "statefulset",
                f"{release}-prism-serve-{member}", "--replicas=1",
            ]
            for member in MEMBERS
        ]


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    """Durably replace the local operator run state before remote mutation."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_or_create_initialize_run_state(
    path: Path,
    *,
    release: str,
    namespace: str,
    chart: str,
    generation: str,
) -> dict[str, object]:
    identity = {
        "format_version": 2,
        "operation": "initialize",
        "release": release,
        "namespace": namespace,
        "chart": chart,
        "topology_generation": generation,
    }
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or any(
            value.get(key) != expected for key, expected in identity.items()
        ):
            raise RuntimeError(
                "pd-worldctl initialize state does not match this command"
            )
        if value.get("phase") not in {
            "CREATED", "CLAIMED", "PERMIT_READY", "ACCEPTED"
        }:
            raise RuntimeError(
                "pd-worldctl initialize state phase is invalid"
            )
        for field in ("initialize_run_id", "permit_id"):
            if not isinstance(value.get(field), str) or not value[field]:
                raise RuntimeError(
                    f"pd-worldctl initialize {field} is invalid"
                )
        return value
    value = {
        **identity,
        "initialize_run_id": uuid.uuid4().hex,
        "permit_id": str(uuid.uuid4()),
        "configmap_uid": None,
        "phase": "CREATED",
        "startup_permit": None,
        "accepted_response": None,
    }
    _atomic_write_json(path, value)
    return value


def _load_or_create_run_state(
    path: Path,
    *,
    release: str,
    namespace: str,
    chart: str,
    old_generation: str,
    new_generation: str,
    fault_member: str | None,
    watchdog_member: str | None,
    expected_operation_ids: tuple[str, ...] = (),
    required_old_operation_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    identity = {
        "format_version": 1,
        "release": release,
        "namespace": namespace,
        "chart": chart,
        "old_generation": old_generation,
        "new_generation": new_generation,
        "fault_member": fault_member or "",
        "watchdog_member": watchdog_member or "",
        "expected_operation_ids": sorted(set(expected_operation_ids)),
        "required_old_operation_ids": sorted(set(required_old_operation_ids)),
    }
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or any(
            value.get(key) != expected for key, expected in identity.items()
        ):
            raise RuntimeError("pd-worldctl run state does not match this command")
        if value.get("phase") not in {
            "CREATED", "STOPPING", "EVIDENCE_READY", "ACCEPTED"
        }:
            raise RuntimeError("pd-worldctl run state phase is invalid")
        checkpoint = _restart_checkpoint(value)
        allowed_checkpoints = {
            "CREATED": {"CREATED"},
            "STOPPING": {
                "CAPTURED",
                "TERMINATED",
                "GENERATION_PATCHED",
                "STARTED",
                "PERMIT_READY",
                "PERMIT_PUBLISHED",
            },
            "EVIDENCE_READY": {"EVIDENCE_READY"},
            "ACCEPTED": {"ACCEPTED"},
        }
        if checkpoint not in allowed_checkpoints[str(value["phase"])]:
            raise RuntimeError(
                "pd-worldctl run state phase/checkpoint is invalid"
            )
        if not str(value.get("restart_run_id") or ""):
            raise RuntimeError("pd-worldctl run state id is invalid")
        journal = value.setdefault("observation_journal", [])
        if not isinstance(journal, list):
            raise RuntimeError(
                "pd-worldctl observation_journal must be a list"
            )
        return value
    value = {
        **identity,
        "restart_run_id": uuid.uuid4().hex,
        "phase": "CREATED",
        "checkpoint": "CREATED",
        "evidence": None,
        "startup_permit": None,
        "accepted_response": None,
        "protocol_events": [],
        "observation_journal": [],
        "termination_proofs": {},
    }
    _atomic_write_json(path, value)
    return value


def _load_proofs(path: Path) -> list[TerminationProof]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("proof file must contain a JSON list")
    return [TerminationProof(**item) for item in value]


def _load_old_world(path: Path) -> dict[str, ProcessIdentity]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("old world file must contain a JSON list")
    result = {str(item["member"]): ProcessIdentity(**item) for item in value}
    if set(result) != set(MEMBERS):
        raise ValueError("old world file must contain four members")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("command", choices=["initialize", "restart"])
    parser.add_argument("--release", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--context", default="")
    parser.add_argument("--old-generation", type=_canonical_uuid)
    parser.add_argument("--generation", type=_canonical_uuid, required=True)
    parser.add_argument("--termination-proofs", type=Path)
    parser.add_argument("--old-world-identities", type=Path)
    parser.add_argument("--chart", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gateway-url")
    parser.add_argument("--run-state", type=Path)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--gateway-read-timeout-s",
        type=float,
        help=(
            "bounded pre-mutation Gateway GET budget; defaults to --timeout-s"
        ),
    )
    parser.add_argument(
        "--inject-worker-crash",
        choices=MEMBERS,
        help=(
            "SIGKILL the identity-bound app PID in the selected captured worker "
            "before the guarded whole-world stop; requires --execute"
        ),
    )
    parser.add_argument(
        "--expect-nccl-watchdog",
        choices=MEMBERS,
        help=(
            "wait for the selected captured worker to exit itself with the "
            "exact NCCL watchdog termination message before stopping the world"
        ),
    )
    parser.add_argument(
        "--required-old-operation-id",
        action="append",
        default=[],
        help="Gateway tracker operation/request ID that replacement must exclude",
    )
    parser.add_argument(
        "--expected-operation-id",
        action="append",
        default=[],
        help="exact checkpoint operation ID required in live old-world reports",
    )
    args = parser.parse_args()
    if args.command == "initialize":
        if args.old_generation is not None:
            parser.error("initialize does not accept --old-generation")
        if (
            args.termination_proofs is not None
            or args.old_world_identities is not None
            or args.inject_worker_crash is not None
            or args.expect_nccl_watchdog is not None
            or args.gateway_read_timeout_s is not None
            or args.required_old_operation_id
            or args.expected_operation_id
        ):
            parser.error("initialize does not accept restart-only options")
        if not args.execute:
            parser.error("initialize requires --execute")
        if not args.gateway_url or args.run_state is None:
            parser.error(
                "initialize requires --gateway-url and --run-state"
            )
        result = WorldInitializeActuator(KubectlGatewayBackend(
            args.gateway_url, timeout_s=args.timeout_s, context=args.context
        )).initialize(
            release=args.release,
            namespace=args.namespace,
            chart=args.chart,
            generation=args.generation,
            run_state_path=args.run_state,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.old_generation:
        parser.error("restart requires --old-generation")
    if (
        args.inject_worker_crash is not None
        and args.expect_nccl_watchdog is not None
    ):
        parser.error(
            "--inject-worker-crash and --expect-nccl-watchdog are exclusive"
        )
    if args.execute:
        if not args.gateway_url or args.run_state is None:
            parser.error("--execute requires --gateway-url and --run-state")
        result = WorldRestartActuator(KubectlGatewayBackend(
            args.gateway_url,
            timeout_s=args.timeout_s,
            context=args.context,
            gateway_read_timeout_s=args.gateway_read_timeout_s,
        )).restart(
            release=args.release, namespace=args.namespace, chart=args.chart,
            old_generation=args.old_generation, new_generation=args.generation,
            run_state_path=args.run_state,
            fault_member=args.inject_worker_crash,
            watchdog_member=args.expect_nccl_watchdog,
            expected_operation_ids=tuple(args.expected_operation_id),
            required_old_operation_ids=tuple(args.required_old_operation_id),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.termination_proofs is None or args.old_world_identities is None:
        parser.error(
            "dry-run requires --termination-proofs and --old-world-identities"
        )
    if args.inject_worker_crash is not None:
        parser.error("--inject-worker-crash requires --execute")
    if args.expect_nccl_watchdog is not None:
        parser.error("--expect-nccl-watchdog requires --execute")
    if args.gateway_read_timeout_s is not None:
        parser.error("--gateway-read-timeout-s requires --execute")
    guard = WorldRestartGuard(
        args.old_generation, args.generation,
        _load_old_world(args.old_world_identities),
    )
    for proof in _load_proofs(args.termination_proofs):
        guard.record_termination(proof)
    commands = [
        *guard.scale_down_commands(args.release, args.namespace),
        *guard.patch_generation_commands(args.release, args.namespace, args.chart),
        *guard.scale_up_commands(args.release, args.namespace),
    ]
    print(json.dumps(commands, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

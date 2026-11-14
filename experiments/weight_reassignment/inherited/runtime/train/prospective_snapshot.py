"""Atomic snapshot and epoch-reset contracts for the prospective study."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def canonical_snapshot_batches(interact_times, event_priorities, edge_ids=None):
    """Group events by time and canonically order every tied snapshot."""

    times = np.asarray(interact_times)
    priorities = np.asarray(event_priorities, dtype=np.int64)
    edges = priorities.copy() if edge_ids is None else np.asarray(edge_ids, dtype=np.int64)
    if times.ndim != 1 or priorities.ndim != 1 or edges.ndim != 1:
        raise ValueError("Snapshot identity arrays must be one-dimensional")
    if not (len(times) == len(priorities) == len(edges)):
        raise ValueError("Snapshot identity arrays must have equal length")
    if not np.isfinite(np.asarray(times, dtype=np.float64)).all():
        raise ValueError("Snapshot times must be finite")
    if np.any(priorities <= 0) or np.any(edges <= 0):
        raise ValueError("Event priorities and edge IDs must be positive")
    if len(np.unique(priorities)) != len(priorities) or len(np.unique(edges)) != len(edges):
        raise ValueError("Event priorities and edge IDs must be globally unique")
    order = np.lexsort((edges, priorities, times))
    ordered_times = times[order]
    batches = []
    for timestamp in np.unique(ordered_times):
        batches.append(order[ordered_times == timestamp].astype(np.int64, copy=False))
    return batches


def canonical_snapshot_order(interact_times, event_priorities, edge_ids=None):
    """Flatten :func:`canonical_snapshot_batches` into one canonical stream."""

    batches = canonical_snapshot_batches(interact_times, event_priorities, edge_ids=edge_ids)
    return np.concatenate(batches) if batches else np.empty(0, dtype=np.int64)


def _timestamp_scalar(value) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Message timestamps must be scalar")
    result = float(array.reshape(()))
    if not np.isfinite(result):
        raise ValueError("Message timestamps must be finite")
    return result


def _event_key_bytes(value: Any) -> bytes:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("Message event keys must be canonically serializable") from error


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    detached = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(np.asarray(detached.shape, dtype=np.int64).tobytes())
    digest.update(detached.numpy().tobytes(order="C"))
    return digest.digest()


def mean_message_records(records):
    """Canonically average one node's messages from one atomic snapshot.

    Records may be historical ``(message, timestamp)`` tuples or prospective
    ``(message, timestamp, event_key)`` tuples.  Explicit event keys are
    preferred.  The tensor digest fallback provides deterministic summation for
    legacy records without detaching the tensors used in the mean.
    """

    records = list(records)
    if not records:
        raise ValueError("Cannot aggregate an empty message list")
    parsed = []
    timestamp = None
    shape = dtype = device = None
    seen_explicit_keys = set()
    for record in records:
        if not isinstance(record, (tuple, list)) or len(record) not in {2, 3}:
            raise ValueError("Message records must have two or three fields")
        message, raw_timestamp = record[:2]
        if not isinstance(message, torch.Tensor) or not message.is_floating_point():
            raise TypeError("Messages must be floating-point torch tensors")
        current_timestamp = _timestamp_scalar(raw_timestamp)
        timestamp = current_timestamp if timestamp is None else timestamp
        if current_timestamp != timestamp:
            raise ValueError("Messages for one atomic update must share a timestamp")
        if shape is None:
            shape, dtype, device = tuple(message.shape), message.dtype, message.device
        elif tuple(message.shape) != shape or message.dtype != dtype or message.device != device:
            raise ValueError("Messages in one atomic mean must share shape, dtype and device")
        if len(record) == 3:
            key = b"event:" + _event_key_bytes(record[2])
            if key in seen_explicit_keys:
                raise ValueError("Explicit message event keys must be unique per node and snapshot")
            seen_explicit_keys.add(key)
            # Registered edge IDs are unique, so no tensor materialization or
            # device synchronization is needed to establish a total order.
            sort_key = (0, key)
        else:
            sort_key = (1, _tensor_digest(message))
        parsed.append((sort_key, message))
    parsed.sort(key=lambda item: item[0])
    return torch.stack([item[1] for item in parsed], dim=0).mean(dim=0), timestamp


class AtomicMeanMessageAggregator(nn.Module):
    """Drop-in TGN message aggregator using one mean update per node/snapshot."""

    def __init__(self):
        super().__init__()

    def aggregate_messages(self, node_ids, node_raw_messages):
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if node_ids.ndim != 1:
            raise ValueError("node_ids must be one-dimensional")
        update_ids, messages, timestamps = [], [], []
        for node_id in np.unique(node_ids):
            records = node_raw_messages.get(int(node_id), [])
            if records:
                message, timestamp = mean_message_records(records)
                update_ids.append(int(node_id))
                messages.append(message)
                timestamps.append(timestamp)
        if not messages:
            return np.empty(0, dtype=np.int64), torch.empty(0), np.empty(0, dtype=np.float64)
        return (
            np.asarray(update_ids, dtype=np.int64),
            torch.stack(messages, dim=0),
            np.asarray(timestamps, dtype=np.float64),
        )


# Short alias for installer/runtime code.
MeanMessageAggregator = AtomicMeanMessageAggregator


def _resolve_memory_bank(memory_owner):
    if hasattr(memory_owner, "node_memories") and hasattr(memory_owner, "node_raw_messages"):
        return memory_owner
    if hasattr(memory_owner, "memory_bank"):
        return memory_owner.memory_bank
    try:
        first = memory_owner[0]
    except (TypeError, KeyError, IndexError):
        first = None
    if first is not None and hasattr(first, "memory_bank"):
        return first.memory_bank
    raise TypeError("Could not resolve a TGN memory bank from the supplied object")


def assert_epoch_state_reset(memory_owner) -> None:
    """Fail unless memories, clocks and pending messages are at epoch zero."""

    memory_bank = _resolve_memory_bank(memory_owner)
    if bool(torch.count_nonzero(memory_bank.node_memories.detach()).item()):
        raise RuntimeError("Node memories were not reset to zero")
    if bool(torch.count_nonzero(memory_bank.node_last_updated_times.detach()).item()):
        raise RuntimeError("Node update times were not reset to zero")
    nonempty = {
        int(node_id): len(records)
        for node_id, records in memory_bank.node_raw_messages.items()
        if len(records)
    }
    if nonempty:
        raise RuntimeError(f"Pending node messages survived the epoch reset: {nonempty}")


@dataclass(frozen=True)
class EpochResetReport:
    epoch: int
    memories_zero: bool
    update_times_zero: bool
    pending_message_count: int
    negative_schedule_position: tuple[int, int] | None


def reset_epoch_state(
    memory_owner,
    *,
    epoch: int,
    negative_sampler=None,
    neighbor_sampler=None,
) -> EpochResetReport:
    """Apply and verify the prospective epoch-reset contract.

    Model parameters and optimizer state are intentionally untouched.  Only TGN
    memory, last-update clocks, pending messages, and explicitly epoch-scoped
    sampler state participate in this reset.
    """

    if not isinstance(epoch, (int, np.integer)) or int(epoch) < 0:
        raise ValueError("epoch must be a non-negative integer")
    memory_bank = _resolve_memory_bank(memory_owner)
    initializer = getattr(memory_bank, "__init_memory_bank__", None)
    if not callable(initializer):
        raise TypeError("Memory bank does not expose the audited reset method")
    initializer()
    if neighbor_sampler is not None:
        reset_epoch = getattr(neighbor_sampler, "reset_epoch", None)
        if callable(reset_epoch):
            reset_epoch(int(epoch))
        elif getattr(neighbor_sampler, "sample_neighbor_strategy", None) != "recent":
            raise TypeError("Mutable neighbor samplers require an explicit reset_epoch contract")
    if negative_sampler is not None:
        begin_epoch = getattr(negative_sampler, "begin_epoch", None)
        if not callable(begin_epoch):
            raise TypeError("Prospective negative sampler must expose begin_epoch")
        begin_epoch(int(epoch))
    assert_epoch_state_reset(memory_bank)
    position = None if negative_sampler is None else tuple(negative_sampler.position)
    return EpochResetReport(
        epoch=int(epoch),
        memories_zero=True,
        update_times_zero=True,
        pending_message_count=0,
        negative_schedule_position=position,
    )


def memory_state_fingerprint(memory_owner) -> str:
    """Hash memory, clocks and canonically ordered pending messages."""

    memory_bank = _resolve_memory_bank(memory_owner)
    digest = hashlib.sha256()
    for name, tensor in (
        ("node_memories", memory_bank.node_memories),
        ("node_last_updated_times", memory_bank.node_last_updated_times),
    ):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    for node_id in sorted(memory_bank.node_raw_messages):
        records = memory_bank.node_raw_messages[node_id]
        digest.update(int(node_id).to_bytes(8, "little", signed=True))
        normalized = []
        for record in records:
            timestamp = _timestamp_scalar(record[1])
            # Use a string sentinel so mixed legacy/prospective records remain
            # sortable when their timestamps happen to be equal.
            event_key = "" if len(record) == 2 else _event_key_bytes(record[2]).hex()
            normalized.append((timestamp, event_key, _tensor_digest(record[0]).hex()))
        for item in sorted(normalized):
            digest.update(_event_key_bytes(item))
    return digest.hexdigest()

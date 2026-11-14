"""Deterministic prospective-study temporal controls.

The historical Stage C2 permutation deliberately avoided unchanged assignments.
That is useful as an intervention, but it is not a draw from the uniform
permutation distribution.  The prospective ``P`` arm uses the functions here:
one query-keyed uniform draw over valid neighbors, including the identity draw.

This module intentionally has no dependency on the model implementation.  It
returns gather indices which the existing temporal gate can apply to any tensor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


ALGORITHM_VERSION = "prospective-uniform-row-permutation-v1"
_UINT64_RANGE = 1 << 64


def _json_value(value: Any) -> Any:
    """Convert common scalar/container types to a canonical JSON value."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Permutation identities may not contain non-finite floats")
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported permutation context value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


class _HashStream:
    """Deterministic byte stream keyed by a complete query identity."""

    def __init__(self, key: bytes):
        self._key = key
        self._counter = 0
        self._buffer = b""

    def _fill(self) -> None:
        block = hashlib.blake2b(
            self._key + self._counter.to_bytes(8, "little"),
            digest_size=64,
            person=b"TGL-OPT-A-P1",
        ).digest()
        self._counter += 1
        self._buffer += block

    def uint64(self) -> int:
        while len(self._buffer) < 8:
            self._fill()
        raw, self._buffer = self._buffer[:8], self._buffer[8:]
        return int.from_bytes(raw, "little", signed=False)

    def randbelow(self, upper: int) -> int:
        """Return an exactly uniform integer in ``range(upper)``.

        Rejection sampling avoids the small modulo bias that would otherwise
        remain when ``upper`` does not divide 2**64.
        """

        if not isinstance(upper, int) or upper <= 0:
            raise ValueError("upper must be a positive integer")
        limit = _UINT64_RANGE - (_UINT64_RANGE % upper)
        while True:
            value = self.uint64()
            if value < limit:
                return value % upper


def _uniform_permutation(length: int, key: bytes) -> np.ndarray:
    """Fisher-Yates draw driven by an unbiased hash stream."""

    result = np.arange(length, dtype=np.int64)
    stream = _HashStream(key)
    for right in range(length - 1, 0, -1):
        left = stream.randbelow(right + 1)
        result[right], result[left] = result[left], result[right]
    return result


def uniform_row_permutations(
    query_node_ids,
    query_times,
    neighbor_ids,
    edge_ids,
    gaps,
    *,
    layer: int,
    control_seed: int | str,
    context: Mapping[str, Any],
) -> np.ndarray:
    """Return query-keyed uniform gather indices for valid neighbor rows.

    Valid positions are those with ``neighbor_id > 0``.  Padding maps to itself.
    Every valid-row permutation, including identity, has probability ``1/n!``
    under the keyed random-oracle construction.  There is no redraw or cyclic
    fallback.  Canonical event ordering makes the assignment independent of
    batch order and neighbor-slot order.

    ``context`` is mandatory and becomes part of the key.  Callers should bind
    at least the graph, training block/history, arm or replay condition, and
    control replicate.  Valid edge IDs must be positive and unique within a row;
    ambiguity is rejected instead of being resolved with slot order.
    """

    if not isinstance(context, Mapping):
        raise TypeError("context must be a mapping and is required in the permutation key")
    if not isinstance(layer, (int, np.integer)) or int(layer) < 0:
        raise ValueError("layer must be a non-negative integer")

    query_node_ids = np.asarray(query_node_ids)
    query_times = np.asarray(query_times)
    neighbor_ids = np.asarray(neighbor_ids)
    edge_ids = np.asarray(edge_ids)
    gaps = np.asarray(gaps)

    if neighbor_ids.ndim != 2:
        raise ValueError("neighbor_ids must be a two-dimensional array")
    if edge_ids.shape != neighbor_ids.shape or gaps.shape != neighbor_ids.shape:
        raise ValueError("neighbor_ids, edge_ids and gaps must have identical shapes")
    if query_node_ids.ndim != 1 or query_times.ndim != 1:
        raise ValueError("query_node_ids and query_times must be one-dimensional")
    if len(query_node_ids) != len(neighbor_ids) or len(query_times) != len(neighbor_ids):
        raise ValueError("Every neighborhood row requires one query node and query time")
    if not np.isfinite(np.asarray(query_times, dtype=np.float64)).all():
        raise ValueError("query_times must be finite")
    if not np.isfinite(np.asarray(gaps, dtype=np.float64)).all():
        raise ValueError("gaps must be finite")

    output = np.broadcast_to(
        np.arange(neighbor_ids.shape[1], dtype=np.int64), neighbor_ids.shape
    ).copy()
    # Absolute clock origin is transport metadata, not part of the scientific
    # control map.  Binding the relative query time while excluding this one
    # transport field lets the registered T1 canary shift every clock by a
    # constant without silently changing permutation indices.
    clock_origin = float(context.get("clock_origin", 0.0))
    if not np.isfinite(clock_origin):
        raise ValueError("context clock_origin must be finite")
    keyed_context = {key: value for key, value in context.items() if key != "clock_origin"}
    shared_identity = {
        "algorithm": ALGORITHM_VERSION,
        "control_seed": control_seed,
        "context": keyed_context,
        "layer": int(layer),
    }

    for row_index in range(len(neighbor_ids)):
        valid_positions = np.flatnonzero(neighbor_ids[row_index] > 0)
        if len(valid_positions) == 0:
            continue

        valid_edge_ids = np.asarray(edge_ids[row_index, valid_positions], dtype=np.int64)
        if np.any(valid_edge_ids <= 0):
            raise ValueError("Every valid neighbor must have a positive edge ID")
        if len(np.unique(valid_edge_ids)) != len(valid_edge_ids):
            raise ValueError("Valid edge IDs must be unique within each query row")

        canonical_positions = sorted(
            (int(position) for position in valid_positions),
            key=lambda position: (
                int(edge_ids[row_index, position]),
                int(neighbor_ids[row_index, position]),
                float(gaps[row_index, position]).hex(),
            ),
        )
        events = [
            {
                "edge_id": int(edge_ids[row_index, position]),
                "neighbor_id": int(neighbor_ids[row_index, position]),
                "gap": float(gaps[row_index, position]),
            }
            for position in canonical_positions
        ]
        row_identity = {
            **shared_identity,
            "query_node_id": int(query_node_ids[row_index]),
            "query_time_relative_to_context_clock": (
                float(query_times[row_index]) - clock_origin
            ),
            "events": events,
        }
        draw = _uniform_permutation(len(canonical_positions), _canonical_bytes(row_identity))
        canonical_array = np.asarray(canonical_positions, dtype=np.int64)
        output[row_index, canonical_array] = canonical_array[draw]

    return output


# Descriptive alias used in protocol prose and downstream audit code.
uniform_valid_row_permutations = uniform_row_permutations


@dataclass(frozen=True)
class PermutationDiagnostics:
    rows: int
    valid_rows: int
    singleton_rows: int
    identity_rows: int
    changed_rows: int


def validate_row_permutations(neighbor_ids, gather_indices) -> PermutationDiagnostics:
    """Fail closed unless each row permutes valid slots and fixes padding."""

    neighbor_ids = np.asarray(neighbor_ids)
    gather_indices = np.asarray(gather_indices)
    if neighbor_ids.ndim != 2 or gather_indices.shape != neighbor_ids.shape:
        raise ValueError("neighbor_ids and gather_indices must be equal two-dimensional shapes")

    valid_rows = singleton_rows = identity_rows = changed_rows = 0
    for row in range(len(neighbor_ids)):
        valid = np.flatnonzero(neighbor_ids[row] > 0)
        padding = np.flatnonzero(neighbor_ids[row] <= 0)
        if not np.array_equal(gather_indices[row, padding], padding):
            raise ValueError("Padding positions must map to themselves")
        if not np.array_equal(np.sort(gather_indices[row, valid]), valid):
            raise ValueError("Valid positions must be permuted exactly once")
        if len(valid):
            valid_rows += 1
            singleton_rows += int(len(valid) == 1)
            if np.array_equal(gather_indices[row, valid], valid):
                identity_rows += 1
            else:
                changed_rows += 1
    return PermutationDiagnostics(
        rows=len(neighbor_ids),
        valid_rows=valid_rows,
        singleton_rows=singleton_rows,
        identity_rows=identity_rows,
        changed_rows=changed_rows,
    )

"""Prospective-study negative schedules and canonical recent neighbors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROTOCOL_NODE_COUNT = 128
PROTOCOL_EPOCHS = 300
PROTOCOL_SNAPSHOTS = 96
PROTOCOL_POSITIVES_PER_SNAPSHOT = 160


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_hash(named_arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in named_arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def positive_matrices_from_stream(src_node_ids, dst_node_ids, interact_times):
    """Reshape a chronological positive stream into equal-width snapshots."""

    src = np.asarray(src_node_ids)
    dst = np.asarray(dst_node_ids)
    times = np.asarray(interact_times)
    if src.ndim != 1 or dst.ndim != 1 or times.ndim != 1:
        raise ValueError("Positive stream arrays must be one-dimensional")
    if not (len(src) == len(dst) == len(times)) or len(src) == 0:
        raise ValueError("Positive stream arrays must be nonempty and equal length")
    if not np.isfinite(np.asarray(times, dtype=np.float64)).all():
        raise ValueError("Positive times must be finite")
    unique_times = np.unique(times)
    rows_src, rows_dst = [], []
    width = None
    for timestamp in unique_times:
        positions = np.flatnonzero(times == timestamp)
        width = len(positions) if width is None else width
        if len(positions) != width:
            raise ValueError("Every prospective training snapshot must have equal width")
        rows_src.append(np.asarray(src[positions], dtype=np.int64))
        rows_dst.append(np.asarray(dst[positions], dtype=np.int64))
    return np.stack(rows_src), np.stack(rows_dst), np.asarray(unique_times)


@dataclass(frozen=True)
class NegativeSchedule:
    negative_dst: np.ndarray
    positive_src: np.ndarray
    positive_dst: np.ndarray
    snapshot_times: np.ndarray
    node_ids: np.ndarray
    file_sha256: str
    logical_sha256: str

    @property
    def epochs(self) -> int:
        return int(self.negative_dst.shape[0])

    @property
    def snapshots(self) -> int:
        return int(self.negative_dst.shape[1])

    @property
    def positives_per_snapshot(self) -> int:
        return int(self.negative_dst.shape[2])


def _optional_embedded(archive, name):
    return np.asarray(archive[name]) if name in archive.files else None


def load_negative_schedule(
    path,
    *,
    positive_src=None,
    positive_dst=None,
    snapshot_times=None,
    node_ids=None,
    expected_epochs: int | None = PROTOCOL_EPOCHS,
    expected_snapshots: int | None = PROTOCOL_SNAPSHOTS,
    expected_positives_per_snapshot: int | None = PROTOCOL_POSITIVES_PER_SNAPSHOT,
    expected_file_sha256: str | None = None,
) -> NegativeSchedule:
    """Load and fully validate a compact, source-preserving schedule.

    ``.npy`` files contain only the ``negative_dst`` cube and therefore require
    all reference arrays as arguments.  ``.npz`` files may additionally embed
    ``positive_src``, ``positive_dst``, ``snapshot_times`` and ``node_ids``.
    Object arrays/pickles are never accepted.
    """

    path = Path(path).resolve()
    if not path.is_file() or path.suffix not in {".npy", ".npz"}:
        raise ValueError("Negative schedule must be an existing .npy or .npz file")
    file_sha256 = _sha256_file(path)
    if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
        raise ValueError("Negative schedule file hash does not match the frozen manifest")

    embedded = {}
    if path.suffix == ".npy":
        negative_dst = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as archive:
            if "negative_dst" not in archive.files:
                raise ValueError("Schedule archive is missing negative_dst")
            negative_dst = np.asarray(archive["negative_dst"])
            for name in ("positive_src", "positive_dst", "snapshot_times", "node_ids"):
                embedded[name] = _optional_embedded(archive, name)

    if negative_dst.ndim != 3 or not np.issubdtype(negative_dst.dtype, np.integer):
        raise ValueError("negative_dst must be a three-dimensional integer cube")
    if negative_dst.dtype.itemsize > 2:
        raise ValueError("A 128-node negative schedule must use a compact <=16-bit dtype")
    epochs, snapshots, width = map(int, negative_dst.shape)
    for observed, expected, name in (
        (epochs, expected_epochs, "epochs"),
        (snapshots, expected_snapshots, "snapshots"),
        (width, expected_positives_per_snapshot, "positives per snapshot"),
    ):
        if expected is not None and observed != int(expected):
            raise ValueError(f"Schedule has {observed} {name}; expected {expected}")

    def resolve_reference(name, supplied):
        embedded_value = embedded.get(name)
        if supplied is None and embedded_value is None:
            raise ValueError(f"Schedule requires a frozen {name} reference")
        supplied_value = None if supplied is None else np.asarray(supplied)
        if supplied_value is not None and embedded_value is not None:
            if not np.array_equal(supplied_value, embedded_value):
                raise ValueError(f"Embedded and supplied {name} references disagree")
        return embedded_value if supplied_value is None else supplied_value

    positive_src = np.asarray(resolve_reference("positive_src", positive_src), dtype=np.int64)
    positive_dst = np.asarray(resolve_reference("positive_dst", positive_dst), dtype=np.int64)
    snapshot_times = np.asarray(resolve_reference("snapshot_times", snapshot_times))
    if node_ids is None and embedded.get("node_ids") is None:
        node_ids = np.arange(1, PROTOCOL_NODE_COUNT + 1, dtype=np.int64)
    else:
        node_ids = np.asarray(resolve_reference("node_ids", node_ids), dtype=np.int64)

    if positive_src.shape != (snapshots, width) or positive_dst.shape != (snapshots, width):
        raise ValueError("Positive references must match schedule snapshot and row dimensions")
    if snapshot_times.shape != (snapshots,) or not np.isfinite(
        np.asarray(snapshot_times, dtype=np.float64)
    ).all():
        raise ValueError("snapshot_times must contain one finite value per snapshot")
    if node_ids.ndim != 1 or len(node_ids) != PROTOCOL_NODE_COUNT:
        raise ValueError("The prospective schedule must bind exactly 128 node IDs")
    if len(np.unique(node_ids)) != len(node_ids) or np.any(node_ids <= 0):
        raise ValueError("Node IDs must be unique and positive")
    if not np.all(np.isin(positive_src, node_ids)) or not np.all(np.isin(positive_dst, node_ids)):
        raise ValueError("Positive references contain IDs outside the node universe")
    if not np.all(np.isin(negative_dst, node_ids)):
        raise ValueError("Negative schedule contains IDs outside the node universe")

    # The source of each negative is implicit: it is the aligned positive source.
    if np.any(negative_dst == positive_src[None, :, :]):
        raise ValueError("Negative schedule contains a self-edge")
    for snapshot in range(snapshots):
        for source in np.unique(positive_src[snapshot]):
            positions = np.flatnonzero(positive_src[snapshot] == source)
            forbidden = np.unique(positive_dst[snapshot, positions])
            if np.isin(negative_dst[:, snapshot, positions], forbidden).any():
                raise ValueError(
                    "Negative schedule contains a current-snapshot positive destination"
                )

    negative_dst = np.ascontiguousarray(negative_dst)
    positive_src = np.ascontiguousarray(positive_src)
    positive_dst = np.ascontiguousarray(positive_dst)
    snapshot_times = np.ascontiguousarray(snapshot_times)
    node_ids = np.ascontiguousarray(node_ids)
    for array in (negative_dst, positive_src, positive_dst, snapshot_times, node_ids):
        array.flags.writeable = False
    logical_sha256 = _logical_hash(
        (
            ("negative_dst", negative_dst),
            ("positive_src", positive_src),
            ("positive_dst", positive_dst),
            ("snapshot_times", snapshot_times),
            ("node_ids", node_ids),
        )
    )
    return NegativeSchedule(
        negative_dst=negative_dst,
        positive_src=positive_src,
        positive_dst=positive_dst,
        snapshot_times=snapshot_times,
        node_ids=node_ids,
        file_sha256=file_sha256,
        logical_sha256=logical_sha256,
    )


class FrozenNegativeScheduleSampler:
    """Sequential adapter compatible with ``NegativeEdgeSampler.sample``."""

    negative_sample_strategy = "prospective_fixed"
    seed = None
    doubled = False

    def __init__(self, schedule_path, **loader_kwargs):
        self.schedule = load_negative_schedule(schedule_path, **loader_kwargs)
        self._epoch = 0
        self._snapshot = 0

    @property
    def position(self) -> tuple[int, int]:
        return self._epoch, self._snapshot

    @property
    def schedule_sha256(self) -> str:
        return self.schedule.logical_sha256

    def begin_epoch(self, epoch: int) -> None:
        """Assert, rather than alter, the expected sequential epoch boundary."""

        expected = (int(epoch), 0)
        if self.position != expected:
            raise RuntimeError(
                f"Negative schedule cursor is {self.position}; expected epoch boundary {expected}"
            )
        if not 0 <= int(epoch) < self.schedule.epochs:
            raise ValueError("Epoch index is outside the frozen negative schedule")

    def sample(
        self,
        size: int,
        batch_src_node_ids=None,
        batch_dst_node_ids=None,
        current_batch_start_time: float = 0.0,
        current_batch_end_time: float = 0.0,
        batch_times=None,
    ):
        del current_batch_start_time, current_batch_end_time
        if self._epoch >= self.schedule.epochs:
            raise RuntimeError("Frozen negative schedule is exhausted")
        if int(size) != self.schedule.positives_per_snapshot:
            raise ValueError("Requested negative count does not equal the frozen snapshot width")
        if batch_src_node_ids is None or batch_dst_node_ids is None or batch_times is None:
            raise ValueError("Frozen schedules require positive source, destination and time alignment")

        observed_src = np.asarray(batch_src_node_ids, dtype=np.int64)
        observed_dst = np.asarray(batch_dst_node_ids, dtype=np.int64)
        observed_times = np.asarray(batch_times)
        expected_src = self.schedule.positive_src[self._snapshot]
        expected_dst = self.schedule.positive_dst[self._snapshot]
        expected_time = self.schedule.snapshot_times[self._snapshot]
        if observed_src.shape != expected_src.shape or not np.array_equal(observed_src, expected_src):
            raise ValueError("Positive source order does not match the frozen schedule")
        if observed_dst.shape != expected_dst.shape or not np.array_equal(observed_dst, expected_dst):
            raise ValueError("Positive destination order does not match the frozen schedule")
        if observed_times.shape != expected_src.shape or not np.all(observed_times == expected_time):
            raise ValueError("Positive time/order does not match the frozen schedule")

        negative_dst = self.schedule.negative_dst[self._epoch, self._snapshot].astype(
            np.int64, copy=True
        )
        negative_src = expected_src.copy()
        self._snapshot += 1
        if self._snapshot == self.schedule.snapshots:
            self._epoch += 1
            self._snapshot = 0
        return negative_src, negative_dst

    def assert_exhausted(self) -> None:
        if self.position != (self.schedule.epochs, 0):
            raise RuntimeError(f"Negative schedule stopped early at {self.position}")

    def reset_random_state(self) -> None:
        """Compatibility reset; partial schedules may not be silently rewound."""

        if self.position not in {(0, 0), (self.schedule.epochs, 0)}:
            raise RuntimeError("Refusing to rewind a partially consumed frozen schedule")
        self._epoch = 0
        self._snapshot = 0


def canonical_sort_key(timestamp, event_priority_or_edge_id):
    """Stable primary key for tied temporal interactions."""

    timestamp = float(timestamp)
    priority = int(event_priority_or_edge_id)
    if not np.isfinite(timestamp):
        raise ValueError("Interaction timestamps must be finite")
    if priority <= 0:
        raise ValueError("Event priorities/edge IDs must be positive")
    return timestamp, priority


def build_canonical_adjacency(
    src_node_ids,
    dst_node_ids,
    edge_ids,
    interact_times,
    *,
    event_priorities=None,
    num_nodes: int = PROTOCOL_NODE_COUNT,
    undirected: bool = True,
):
    """Build file-order-independent adjacency with explicit tie priorities."""

    src = np.asarray(src_node_ids, dtype=np.int64)
    dst = np.asarray(dst_node_ids, dtype=np.int64)
    edge = np.asarray(edge_ids, dtype=np.int64)
    times = np.asarray(interact_times)
    priority = edge.copy() if event_priorities is None else np.asarray(event_priorities, dtype=np.int64)
    if any(array.ndim != 1 for array in (src, dst, edge, times, priority)):
        raise ValueError("Event arrays must be one-dimensional")
    if not (len(src) == len(dst) == len(edge) == len(times) == len(priority)):
        raise ValueError("Event arrays must have equal lengths")
    if len(np.unique(edge)) != len(edge) or len(np.unique(priority)) != len(priority):
        raise ValueError("Edge IDs and event priorities must be globally unique")
    if not np.isfinite(np.asarray(times, dtype=np.float64)).all():
        raise ValueError("Interaction timestamps must be finite")
    if np.any(src <= 0) or np.any(dst <= 0) or np.any(src > num_nodes) or np.any(dst > num_nodes):
        raise ValueError("Event node IDs are outside the declared graph")
    if np.any(src == dst):
        raise ValueError("Prospective graphs may not contain self-edges")

    adjacency = [[] for _ in range(int(num_nodes) + 1)]
    for source, destination, edge_id, timestamp, event_priority in zip(
        src, dst, edge, times, priority
    ):
        record = (int(destination), int(edge_id), float(timestamp), int(event_priority))
        adjacency[int(source)].append(record)
        if undirected:
            reverse = (int(source), int(edge_id), float(timestamp), int(event_priority))
            adjacency[int(destination)].append(reverse)
    return adjacency


class CanonicalRecentNeighborSampler:
    """Strict-past recent sampler with deterministic same-time ordering."""

    sample_neighbor_strategy = "recent"
    seed = None

    def __init__(self, adj_list: Sequence[Sequence[Sequence[float | int]]]):
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []
        self.nodes_event_priorities = []
        for neighbors in adj_list:
            parsed = []
            seen_edge_ids = set()
            for record in neighbors:
                if len(record) not in {3, 4}:
                    raise ValueError("Adjacency records must have 3 or 4 entries")
                neighbor, edge_id, timestamp = record[:3]
                priority = edge_id if len(record) == 3 else record[3]
                canonical_sort_key(timestamp, priority)
                if int(neighbor) <= 0 or int(edge_id) <= 0:
                    raise ValueError("Real neighbors and edge IDs must be positive")
                if int(edge_id) in seen_edge_ids:
                    raise ValueError("An edge may appear only once in each node adjacency")
                seen_edge_ids.add(int(edge_id))
                parsed.append((int(neighbor), int(edge_id), float(timestamp), int(priority)))
            parsed.sort(key=lambda item: (*canonical_sort_key(item[2], item[3]), item[0], item[1]))
            self.nodes_neighbor_ids.append(np.asarray([item[0] for item in parsed], dtype=np.int64))
            self.nodes_edge_ids.append(np.asarray([item[1] for item in parsed], dtype=np.int64))
            self.nodes_neighbor_times.append(np.asarray([item[2] for item in parsed], dtype=np.float64))
            self.nodes_event_priorities.append(np.asarray([item[3] for item in parsed], dtype=np.int64))

    def find_neighbors_before(self, node_id, interact_time, return_sampled_probabilities=False):
        node_id = int(node_id)
        if not 0 <= node_id < len(self.nodes_neighbor_ids):
            raise ValueError("node_id is outside the adjacency")
        interact_time = float(interact_time)
        if not np.isfinite(interact_time):
            raise ValueError("interact_time must be finite")
        stop = int(np.searchsorted(self.nodes_neighbor_times[node_id], interact_time, side="left"))
        probabilities = None
        if return_sampled_probabilities:
            probabilities = np.empty(stop, dtype=np.float64)
            probabilities.fill(np.nan)
        return (
            self.nodes_neighbor_ids[node_id][:stop],
            self.nodes_edge_ids[node_id][:stop],
            self.nodes_neighbor_times[node_id][:stop],
            probabilities,
        )

    def get_historical_neighbors(self, node_ids, node_interact_times, num_neighbors=20):
        if int(num_neighbors) <= 0:
            raise ValueError("num_neighbors must be positive")
        node_ids = np.asarray(node_ids, dtype=np.int64)
        node_interact_times = np.asarray(node_interact_times)
        if node_ids.ndim != 1 or node_interact_times.shape != node_ids.shape:
            raise ValueError("node_ids and node_interact_times must be equal one-dimensional arrays")
        count = int(num_neighbors)
        neighbors_out = np.zeros((len(node_ids), count), dtype=np.int64)
        edges_out = np.zeros((len(node_ids), count), dtype=np.int64)
        times_out = np.zeros((len(node_ids), count), dtype=np.float32)
        for row, (node_id, query_time) in enumerate(zip(node_ids, node_interact_times)):
            neighbors, edges, times, _ = self.find_neighbors_before(node_id, query_time)
            neighbors, edges, times = neighbors[-count:], edges[-count:], times[-count:]
            start = count - len(neighbors)
            neighbors_out[row, start:] = neighbors
            edges_out[row, start:] = edges
            times_out[row, start:] = times
        return neighbors_out, edges_out, times_out

    def get_multi_hop_neighbors(self, num_hops, node_ids, node_interact_times, num_neighbors=20):
        if int(num_hops) <= 0:
            raise ValueError("num_hops must be positive")
        node_ids = np.asarray(node_ids, dtype=np.int64)
        ids, edges, times = self.get_historical_neighbors(
            node_ids, node_interact_times, num_neighbors=num_neighbors
        )
        ids_list, edge_list, time_list = [ids], [edges], [times]
        for _ in range(1, int(num_hops)):
            ids, edges, times = self.get_historical_neighbors(
                ids_list[-1].reshape(-1), time_list[-1].reshape(-1), num_neighbors=num_neighbors
            )
            ids_list.append(ids.reshape(len(node_ids), -1))
            edge_list.append(edges.reshape(len(node_ids), -1))
            time_list.append(times.reshape(len(node_ids), -1))
        return ids_list, edge_list, time_list

    def get_all_first_hop_neighbors(self, node_ids, node_interact_times):
        result = [[], [], []]
        for node_id, query_time in zip(node_ids, node_interact_times):
            neighbors, edges, times, _ = self.find_neighbors_before(node_id, query_time)
            result[0].append(neighbors)
            result[1].append(edges)
            result[2].append(times)
        return tuple(result)

    def reset_random_state(self):
        """Compatibility no-op: this sampler has no mutable RNG state."""

        return None

"""RECONSTRUCTED MODULE (not part of the original dess-mannheim/tgl release).

The public repository imports `data.utils` from 15 call sites, but the module
was never committed (the datasets folder `data/*` is gitignored and the code
files under it appear to have been lost with it). This file reconstructs the
missing functionality:

* `Data`, `get_idx_data_loader` -- taken from DyGLib (yule-BUAA/DyGLib,
  utils/DataLoader.py), which this repository is explicitly "adapted from"
  (see repo README and module docstrings "Based on code from DyGLib").
* `load_data` -- reconstructed against its single call site in run_model.py
  and the on-disk formats written by generate/models/graph.py (synthetic:
  data/<family>/<config>/data_<seed>.csv + node_<seed>.npy/edge_<seed>.npy)
  and generate/empirical/preprocessing.py (empirical:
  data/empirical/<config>/<variant>/ml_<config>.csv + .npy/_node.npy).
* split conventions -- taken from the paper (arXiv:2510.09416v4) and the
  repository README:
    - empirical/sbm/pa: 70/15/15 by np.quantile(ts, [0.70, 0.85])
      (DyGLib convention; consistent with the paper's "T=100 training and
      21 validation time steps" for pa with 143 total steps),
    - periodic: final one-period block held out (README, Section
      "Periodicity or persistence"); train = first repeats*periodicity steps,
    - partition (recency): last two time steps held out (paper: "datasets of
      10 time steps ... predictions are made at time t=10" with 12 generated
      steps 0..11 -> train 0..9, val {10}, test {11}).
* small helpers (`repeat_or_validate`, `interval_size`, `time_scales`,
  `bipartite_check`, `get_timestamps`) -- reconstructed from their call sites
  in generate/empirical/preprocessing.py and generate/models/*.py. The
  discretisation scales reproduce Table 1 of the paper
  (Enron 45 monthly, Bitcoin-Alpha 63 monthly, UCI 29 weekly,
  Wikipedia 745 hourly steps).

If the upstream authors publish the original files
(https://github.com/dess-mannheim/tgl), diff them against this module and
prefer theirs.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants used by generate/empirical/preprocessing.py
# ---------------------------------------------------------------------------

#: seconds per discretisation interval
interval_size = {
    "hourly": 60 * 60,
    "daily": 24 * 60 * 60,
    "weekly": 7 * 24 * 60 * 60,
    "monthly": 30 * 24 * 60 * 60,
}

#: discretisation scale per empirical dataset (reproduces Table 1 of the paper)
time_scales = {
    "uci": "weekly",
    "wikipedia": "hourly",
    "enron": "monthly",
    "bcalpha": "monthly",
    "reddit": "daily",
}

#: datasets whose raw node ids are bipartite (JODIE-style user/item graphs)
bipartite_check = ["wikipedia", "reddit"]


REQUIRED_EDGE_COLUMNS = {"u", "i", "ts", "label", "idx"}


def get_data_root() -> Path:
    """Return the data root, optionally overridden for read-only validation."""
    configured = os.environ.get("TGL_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent


def _validate_graph_frame(graph_df: pd.DataFrame, source: Path) -> None:
    """Fail early when reconstructed inputs do not satisfy TGL's assumptions."""
    missing = REQUIRED_EDGE_COLUMNS - set(graph_df.columns)
    assert not missing, f"{source} is missing required columns: {sorted(missing)}"
    assert not graph_df.empty, f"{source} contains no interactions"
    assert graph_df[list(REQUIRED_EDGE_COLUMNS)].notna().all().all(), f"{source} contains null values"
    assert (graph_df["u"] > 0).all() and (graph_df["i"] > 0).all(), (
        f"{source} must reserve node id 0 for padding"
    )
    assert (graph_df["idx"] > 0).all(), f"{source} must reserve edge id 0 for padding"
    assert graph_df["idx"].is_unique, f"{source} contains duplicate edge ids"
    assert graph_df["ts"].is_monotonic_increasing, f"{source} must be sorted by timestamp"


def _validate_feature_alignment(
    graph_df: pd.DataFrame,
    node_raw_features: np.ndarray,
    edge_raw_features: np.ndarray,
    source: Path,
) -> None:
    assert max(graph_df["u"].max(), graph_df["i"].max()) < len(node_raw_features), (
        f"{source}: node ids exceed the node feature matrix"
    )
    assert graph_df["idx"].max() < len(edge_raw_features), (
        f"{source}: edge ids exceed the edge feature matrix"
    )


def _validate_synthetic_timeline(args, graph_df: pd.DataFrame) -> None:
    """Validate the exact timelines used by the reconstructed split logic."""
    observed = sorted(int(t) for t in graph_df["ts"].unique())
    assert observed and observed[0] == 0, "synthetic timestamps must start at 0"
    assert observed == list(range(observed[-1] + 1)), "synthetic timelines cannot contain empty timestamp gaps"

    if args.data_family == "periodic":
        from generate.configs import settings

        cfg = settings["periodic"][args.data_configs]
        expected_steps = (cfg["training_repeats"] + 1) * cfg["periodicity"]
        assert observed[-1] == expected_steps - 1, (
            f"{args.data_configs}: expected {expected_steps} timestamps, found {observed[-1] + 1}"
        )
    elif args.data_family == "partition":
        from generate.configs import settings

        expected_steps = settings["partition"][args.data_configs]["ts"]
        assert observed[-1] == expected_steps - 1, (
            f"{args.data_configs}: expected {expected_steps} timestamps, found {observed[-1] + 1}"
        )


def get_timestamps(df: pd.DataFrame):
    """Return (val_time, test_time) split thresholds: 70/15/15 over edges."""
    val_time, test_time = list(np.quantile(df.ts, [0.70, 0.85]))
    return val_time, test_time


# ---------------------------------------------------------------------------
# Generic helpers used by generate/models/*.py
# ---------------------------------------------------------------------------

def repeat_or_validate(m, ts: int):
    """Return a per-timestep list of length ``ts`` from a scalar or validate a list."""
    if isinstance(m, (list, tuple, np.ndarray)):
        assert len(m) == ts, f"Expected list of length {ts}, got {len(m)}"
        return list(m)
    return [m] * ts


# ---------------------------------------------------------------------------
# Data container + index loader (verbatim from DyGLib utils/DataLoader.py)
# ---------------------------------------------------------------------------

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        super().__init__()
        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    dataset = CustomizedDataset(indices_list=indices_list)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


class Data:
    def __init__(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        labels: np.ndarray,
    ):
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


# ---------------------------------------------------------------------------
# load_data -- reconstructed against run_model.py
# ---------------------------------------------------------------------------

def _make_data(df: pd.DataFrame) -> Data:
    return Data(
        src_node_ids=df.u.values.astype(np.longlong),
        dst_node_ids=df.i.values.astype(np.longlong),
        node_interact_times=df.ts.values.astype(np.float64),
        edge_ids=df.idx.values.astype(np.longlong),
        labels=df.label.values,
    )


def load_data(args):
    """Load one dataset for training.

    Returns (node_raw_features, edge_raw_features, full_data, train_data,
    val_data, test_data, self_loops), matching the unpacking in run_model.py.
    """
    data_root = get_data_root()
    if args.data_family == "empirical":
        base = data_root / "empirical" / args.data_configs / args.train_variant
        graph_path = base / f"ml_{args.data_configs}.csv"
        edge_path = base / f"ml_{args.data_configs}.npy"
        node_path = base / f"ml_{args.data_configs}_node.npy"
    else:
        base = data_root / args.data_family / args.data_configs
        graph_path = base / f"data_{args.gen_seed}.csv"
        edge_path = base / f"edge_{args.gen_seed}.npy"
        node_path = base / f"node_{args.gen_seed}.npy"

    for path in (graph_path, edge_path, node_path):
        assert path.is_file(), f"required data asset not found: {path}"

    graph_df = pd.read_csv(graph_path)
    edge_raw_features = np.load(edge_path)
    node_raw_features = np.load(node_path)
    _validate_graph_frame(graph_df=graph_df, source=graph_path)
    _validate_feature_alignment(
        graph_df=graph_df,
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        source=graph_path,
    )
    if args.data_family in {"periodic", "partition"}:
        _validate_synthetic_timeline(args=args, graph_df=graph_df)

    # feature dimension guard (DyGLib pads to 172 dims)
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert node_raw_features.shape[1] <= NODE_FEAT_DIM
    assert edge_raw_features.shape[1] <= EDGE_FEAT_DIM
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_raw_features = np.concatenate(
            [node_raw_features, np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))],
            axis=1,
        )
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_raw_features = np.concatenate(
            [edge_raw_features, np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))],
            axis=1,
        )

    ts = graph_df.ts.values
    max_ts = ts.max()

    if args.data_family == "periodic":
        # final one-period block is the holdout (repository README);
        # validation coincides with the holdout period
        from generate.configs import settings

        periodicity = settings["periodic"][args.data_configs]["periodicity"]
        holdout_start = max_ts - periodicity + 1
        train_mask = ts < holdout_start
        # val == test == final period. Periodic runs must use --skip_validation
        # (paper: "where validation data was not logically related to the
        # training data, the training loss was used instead"): any validation
        # pass would advance memory-based models' memory into the holdout and
        # trip MemoryModel's "update memory to time in the past" assertion at
        # final prediction/evaluation time.
        val_mask = ts >= holdout_start
        test_mask = ts >= holdout_start
    elif args.data_family == "partition":
        # paper: 10 training time steps, predictions made at t = 10
        train_mask = ts <= max_ts - 2
        val_mask = ts == max_ts - 1
        test_mask = ts == max_ts
    else:
        # empirical / sbm / pa and other generated families: 70/15/15 split
        val_time, test_time = get_timestamps(graph_df)
        train_mask = ts <= val_time
        val_mask = (ts > val_time) & (ts <= test_time)
        test_mask = ts > test_time

    full_data = _make_data(graph_df)
    train_data = _make_data(graph_df[train_mask])
    val_data = _make_data(graph_df[val_mask])
    test_data = _make_data(graph_df[test_mask])

    assert train_data.num_interactions > 0
    assert val_data.num_interactions > 0
    assert test_data.num_interactions > 0
    assert train_data.node_interact_times.max() < val_data.node_interact_times.min()
    if args.data_family != "periodic":
        assert val_data.node_interact_times.max() < test_data.node_interact_times.min()

    self_loops = bool((graph_df.u.values == graph_df.i.values).any())

    return (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        self_loops,
    )

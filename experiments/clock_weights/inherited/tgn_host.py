"""GJ1 model definitions; importing this file does not instantiate/run a model.

This is a NEW functional atomic-snapshot adapter around the vendored audited
TGN components. It does not call MemoryModel's delayed-message query/update
API: that API detaches base memories and cannot provide the full 16-step BPTT
contract required here. The model is constructed only after Colab approval.

API: make_model(arm, seed, device) returns a torch Module. Its forward_batch
accepts own/neighbor boolean Bx16 record arrays, scalar shared/visible flags,
and a list of horizons; it returns (logits[B,H], gate[B,H] or None). Every call
rebuilds its own functional state. No history, labels or latent clock is an
inference argument. Only the gated arm accepts the evaluation-only gate_open
and gate_closed interventions. Checkpoints must contain state_dict, not this
dynamically defined class.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np


ARMS = ("native", "visible", "gated", "simple")
TIMESTAMPS = np.arange(80, 96, dtype=np.int64)
MEMORY_DIM = 32
TIME_DIM = 16
EDGE_DIM = 4
RECENT_NEIGHBORS = 32


def _records(value, label):
    result = np.asarray(value)
    if result.dtype.kind != "b" or result.ndim != 2 or result.shape[1] != 16:
        raise ValueError(f"{label} must be a boolean Bx16 array")
    if not result.shape[0]:
        raise ValueError("A batch must contain at least one motif")
    return result.copy()


def recurrence_inputs(own, horizons, shared, visible):
    """Observed-only metadata BxHx6 and sine/cosine cue BxHx2.

    Metadata order: h/8, shared, visible, record_count/16,
    (95-last_record)/16 (1 if no record), availability. At least three records
    are required; unavailable phase is (0,0). No phase modulo operation is
    imposed beyond the cosine/sine evaluation of the estimated cycle.
    """
    own = _records(own, "own")
    h = np.asarray(horizons)
    if h.ndim != 1 or h.size == 0 or h.dtype.kind not in "iu" or np.any((h < 1) | (h > 8)):
        raise ValueError("horizons must be nonempty integers between one and eight")
    if len(np.unique(h)) != len(h):
        raise ValueError("Duplicate horizons are not allowed")
    if not isinstance(shared, (bool, np.bool_)) or not isinstance(visible, (bool, np.bool_)):
        raise ValueError("shared and visible must be explicit boolean metadata")
    metadata = np.zeros((len(own), len(h), 6), dtype=np.float32)
    phi = np.zeros((len(own), len(h), 2), dtype=np.float32)
    metadata[:, :, 0] = h[None, :] / 8.0
    metadata[:, :, 1] = float(shared)
    metadata[:, :, 2] = float(visible)
    for row, bits in enumerate(own):
        times = TIMESTAMPS[bits]
        metadata[row, :, 3] = len(times) / 16.0
        metadata[row, :, 4] = (95 - int(times[-1])) / 16.0 if len(times) else 1.0
        if len(times) >= 3:
            period = float(np.median(np.diff(times)))
            if period <= 0:
                raise ValueError("Distinct observed timestamps must yield positive gaps")
            phase = 2 * np.pi * (95 + h - int(times[-1])) / period
            phi[row, :, 0] = np.cos(phase).astype(np.float32)
            phi[row, :, 1] = np.sin(phase).astype(np.float32)
            metadata[row, :, 5] = 1.0
    return metadata, phi


def _digest_tensors(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().contiguous().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def learned_weight_fingerprint(model):
    """The host's non-trainable legacy bank is not a learned parameter."""
    return _digest_tensors((name, p) for name, p in model.named_parameters() if p.requires_grad)


def _sampler_fingerprint(sampler):
    digest = hashlib.sha256()
    for name in ("nodes_neighbor_ids", "nodes_edge_ids", "nodes_neighbor_times", "nodes_event_priorities"):
        digest.update(name.encode("ascii"))
        for value in getattr(sampler, name):
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _runtime():
    # Configure before importing Torch; the worker is a separate process.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for key in list(os.environ):
        if key.startswith(("TGL_", "KEAT_")):
            del os.environ[key]
    os.environ.update(KEAT_KERNEL="none", KEAT_STRENGTH="0", KEAT_PLACEMENT="joint",
                      KEAT_KEY_LOGIT_CONTROL="none", KEAT_VALUE_NORM_CONTROL="none",
                      TGL_ZERO_EMPTY_ATTENTION="1", TGL_MESSAGE_AGGREGATION="snapshot_mean")
    root = Path(__file__).resolve().parent / "runtime"
    sys.path.insert(0, str(root))
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from train.models.MemoryModel import MemoryModel
    from train.prospective_sampler import CanonicalRecentNeighborSampler, build_canonical_adjacency
    module_path = Path(sys.modules[MemoryModel.__module__].__file__).resolve()
    if not module_path.is_relative_to(root):
        raise RuntimeError("Another train package was already imported; use the isolated worker")
    return torch, MemoryModel, CanonicalRecentNeighborSampler, build_canonical_adjacency


def make_model(arm, seed, device):
    """Create one fresh model; never called during local package preparation."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    torch, MemoryModel, Sampler, adjacency = _runtime()
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    class JointModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.arm = arm
            self.capture_gradient = False
            self.earliest_memory = None
            self.last_state_report = None
            if arm == "simple":
                self.simple = torch.nn.Sequential(
                    torch.nn.Linear(40, 256), torch.nn.ReLU(),
                    torch.nn.Linear(256, 128), torch.nn.ReLU(),
                    torch.nn.Linear(128, 64), torch.nn.ReLU(),
                    torch.nn.Linear(64, 1))
            else:
                # The one-row bank is unused. Real functional states and static
                # features are constructed per batch; no IDs are parameters.
                self.host = MemoryModel(
                    np.zeros((1, MEMORY_DIM), np.float32),
                    np.zeros((1, EDGE_DIM), np.float32), Sampler([[]]),
                    time_feat_dim=TIME_DIM, model_name="tgn", num_layers=1,
                    num_heads=2, dropout=0.0, device=str(device))
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(104, 128), torch.nn.ReLU(),
                    torch.nn.Linear(128, 64), torch.nn.ReLU(),
                    torch.nn.Linear(64, 1))
                self.gate = torch.nn.Sequential(
                    torch.nn.Linear(104, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
                # All three TGN arms have exactly the same parameter tensors.
                # Visible/native use this capacity as an additive residual.
                torch.nn.init.zeros_(self.gate[-1].weight)
                torch.nn.init.zeros_(self.gate[-1].bias)

        def parameter_counts(self):
            trainable = [(n, p.numel()) for n, p in self.named_parameters() if p.requires_grad]
            return {"trainable": sum(n for _, n in trainable),
                    "nontrainable": sum(p.numel() for p in self.parameters() if not p.requires_grad),
                    "by_parameter": {name: count for name, count in trainable}}

        def _functional_history(self, own, neighbor):
            batch = len(own)
            node_count = 3 * batch
            d = next(self.parameters()).device
            static = np.zeros((node_count + 1, MEMORY_DIM), np.float32)
            for motif in range(batch):
                for role in range(3):
                    static[3 * motif + role + 1, role] = 1.0
            # Stable timestamp -> motif -> stream ordering. Only actual records
            # create events; static relations and absences never do.
            records = []
            for offset, timestamp in enumerate(TIMESTAMPS):
                for motif in range(batch):
                    a, b, c = 3 * motif + np.arange(1, 4)
                    if own[motif, offset]:
                        records.append((a, b, int(timestamp), 0))
                    if neighbor[motif, offset]:
                        records.append((b, c, int(timestamp), 1))
            event = np.asarray(records, dtype=np.int64).reshape(-1, 4)
            edge_ids = np.arange(1, len(event) + 1, dtype=np.int64)
            edge_features = np.zeros((len(event) + 1, EDGE_DIM), np.float32)
            if len(event):
                edge_features[edge_ids, event[:, 3]] = 1.0
            sampler = Sampler(adjacency(
                event[:, 0], event[:, 1], edge_ids, event[:, 2],
                event_priorities=edge_ids, num_nodes=node_count, undirected=True))
            self.host.node_raw_features = torch.as_tensor(static, device=d)
            self.host.edge_raw_features = torch.as_tensor(edge_features, device=d)
            self.host.embedding_module.node_raw_features = self.host.node_raw_features
            self.host.embedding_module.edge_raw_features = self.host.edge_raw_features
            self.host.set_neighbor_sampler(sampler)
            memory = torch.zeros((node_count + 1, MEMORY_DIM), device=d)
            last = torch.full((node_count + 1,), 79.0, device=d)
            self.earliest_memory = None
            for timestamp in TIMESTAMPS:
                indices = np.flatnonzero(event[:, 2] == timestamp)
                if not len(indices):
                    continue
                # Both directions read the identical pre-snapshot memory.
                src_np = np.concatenate([event[indices, 0], event[indices, 1]])
                dst_np = np.concatenate([event[indices, 1], event[indices, 0]])
                ids_np = np.concatenate([edge_ids[indices], edge_ids[indices]])
                src = torch.as_tensor(src_np, dtype=torch.long, device=d)
                dst = torch.as_tensor(dst_np, dtype=torch.long, device=d)
                ids = torch.as_tensor(ids_np, dtype=torch.long, device=d)
                delta = torch.full((len(src), 1), float(timestamp), device=d) - last[src, None]
                time_features = self.host.time_encoder(delta).reshape(len(src), TIME_DIM)
                messages = torch.cat([memory[src], memory[dst], time_features,
                                      self.host.edge_raw_features[ids]], dim=1)
                unique = np.unique(src_np)
                means = []
                for node in unique:
                    positions = np.flatnonzero(src_np == node)
                    order = positions[np.argsort(ids_np[positions], kind="stable")]
                    means.append(messages[torch.as_tensor(order, device=d)].mean(dim=0))
                unique_t = torch.as_tensor(unique, device=d)
                updated = self.host.memory_updater.memory_updater(torch.stack(means), memory[unique_t])
                memory = memory.index_copy(0, unique_t, updated)
                last = last.index_copy(0, unique_t, torch.full((len(unique),), float(timestamp), device=d))
                if self.capture_gradient and self.earliest_memory is None and memory.requires_grad:
                    memory.retain_grad()
                    self.earliest_memory = memory
            return memory, last, sampler, event

        def forward_batch(self, own, neighbor, shared, visible, horizons, intervention="none"):
            own = _records(own, "own")
            neighbor = _records(neighbor, "neighbor")
            if own.shape != neighbor.shape:
                raise ValueError("Own and neighbor batches must have the same shape")
            metadata_np, phi_np = recurrence_inputs(own, horizons, shared, visible)
            if intervention not in {"none", "gate_open", "gate_closed"}:
                raise ValueError("Only the frozen gate interventions are implemented")
            if intervention != "none" and (self.arm != "gated" or self.training):
                raise ValueError("Gate interventions require the gated arm in evaluation mode")
            if not visible:
                neighbor.fill(False)
            batch, horizon_count = metadata_np.shape[:2]
            d = next(self.parameters()).device
            metadata = torch.as_tensor(metadata_np, device=d)
            phi = torch.as_tensor(phi_np, device=d)
            weights_before = learned_weight_fingerprint(self)
            gate = None
            self.earliest_memory = None
            if self.arm == "simple":
                raw = np.concatenate([own, neighbor], axis=1).astype(np.float32)
                raw_t = torch.as_tensor(raw, device=d)[:, None, :].expand(-1, horizon_count, -1)
                logits = self.simple(torch.cat([raw_t, metadata, phi], dim=-1)).squeeze(-1)
                self.last_state_report = {"pending_messages": 0, "functional_state": False,
                                          "query_read_only": True, "learned_weights_unchanged": True}
            else:
                memory, last, sampler, event = self._functional_history(own, neighbor)
                state_before = _digest_tensors((("memory", memory), ("last_update", last)))
                sampler_before = _sampler_fingerprint(sampler)
                # Pack motif, horizon, node role. All queries use one unchanged
                # cutoff state; no query events are ingested.
                node_ids = np.broadcast_to(
                    np.arange(1, 3 * batch + 1).reshape(batch, 1, 3),
                    (batch, horizon_count, 3)).reshape(-1)
                queries = np.broadcast_to(
                    (95 + np.asarray(horizons, dtype=np.int64))[None, :, None],
                    (batch, horizon_count, 3)).reshape(-1).astype(np.float64)
                self.host.embedding_module.audit_query_kind = "gj1_read_only"
                representation = self.host.embedding_module.compute_node_temporal_embeddings(
                    node_memories=memory, node_ids=node_ids, node_interact_times=queries,
                    current_layer_num=1, num_neighbors=RECENT_NEIGHBORS).reshape(batch, horizon_count, 96)
                if _digest_tensors((("memory", memory), ("last_update", last))) != state_before:
                    raise RuntimeError("Read-only query changed functional memory or last-update state")
                if _sampler_fingerprint(sampler) != sampler_before:
                    raise RuntimeError("Read-only query changed neighbor history")
                if self.arm == "native":
                    phi = torch.zeros_like(phi)
                x = torch.cat([representation, metadata, phi], dim=-1)
                if self.arm == "gated":
                    gate = torch.sigmoid(self.gate(x)).squeeze(-1)
                    if intervention == "gate_open":
                        gate = torch.ones_like(gate)
                    elif intervention == "gate_closed":
                        gate = torch.zeros_like(gate)
                    logits = self.decoder(torch.cat([representation, metadata, gate[..., None] * phi], dim=-1)).squeeze(-1)
                else:
                    logits = (self.decoder(x) + self.gate(x)).squeeze(-1)
                self.last_state_report = {
                    "pending_messages": 0, "functional_state": True,
                    "query_read_only": True, "state_sha256": state_before,
                    "sampler_sha256": sampler_before, "event_count": len(event),
                    "node_count": 3 * batch, "cutoff": 95,
                    "last_update_initial": 79, "learned_weights_unchanged": True,
                    "full_history_gradient_enabled": bool(torch.is_grad_enabled())}
            if learned_weight_fingerprint(self) != weights_before:
                raise RuntimeError("Forward pass modified learned weights")
            if logits.shape != (batch, horizon_count) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Nonfinite or incorrectly shaped logits")
            return logits, gate

        def forward(self, own, neighbor, shared, visible, horizons, intervention="none"):
            return self.forward_batch(own, neighbor, shared, visible, horizons, intervention)

    return JointModel().to(device)


def gradient_canary(model):
    """Hand-built technical fixture, executed only in approved Colab preflight.

    No optimizer step, generated research cohort, forecast metric or training
    outcome is produced. The earliest functional memory must influence the
    final query and the inherited GRU/time encoder must receive gradients.
    """
    if model.arm == "simple":
        raise ValueError("The historical gradient canary requires a TGN arm")
    import torch
    own = np.zeros((2, 16), dtype=bool)
    neighbor = np.zeros_like(own)
    own[0, [0, 4, 8, 12]] = True
    own[1, [1, 7, 13]] = True
    neighbor[0, [2, 6, 10, 14]] = True
    neighbor[1, [3, 9, 15]] = True
    mode = model.training
    before = learned_weight_fingerprint(model)
    model.train()
    model.zero_grad(set_to_none=True)
    model.capture_gradient = True
    try:
        logits, _ = model.forward_batch(own, neighbor, True, True, [1, 2, 3, 4])
        # A deterministic non-cancelling scalar probe is not a label fit.
        (logits.square().mean() + logits.mean()).backward()
        early = model.earliest_memory
        if early is None or early.grad is None:
            raise RuntimeError("No gradient reached earliest recorded snapshot memory")
        early_norm = float(early.grad.detach().abs().sum().cpu())
        groups = {}
        for label, module in (("time_encoder", model.host.time_encoder),
                              ("memory_gru", model.host.memory_updater.memory_updater),
                              ("attention", model.host.embedding_module.temporal_conv_layers)):
            values = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            if not values or not all(bool(torch.isfinite(g).all()) for g in values):
                raise RuntimeError(f"Missing or nonfinite {label} gradients")
            groups[label] = sum(float(g.detach().abs().sum().cpu()) for g in values)
        if not early_norm > 0 or not all(value > 0 for value in groups.values()):
            raise RuntimeError("The hand-built historical gradient probe is zero")
        if learned_weight_fingerprint(model) != before:
            raise RuntimeError("Gradient canary changed learned weights")
        return {"earliest_snapshot_gradient_l1": early_norm, **groups,
                "optimizer_steps": 0, "learned_weights_unchanged": True}
    finally:
        model.zero_grad(set_to_none=True)
        model.capture_gradient = False
        model.earliest_memory = None
        model.train(mode)

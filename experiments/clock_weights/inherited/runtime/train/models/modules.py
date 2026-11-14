"""
Code from DyGLib.
"""

import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F


def _keat_masked_dispersion_match(source, target, padding_mask):
    """Match valid-neighbor centered RMS without changing source ordering."""
    valid = ~padding_mask
    count = valid.sum(dim=-1, keepdim=True)
    denominator = count.clamp_min(1)
    source_mean = (source * valid).sum(dim=-1, keepdim=True) / denominator
    target_mean = (target * valid).sum(dim=-1, keepdim=True) / denominator
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_rms = ((source_centered.square() * valid).sum(dim=-1, keepdim=True) / denominator).sqrt()
    target_rms = ((target_centered.square() * valid).sum(dim=-1, keepdim=True) / denominator).sqrt()
    epsilon = 32 * torch.finfo(source.dtype).eps
    eligible = count > 1
    impossible = eligible & (source_rms <= epsilon) & (target_rms > epsilon)
    if bool(impossible.any()):
        raise RuntimeError("Cannot match a nonzero target logit dispersion from a zero-dispersion source")
    scalable = eligible & (source_rms > epsilon)
    ratio = torch.where(scalable, target_rms / source_rms, torch.ones_like(source_rms))
    matched = target_mean + source_centered * ratio
    matched = torch.where(valid & scalable, matched, source)
    diagnostics = {
        "eligible_groups": int(eligible.sum().item()),
        "scaled_groups": int(scalable.sum().item()),
        "zero_target_groups": int((eligible & (target_rms <= epsilon)).sum().item()),
        "singleton_or_empty_groups": int((~eligible).sum().item()),
    }
    return matched, diagnostics


def _keat_projected_value_norm_match(source, target, padding_mask):
    """Match each valid projected value-vector norm while preserving direction."""
    valid = (~padding_mask.squeeze(2)).unsqueeze(-1)
    source_norm = source.norm(dim=-1, keepdim=True)
    target_norm = target.norm(dim=-1, keepdim=True)
    epsilon = 32 * torch.finfo(source.dtype).eps
    impossible = valid & (source_norm <= epsilon) & (target_norm > epsilon)
    if bool(impossible.any()):
        raise RuntimeError("Cannot match a nonzero raw projected-value norm from a zero gated vector")
    scalable = valid & (source_norm > epsilon)
    ratio = torch.where(scalable, target_norm / source_norm, torch.ones_like(source_norm))
    matched = torch.where(valid, source * ratio, source)
    diagnostics = {
        "eligible_vectors": int(valid.sum().item()),
        "scaled_vectors": int(scalable.sum().item()),
        "zero_target_vectors": int((valid & (target_norm <= epsilon)).sum().item()),
    }
    return matched, diagnostics


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.

        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps

        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.

        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs

        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Multi-Layer Perceptron Classifier.

        :param input_dim: int, dimension of input
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        multi-layer perceptron classifier forward process

        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        # Tensor, shape (*, 80)
        x = self.dropout(self.act(self.fc1(x)))
        # Tensor, shape (*, 10)
        x = self.dropout(self.act(self.fc2(x)))
        # Tensor, shape (*, 1)
        return self.fc3(x)


class MultiHeadAttention(nn.Module):

    def __init__(
        self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, num_heads: int = 2, dropout: float = 0.1
    ):
        """
        Multi-head Attention module.

        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim
        self.audit_callback = None
        self.audit_context = None

        assert (
            self.query_dim % num_heads == 0
        ), "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim**-0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        node_time_features: torch.Tensor,
        neighbor_node_features: torch.Tensor,
        neighbor_node_time_features: torch.Tensor,
        neighbor_node_edge_features: torch.Tensor,
        neighbor_masks: np.ndarray,
        raw_neighbor_node_time_features: torch.Tensor = None,
        raw_neighbor_node_edge_features: torch.Tensor = None,
    ):
        """
        temporal attention forward process

        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors),
                                used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # The historical KEAT port feeds one gated context to both projections.
        # The opt-in placement switch separates the attention-logit route
        # (key) from the aggregated-content route (value) without introducing
        # parameters or changing the vanilla path.  Callers that do not supply
        # raw tensors retain the original behavior.
        gated_context = torch.cat(
            [neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2
        )
        if raw_neighbor_node_time_features is None:
            raw_neighbor_node_time_features = neighbor_node_time_features
        if raw_neighbor_node_edge_features is None:
            raw_neighbor_node_edge_features = neighbor_node_edge_features
        raw_context = torch.cat(
            [neighbor_node_features, raw_neighbor_node_edge_features, raw_neighbor_node_time_features], dim=2
        )
        if (
            neighbor_node_time_features is raw_neighbor_node_time_features
            and neighbor_node_edge_features is raw_neighbor_node_edge_features
        ):
            gated_context = raw_context
        placement = os.environ.get("KEAT_PLACEMENT", "joint").strip().lower()
        if placement in {"joint", "both"}:
            key_uses_gated, value_uses_gated = True, True
            key_context, value_context = gated_context, gated_context
        elif placement == "key":
            key_uses_gated, value_uses_gated = True, False
            key_context, value_context = gated_context, raw_context
        elif placement == "value":
            key_uses_gated, value_uses_gated = False, True
            key_context, value_context = raw_context, gated_context
        elif placement in {"none", "identity"}:
            key_uses_gated, value_uses_gated = False, False
            key_context, value_context = raw_context, raw_context
        else:
            raise ValueError(
                f"unknown KEAT_PLACEMENT={placement!r}; expected joint, key, value, or none"
            )

        def project_key(context):
            return self.key_projection(context).reshape(
                context.shape[0], context.shape[1], self.num_heads, self.head_dim
            )

        def project_value(context):
            return self.value_projection(context).reshape(
                context.shape[0], context.shape[1], self.num_heads, self.head_dim
            )

        # Compute the raw and gated endpoints explicitly.  The selected route
        # reuses an endpoint tensor where possible so the Stage C1 canaries
        # remain exact while nuisance controls can reference the alternative.
        raw_key = project_key(raw_context)
        contexts_equal = gated_context is raw_context
        gated_key = raw_key if contexts_equal else project_key(gated_context)
        key = gated_key if key_uses_gated else raw_key
        raw_value = project_value(raw_context)
        gated_value = raw_value if contexts_equal else project_value(gated_context)
        value = gated_value if value_uses_gated else raw_value

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        raw_key = raw_key.permute(0, 2, 1, 3)
        gated_key = raw_key if contexts_equal else gated_key.permute(0, 2, 1, 3)
        key = gated_key if key_uses_gated else raw_key
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        raw_value = raw_value.permute(0, 2, 1, 3)
        gated_value = raw_value if contexts_equal else gated_value.permute(0, 2, 1, 3)
        value = gated_value if value_uses_gated else raw_value

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        raw_attention = torch.einsum("bhld,bhnd->bhln", query, raw_key) * self.scaling_factor
        gated_attention = (
            raw_attention
            if contexts_equal
            else torch.einsum("bhld,bhnd->bhln", query, gated_key) * self.scaling_factor
        )
        attention = gated_attention if key_uses_gated else raw_attention
        pre_control_attention = attention
        pre_control_value = value

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        key_control = os.environ.get("KEAT_KEY_LOGIT_CONTROL", "none").strip().lower()
        if key_control == "none":
            key_control_diagnostics = {
                "eligible_groups": 0, "scaled_groups": 0,
                "zero_target_groups": 0, "singleton_or_empty_groups": 0,
            }
        elif key_control == "match_raw_std":
            if not key_uses_gated:
                raise ValueError("match_raw_std requires the gated key route")
            attention, key_control_diagnostics = _keat_masked_dispersion_match(
                attention, raw_attention, attention_mask
            )
        elif key_control == "raw_match_gated_std":
            if key_uses_gated:
                raise ValueError("raw_match_gated_std requires the raw key route")
            attention, key_control_diagnostics = _keat_masked_dispersion_match(
                attention, gated_attention, attention_mask
            )
        else:
            raise ValueError(
                "unknown KEAT_KEY_LOGIT_CONTROL; expected none, match_raw_std, or raw_match_gated_std"
            )

        value_control = os.environ.get("KEAT_VALUE_NORM_CONTROL", "none").strip().lower()
        if value_control == "none":
            value_control_diagnostics = {
                "eligible_vectors": 0, "scaled_vectors": 0, "zero_target_vectors": 0,
            }
        elif value_control == "match_raw_l2":
            if not value_uses_gated:
                raise ValueError("match_raw_l2 requires the gated value route")
            value, value_control_diagnostics = _keat_projected_value_norm_match(
                value, raw_value, attention_mask
            )
        else:
            raise ValueError(
                "unknown KEAT_VALUE_NORM_CONTROL; expected none or match_raw_l2"
            )
        controlled_attention = attention

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))
        if os.environ.get("TGL_ZERO_EMPTY_ATTENTION", "0") == "1":
            # Opt-in scientific variant: empty neighborhoods have no neighbor contribution.
            attention_scores = attention_scores.masked_fill(attention_mask.all(dim=-1, keepdim=True), 0.)
        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        weighted_values = torch.einsum("bhln,bhnd->bhld", attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = weighted_values.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        residual_projection = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        pre_layer_norm = residual_projection + residual
        output = self.layer_norm(pre_layer_norm)

        if self.audit_callback is not None:
            self.audit_callback({
                "context": self.audit_context,
                "raw_logits": raw_attention.detach(),
                "gated_logits": gated_attention.detach(),
                "pre_control_logits": pre_control_attention.detach(),
                "controlled_logits": controlled_attention.detach(),
                "logits": attention.detach(),
                "attention": attention_scores.detach(),
                "raw_values": raw_value.detach(),
                "gated_values": gated_value.detach(),
                "pre_control_values": pre_control_value.detach(),
                "values": value.detach(),
                "weighted_values": weighted_values.detach(),
                "residual_projection": residual_projection.detach(),
                "pre_layer_norm": pre_layer_norm.detach(),
                "post_layer_norm": output.detach(),
                "padding_mask": attention_mask.detach(),
                "key_control_diagnostics": key_control_diagnostics,
                "value_control_diagnostics": value_control_diagnostics,
            })

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.

        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=num_heads, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
                nn.Linear(in_features=4 * attention_dim, out_features=attention_dim),
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(attention_dim), nn.LayerNorm(attention_dim)])

    def forward(
        self,
        inputs_query: torch.Tensor,
        inputs_key: torch.Tensor = None,
        inputs_value: torch.Tensor = None,
        neighbor_masks: np.ndarray = None,
    ):
        """
        encode the inputs by Transformer encoder

        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length),
                                used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape
        # (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = (
            inputs_query.transpose(0, 1),
            inputs_key.transpose(0, 1),
            inputs_value.transpose(0, 1),
        )

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(
            query=transposed_inputs_query,
            key=transposed_inputs_key,
            value=transposed_inputs_value,
            key_padding_mask=neighbor_masks,
        )[0].transpose(0, 1)

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs

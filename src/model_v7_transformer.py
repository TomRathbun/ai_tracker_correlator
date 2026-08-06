"""
V7 Transformer Tracker-Correlator

Pure Transformer architecture for multi-sensor radar tracking:
  - Measurement self-attention (soft clustering / context)
  - Track → measurement cross-attention (association)
  - GRU temporal memory across streaming windows
  - Existence, clutter, and kinematic residual heads

No torch_geometric dependency.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TransformerTrackerV7(nn.Module):
    """
    DETR-style tracker correlator.

    Tokens are a joint set [tracks | measurements]. Measurements attend among
    themselves; tracks cross-attend to measurements for association, then update
    via GRU for temporal continuity.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        max_sensors: int = 8,
        max_assoc_m: float = 50_000.0,
        use_radius_mask: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_assoc_m = max_assoc_m
        self.use_radius_mask = use_radius_mask

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.role_emb = nn.Embedding(2, hidden_dim)  # 0 meas, 1 track
        self.sensor_emb = nn.Embedding(max_sensors + 1, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.meas_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.cross_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(
                            hidden_dim, num_heads, dropout=dropout, batch_first=True
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                            nn.Dropout(dropout),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # Track self-attention (refine queries among themselves)
        self.track_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.track_self_norm = nn.LayerNorm(hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        self.exist_head = nn.Linear(hidden_dim, 2)  # [survival, init]
        self.clutter_head = nn.Linear(hidden_dim, 1)

    def _embed(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        sensor_id: torch.Tensor,
    ) -> torch.Tensor:
        sid = sensor_id.clamp(0, self.sensor_emb.num_embeddings - 1)
        return self.encoder(x) + self.role_emb(node_type) + self.sensor_emb(sid)

    def _radius_attn_mask(
        self,
        track_xy: torch.Tensor,
        meas_xy: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return additive attention mask (num_heads*batch compatible float mask).

        MHA expects attn_mask of shape (L, S) or (N*num_heads, L, S).
        True/finite: we use float mask with -inf blocked.
        """
        if not self.use_radius_mask or track_xy.numel() == 0 or meas_xy.numel() == 0:
            return None
        dist = torch.cdist(track_xy[:, :2], meas_xy[:, :2])  # (T, M)
        # allowed where dist < max
        blocked = dist > self.max_assoc_m
        # If a track would be fully blocked, unmask its nearest measurement
        all_blocked = blocked.all(dim=1)
        if all_blocked.any():
            nearest = dist.argmin(dim=1)
            blocked = blocked.clone()
            blocked[all_blocked, nearest[all_blocked]] = False
        mask = torch.zeros_like(dist)
        mask = mask.masked_fill(blocked, float("-inf"))
        return mask  # (T, M)

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        sensor_id: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        clutter_thresh: float = 0.70,
    ):
        """
        Args:
            x: (N, 8) features
            node_type: (N,) long 0=meas, 1=track — tracks must be prefix [0:num_tracks)
            sensor_id: (N,) long
            hidden_state: (num_prev_tracks, H) or None

        Returns:
            out_raw: (N, 8) [dx..dvz, survival, init]
            new_hidden_full: (N, H)
            attn_weights: (T, M) or None
            clutter_logits: (N,)
        """
        N = x.shape[0]
        device = x.device
        h = self._embed(x, node_type, sensor_id)

        track_mask = node_type == 1
        meas_mask = node_type == 0
        num_tracks = int(track_mask.sum().item())
        num_meas = int(meas_mask.sum().item())

        attn_weights = None

        # --- Measurement encoder (self-attention) ---
        if num_meas > 0:
            meas_h = h[meas_mask].unsqueeze(0)  # (1, M, H)
            meas_h = self.meas_encoder(meas_h).squeeze(0)
            h = h.clone()
            h[meas_mask] = meas_h
        else:
            meas_h = h.new_zeros((0, self.hidden_dim))

        # --- Track decoder (self + cross) ---
        if num_tracks > 0:
            track_h = h[track_mask]
            # Self-attn among tracks
            q = track_h.unsqueeze(0)
            refined, _ = self.track_self_attn(q, q, q)
            track_h = self.track_self_norm(track_h + refined.squeeze(0))

            if num_meas > 0:
                attn_mask = self._radius_attn_mask(x[track_mask, :3], x[meas_mask, :3])
                last_w = None
                for layer in self.cross_layers:
                    q = track_h.unsqueeze(0)
                    k = meas_h.unsqueeze(0)
                    attn_out, w = layer["attn"](q, k, k, attn_mask=attn_mask)
                    track_h = layer["norm1"](track_h + attn_out.squeeze(0))
                    track_h = layer["norm2"](track_h + layer["ffn"](track_h))
                    last_w = w
                # Average heads: (1, T, M) or (1, heads, T, M) depending on version
                if last_w is not None:
                    if last_w.dim() == 4:
                        attn_weights = last_w.mean(dim=1).squeeze(0)
                    else:
                        attn_weights = last_w.squeeze(0)

            h = h.clone()
            h[track_mask] = track_h

        # --- Temporal GRU (all nodes; tracks carry memory) ---
        if hidden_state is None or hidden_state.numel() == 0:
            hidden_full = torch.zeros(N, self.hidden_dim, device=device)
        else:
            prev_t = hidden_state.shape[0]
            if prev_t >= N:
                hidden_full = hidden_state[:N]
            else:
                pad = torch.zeros(N - prev_t, self.hidden_dim, device=device)
                hidden_full = torch.cat([hidden_state, pad], dim=0)

        new_hidden = self.gru(h, hidden_full)
        new_hidden = self.temporal_norm(new_hidden)

        # --- Heads ---
        state_delta = self.state_head(new_hidden)
        exist_logits = self.exist_head(new_hidden)  # (N, 2)
        clutter_logits = self.clutter_head(new_hidden).squeeze(-1)

        out_raw = torch.cat([state_delta, exist_logits], dim=-1)  # (N, 8)
        return out_raw, new_hidden, attn_weights, clutter_logits


# ---------------------------------------------------------------------------
# Data / graph-free helpers (same contract as V6 train loop)
# ---------------------------------------------------------------------------

def frame_to_tensors(frame_data: Union[Dict, List], device, window_t=None):
    """8-feature header: x,y,z,vx,vy,vz,amplitude,dt_offset."""
    from src.data_schema import get_sensor_id, get_time, normalize_measurement_dict

    if isinstance(frame_data, dict):
        measurements = frame_data.get("measurements", [])
    else:
        measurements = frame_data

    meas_list, sid_list = [], []
    for m in measurements:
        if not isinstance(m, dict):
            continue
        mn = normalize_measurement_dict(m)
        mt = get_time(mn, 0.0)
        t_offset = (window_t - mt) if window_t is not None else 0.0
        row = [
            float(mn.get("x", 0.0) or 0.0),
            float(mn.get("y", 0.0) or 0.0),
            float(mn.get("z", 0.0) or 0.0),
            float(mn["vx"]) if mn.get("vx") is not None else 0.0,
            float(mn["vy"]) if mn.get("vy") is not None else 0.0,
            float(mn["vz"]) if mn.get("vz") is not None else 0.0,
            float(mn["amplitude"]) if mn.get("amplitude") is not None else 0.0,
            float(t_offset),
        ]
        meas_list.append(row)
        sid_list.append(get_sensor_id(mn, 0))

    if not meas_list:
        return (
            torch.empty((0, 8), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )
    return (
        torch.tensor(meas_list, dtype=torch.float32, device=device),
        torch.tensor(sid_list, dtype=torch.long, device=device),
    )


def build_full_input(active_tracks, meas, meas_sensor_ids, num_sensors, device):
    """Align tracks (prefix) + measurements into one node tensor."""
    if active_tracks:
        track_kin = torch.stack([tr["state_tensor"] for tr in active_tracks])
        extra = torch.zeros(len(active_tracks), 2, device=device)
        track_features = torch.cat([track_kin, extra], dim=1)
        track_hiddens = torch.stack([tr["hidden"] for tr in active_tracks])
        track_sensor_ids = torch.full(
            (len(active_tracks),), num_sensors, dtype=torch.long, device=device
        )
        full_x = torch.cat([track_features, meas], dim=0)
        full_sensor_id = torch.cat([track_sensor_ids, meas_sensor_ids])
        return full_x, full_sensor_id, track_hiddens, len(active_tracks)
    return meas, meas_sensor_ids, None, 0


def build_gnn_edges(full_x, node_type, device, max_dist=75000.0, k=15):
    """Stub for factory compatibility — Transformer path does not use edges."""
    return (
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty((0, 7), device=device),
    )


def model_forward(
    model,
    x,
    node_type,
    sensor_id,
    edge_index=None,
    edge_attr=None,
    hidden_state=None,
    clutter_thresh=0.70,
):
    """Standardized return matching V6 train loop unpacking."""
    raw_out, new_hidden_full, alpha, clutter_logits = model(
        x, node_type, sensor_id, hidden_state, clutter_thresh
    )

    state_delta = raw_out[:, :6]
    survival_logits = raw_out[:, 6]
    init_logits = raw_out[:, 7]
    existence_logits = torch.where(node_type == 1, survival_logits, init_logits)

    updated_state = x[:, :6] + state_delta
    out = torch.cat([updated_state, existence_logits.unsqueeze(-1)], dim=-1)
    existence_probs = torch.sigmoid(existence_logits)
    clutter_probs = torch.sigmoid(clutter_logits)

    return (
        out,
        new_hidden_full,
        alpha,
        existence_probs,
        existence_logits,
        clutter_probs,
        clutter_logits,
        edge_index,
    )


def manage_tracks(
    active_tracks,
    out,
    new_hidden_full,
    existence_probs,
    existence_logits,
    clutter_probs,
    alpha,
    edge_index,
    num_tracks,
    num_meas,
    init_thresh,
    coast_thresh,
    suppress_thresh,
    del_exist,
    del_age,
    track_cap,
    dt=0.0,
    clutter_thresh=0.70,
):
    """Track lifecycle using existence probs + optional attention suppress."""
    actual_meas_nodes = num_meas
    attn_suppress = torch.zeros(actual_meas_nodes, dtype=torch.bool, device=out.device)

    if num_tracks > 0 and num_meas > 0 and alpha is not None:
        if isinstance(alpha, (tuple, list)):
            alpha = alpha[1] if len(alpha) > 1 else alpha[0]
        if alpha.dim() == 3:
            alpha = alpha.mean(dim=0) if alpha.shape[0] <= 8 else alpha.squeeze(0)
        if alpha.dim() == 2 and alpha.shape[0] == num_tracks:
            meas_incoming = alpha.sum(dim=0)
            if meas_incoming.numel() >= actual_meas_nodes:
                attn_suppress = meas_incoming[:actual_meas_nodes] > suppress_thresh

    selected = []
    if num_tracks > 0:
        for i in range(num_tracks):
            prob = existence_probs[i]
            if prob > coast_thresh:
                track = active_tracks[i].copy()
                track["state_tensor"] = out[i, :6].detach()
                track["hidden"] = new_hidden_full[i].detach()
                track["logit"] = existence_logits[i]
                if prob > 0.4:
                    track["age"] = 0
                    track["hits"] = track.get("hits", 0) + 1
                else:
                    track["age"] = track.get("age", 0) + 1
                s = track["state_tensor"]
                track["x"], track["y"], track["z"] = s[0].item(), s[1].item(), s[2].item()
                track["vx"], track["vy"], track["vz"] = s[3].item(), s[4].item(), s[5].item()
                selected.append(track)

    if num_meas > 0:
        meas_offset = num_tracks
        for i in range(num_meas):
            idx = meas_offset + i
            prob = existence_probs[idx]
            if prob > init_thresh:
                if clutter_probs[idx] > clutter_thresh:
                    continue
                if i < attn_suppress.shape[0] and attn_suppress[i]:
                    continue
                s = out[idx, :6].detach()
                next_id = 0
                if active_tracks:
                    next_id = max(tr.get("id", -1) for tr in active_tracks) + 1
                selected.append(
                    {
                        "id": next_id + idx,
                        "state_tensor": s,
                        "x": s[0].item(),
                        "y": s[1].item(),
                        "z": s[2].item(),
                        "vx": s[3].item(),
                        "vy": s[4].item(),
                        "vz": s[5].item(),
                        "hidden": new_hidden_full[idx].detach(),
                        "logit": existence_logits[idx],
                        "age": 0,
                        "hits": 1,
                        "is_new": True,
                    }
                )

    selected = [
        tr
        for tr in selected
        if (torch.sigmoid(tr["logit"]) > del_exist) or (tr.get("age", 0) < 1)
    ]
    if len(selected) > track_cap:
        selected.sort(key=lambda t: torch.sigmoid(t["logit"]).item(), reverse=True)
        selected = selected[:track_cap]
    return selected


def focal_bce(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = alpha * (1 - p_t) ** gamma * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def compute_loss(
    pred_states,
    pred_logits,
    gt_states_dev,
    num_gt,
    match_gate,
    miss_penalty,
    fp_mult,
    out,
    epoch,
    num_meas,
    meas=None,
    existence_logits=None,
    clutter_logits=None,
    num_tracks=0,
    pred_ages=None,
    aux_init_weight=5.0,
    attn_weights=None,
):
    device = out.device
    reg_loss = exist_matched_loss = exist_fp_loss = matched_exist_loss = torch.tensor(
        0.0, device=device
    )
    miss_loss = torch.tensor(miss_penalty * num_gt, device=device)

    if pred_states.shape[0] > 0 and num_gt > 0:
        cost_matrix = torch.cdist(pred_states[:, :3], gt_states_dev[:, :3])
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        valid = cost_np[row_ind, col_ind] < match_gate
        row_ind, col_ind = row_ind[valid], col_ind[valid]
        row_t = torch.from_numpy(row_ind).to(device)

        if len(row_ind) > 0:
            reg_loss = F.smooth_l1_loss(pred_states[row_t], gt_states_dev[col_ind])
            exist_matched_loss = 2.0 * focal_bce(
                pred_logits[row_t], torch.ones_like(pred_logits[row_t])
            )
            matched_exist_loss = F.mse_loss(
                pred_logits[row_t], torch.full_like(pred_logits[row_t], 4.0)
            )

        matched_mask = torch.zeros(len(pred_logits), dtype=torch.bool, device=device)
        if len(row_ind) > 0:
            matched_mask[row_t] = True
        if (~matched_mask).any():
            exist_fp_loss = fp_mult * focal_bce(
                pred_logits[~matched_mask],
                torch.zeros_like(pred_logits[~matched_mask]),
            )
        miss_loss = torch.tensor(miss_penalty * (num_gt - len(row_ind)), device=device)

    loss = reg_loss + exist_matched_loss + exist_fp_loss + miss_loss + 2.0 * matched_exist_loss

    attn_reg = torch.tensor(0.0, device=device)
    if attn_weights is not None:
        w = attn_weights
        if isinstance(w, tuple):
            w = w[-1]
        if w.dim() >= 2:
            # Encourage peaked attention (low entropy)
            p = w.clamp_min(1e-8)
            p = p / p.sum(dim=-1, keepdim=True)
            entropy = -(p * p.log()).sum(dim=-1).mean()
            attn_reg = 0.05 * entropy
            loss = loss + attn_reg

    metrics = {
        "reg": float(reg_loss.detach()),
        "exist_pos": float(exist_matched_loss.detach()),
        "exist_fp": float(exist_fp_loss.detach()),
        "miss": float(miss_loss.detach()),
        "attn_reg": float(attn_reg.detach()),
        "total": float(loss.detach()),
    }
    return loss, metrics


# Alias expected by factory.get_model_suite
RecurrentGATTrackerV7 = TransformerTrackerV7

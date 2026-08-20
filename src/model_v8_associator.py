"""
V8 Association Transformer.

Scores gated track↔measurement (and meas↔meas) pairs for Hybrid.
Does not predict state, existence, or carry temporal hidden state.

See artifacts/design_v8.md.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_schema import get_meas_type, get_mode_3a, get_sensor_id, get_time, normalize_measurement_dict

NUMERIC_DIM = 15
REL_DIM = 12
CLUSTER_GATE_M = 2000.0
ASSOC_GATE_M = 8000.0
MODE3A_VOCAB = 4096
MODES_VOCAB = 1024


def _as_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _stable_bucket(text: str, n_buckets: int) -> int:
    """Process-stable hash in 1..n_buckets-1 (0 reserved for missing)."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % (n_buckets - 1)) + 1


def mode_3a_index(val) -> int:
    if val is None:
        return 0
    s = str(val).strip()
    if not s:
        return 0
    try:
        n = int(s, 8)
        if 0 <= n < MODE3A_VOCAB:
            return n
    except ValueError:
        pass
    try:
        n = int(s)
        return n % MODE3A_VOCAB
    except ValueError:
        return _stable_bucket(s, MODE3A_VOCAB)


def mode_s_index(val) -> int:
    if val is None:
        return 0
    s = str(val).strip()
    if not s:
        return 0
    return _stable_bucket(s, MODES_VOCAB)


def project_track_to_time(track: Dict, meas_t: Optional[float]) -> Dict:
    """Copy track kinematics to meas_t using constant velocity (Hybrid tmp_t)."""
    out = dict(track)
    if meas_t is None:
        out["_dt"] = 0.0
        return out
    t0 = track.get("kf_t")
    if t0 is None:
        t0 = get_time(track, meas_t)
    dt = float(meas_t) - float(t0)
    out["_dt"] = dt
    if dt > 0:
        out["x"] = _as_float(track.get("x")) + _as_float(track.get("vx")) * dt
        out["y"] = _as_float(track.get("y")) + _as_float(track.get("vy")) * dt
        out["z"] = _as_float(track.get("z")) + _as_float(track.get("vz")) * dt
    return out


def _xy_dist(a: Dict, b: Dict) -> float:
    dx = _as_float(a.get("x")) - _as_float(b.get("x"))
    dy = _as_float(a.get("y")) - _as_float(b.get("y"))
    return math.hypot(dx, dy)


def relative_feature_vec(a: Dict, b: Dict) -> np.ndarray:
    """12-d pair geometry + identity flags. Positions in meters."""
    p1 = np.array([_as_float(a.get("x")), _as_float(a.get("y")), _as_float(a.get("z"))], dtype=np.float32)
    p2 = np.array([_as_float(b.get("x")), _as_float(b.get("y")), _as_float(b.get("z"))], dtype=np.float32)
    v1 = np.array(
        [_as_float(a.get("vx")), _as_float(a.get("vy")), _as_float(a.get("vz"))],
        dtype=np.float32,
    )
    v2 = np.array(
        [_as_float(b.get("vx")), _as_float(b.get("vy")), _as_float(b.get("vz"))],
        dtype=np.float32,
    )
    delta = p1 - p2
    dist = float(np.linalg.norm(delta))
    v1_ok = a.get("vx") is not None or a.get("vy") is not None
    v2_ok = b.get("vx") is not None or b.get("vy") is not None
    n1 = float(np.linalg.norm(v1)) + 1e-8
    n2 = float(np.linalg.norm(v2)) + 1e-8
    cos_vel = float(np.dot(v1, v2) / (n1 * n2)) if (v1_ok and v2_ok) else 0.0
    dv_mag = abs(n1 - n2) / 1e3 if (v1_ok and v2_ok) else 0.0
    az1, az2 = math.atan2(p1[1], p1[0]), math.atan2(p2[1], p2[0])
    az_diff = abs(az1 - az2)
    if az_diff > math.pi:
        az_diff = 2 * math.pi - az_diff
    r1 = math.hypot(float(p1[0]), float(p1[1])) + 1e-8
    r2 = math.hypot(float(p2[0]), float(p2[1])) + 1e-8
    el1, el2 = math.atan2(float(p1[2]), r1), math.atan2(float(p2[2]), r2)
    m3a_1, m3a_2 = get_mode_3a(a), get_mode_3a(b)
    if m3a_1 is not None and m3a_2 is not None:
        m3a_match = 1.0 if str(m3a_1) == str(m3a_2) else -1.0
    else:
        m3a_match = 0.0
    ms_1, ms_2 = a.get("mode_s"), b.get("mode_s")
    if ms_1 is not None and ms_2 is not None:
        ms_match = 1.0 if str(ms_1) == str(ms_2) else -1.0
    else:
        ms_match = 0.0
    sid_1, sid_2 = get_sensor_id(a, -1), get_sensor_id(b, -1)
    same_sensor = 1.0 if sid_1 >= 0 and sid_1 == sid_2 else 0.0
    dt_a = _as_float(a.get("_dt"), 0.0)
    if "_dt" not in a and "_dt" not in b:
        dt_a = get_time(a, 0.0) - get_time(b, 0.0)
    return np.array(
        [
            delta[0] / 1e5,
            delta[1] / 1e5,
            delta[2] / 1e5,
            dist / 1e5,
            dv_mag,
            cos_vel,
            az_diff,
            abs(el1 - el2),
            float(dt_a),
            m3a_match,
            ms_match,
            same_sensor,
        ],
        dtype=np.float32,
    )


def _numeric_row(item: Dict, role_is_track: bool) -> np.ndarray:
    vx, vy, vz = item.get("vx"), item.get("vy"), item.get("vz")
    amp = item.get("amplitude")
    return np.array(
        [
            _as_float(item.get("x")) / 1e5,
            _as_float(item.get("y")) / 1e5,
            _as_float(item.get("z")) / 1e5,
            _as_float(vx) / 1e3,
            _as_float(vy) / 1e3,
            _as_float(vz) / 1e3,
            1.0 if vx is not None else 0.0,
            1.0 if vy is not None else 0.0,
            1.0 if vz is not None else 0.0,
            _as_float(amp) / 100.0,
            1.0 if amp is not None else 0.0,
            min(_as_float(item.get("age"), 0.0), 20.0) / 20.0 if role_is_track else 0.0,
            min(_as_float(item.get("hits"), 0.0), 20.0) / 20.0 if role_is_track else 0.0,
            _as_float(item.get("_dt"), 0.0),
            1.0 if get_mode_3a(item) is not None else 0.0,
        ],
        dtype=np.float32,
    )


def _type_index(item: Dict, role_is_track: bool) -> int:
    if role_is_track:
        return 1 if (get_mode_3a(item) or item.get("mode_s")) else 0
    return 1 if get_meas_type(item, "PSR") == "SSR" else 0


class AssociationTransformerV8(nn.Module):
    """Set matcher: self-attn within each side + MLP score on [h_i; h_j; rel_ij]."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_sensors: int = 8,
        use_self_attn: bool = True,
        rel_only: bool = False,
        gated_encode: bool = False,
        dual_heads: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.hidden_dim = hidden_dim
        self.use_self_attn = use_self_attn
        self.max_sensors = max_sensors
        self.rel_only = bool(rel_only)
        self.gated_encode = bool(gated_encode)
        self.dual_heads = bool(dual_heads)

        self.input_proj = nn.Linear(NUMERIC_DIM, hidden_dim)
        self.role_emb = nn.Embedding(2, hidden_dim)
        self.type_emb = nn.Embedding(2, hidden_dim)
        self.sensor_emb = nn.Embedding(max_sensors + 1, hidden_dim)
        self.mode3a_emb = nn.Embedding(MODE3A_VOCAB, hidden_dim)
        self.modes_emb = nn.Embedding(MODES_VOCAB, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        pair_in = REL_DIM if self.rel_only else hidden_dim * 2 + REL_DIM

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(pair_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.score_head = _head()
        self.psr_head = _head() if self.dual_heads else None
        self.ssr_head = _head() if self.dual_heads else None
        self.dustbin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("temperature", torch.tensor(1.0))

    def copy_shared_to_dual(self) -> None:
        if self.dual_heads and self.psr_head is not None and self.ssr_head is not None:
            self.psr_head.load_state_dict(self.score_head.state_dict())
            self.ssr_head.load_state_dict(self.score_head.state_dict())

    def _role_id(self, role) -> int:
        if isinstance(role, str):
            return 1 if role.lower() in ("track", "tracks") else 0
        return 1 if int(role) == 1 else 0

    def _featurize(self, items: Sequence[Dict], role, device: torch.device):
        n = len(items)
        numeric = np.zeros((n, NUMERIC_DIM), dtype=np.float32)
        role_id = self._role_id(role)
        roles = np.full((n,), role_id, dtype=np.int64)
        types = np.zeros((n,), dtype=np.int64)
        sensors = np.zeros((n,), dtype=np.int64)
        m3a = np.zeros((n,), dtype=np.int64)
        ms = np.zeros((n,), dtype=np.int64)
        role_is_track = role_id == 1
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                raw = {}
            try:
                item = normalize_measurement_dict(raw)
            except Exception:
                item = {}
            merged = dict(item)
            for k in ("x", "y", "z", "vx", "vy", "vz", "age", "hits", "kf_t", "_dt", "mode_3a", "mode3a", "mode_s"):
                if k in raw and raw[k] is not None:
                    merged[k] = raw[k]
            numeric[i] = _numeric_row(merged, role_is_track)
            types[i] = _type_index(merged, role_is_track)
            sid = get_sensor_id(merged, self.max_sensors if role_is_track else 0)
            sensors[i] = int(np.clip(sid, 0, self.max_sensors))
            m3a[i] = mode_3a_index(get_mode_3a(merged))
            ms[i] = mode_s_index(merged.get("mode_s"))
        return (
            torch.from_numpy(numeric).to(device),
            torch.from_numpy(roles).to(device),
            torch.from_numpy(types).to(device),
            torch.from_numpy(sensors).to(device),
            torch.from_numpy(m3a).to(device),
            torch.from_numpy(ms).to(device),
        )

    def encode(self, items: Sequence[Dict], role: str | int = "meas") -> torch.Tensor:
        """(N, d_model) contextualized tokens. role in {track, meas}."""
        if not items:
            device = next(self.parameters()).device
            return torch.zeros((0, self.hidden_dim), device=device)
        device = next(self.parameters()).device
        numeric, roles, types, sensors, m3a, ms = self._featurize(items, role, device)
        h = (
            self.input_proj(numeric)
            + self.role_emb(roles)
            + self.type_emb(types)
            + self.sensor_emb(sensors)
            + self.mode3a_emb(m3a)
            + self.modes_emb(ms)
        )
        if self.use_self_attn and h.shape[0] > 1:
            h = self.encoder(h.unsqueeze(0)).squeeze(0)
        return h

    def encode_cliques(
        self,
        items: Sequence[Dict],
        pair_index: torch.Tensor,
        role: str = "meas",
    ) -> torch.Tensor:
        """Self-attn inside connected components of the gated pair graph (ablation 1)."""
        n = len(items)
        device = next(self.parameters()).device
        if n == 0:
            return torch.zeros((0, self.hidden_dim), device=device)
        if pair_index is None or pair_index.numel() == 0 or n == 1:
            return self.encode(items, role)
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        ii = pair_index[:, 0].detach().cpu().numpy().astype(np.int64)
        jj = pair_index[:, 1].detach().cpu().numpy().astype(np.int64)
        used = np.unique(np.concatenate([ii, jj]))
        h = torch.zeros((n, self.hidden_dim), device=device)
        if used.size == 0:
            return h
        # components on the used-node subgraph
        remap = {int(old): k for k, old in enumerate(used.tolist())}
        u = np.array([remap[int(a)] for a in ii], dtype=np.int64)
        v = np.array([remap[int(b)] for b in jj], dtype=np.int64)
        m = used.size
        adj = csr_matrix(
            (np.ones(len(u) * 2, dtype=np.float32), (np.concatenate([u, v]), np.concatenate([v, u]))),
            shape=(m, m),
        )
        n_comp, labels = connected_components(adj, directed=False)
        for c in range(n_comp):
            local = np.where(labels == c)[0]
            orig = used[local].tolist()
            sub = [items[int(i)] for i in orig]
            h_sub = self.encode(sub, role)
            h[torch.tensor(orig, dtype=torch.long, device=device)] = h_sub
        return h

    def _pair_feat(self, h_left: torch.Tensor, h_right: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        if self.rel_only:
            return rel
        return torch.cat([h_left, h_right, rel], dim=-1)

    def _apply_head(self, feat: torch.Tensor, psr_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.dual_heads and self.psr_head is not None and self.ssr_head is not None:
            logits = self.ssr_head(feat).squeeze(-1)
            if psr_mask is not None:
                logits = torch.where(psr_mask, self.psr_head(feat).squeeze(-1), logits)
        else:
            logits = self.score_head(feat).squeeze(-1)
        temp = float(self.temperature.clamp_min(1e-3).item()) if self.temperature.ndim == 0 else float(self.temperature.reshape(-1)[0].clamp_min(1e-3).item())
        return logits / temp

    def _psr_mask_pairs(self, left: Sequence[Dict], right: Sequence[Dict], ii, jj, device) -> torch.Tensor:
        mask = []
        for i, j in zip(ii, jj):
            a, b = left[int(i)], right[int(j)]
            t1 = get_meas_type(a, "PSR")
            t2 = get_meas_type(b, "PSR")
            # tracks: SSR if they carry identity
            if a.get("mode_3a") or a.get("mode3a") or a.get("mode_s"):
                t1 = "SSR"
            if b.get("mode_3a") or b.get("mode3a") or b.get("mode_s"):
                t2 = "SSR"
            mask.append(t1 == "PSR" and t2 == "PSR")
        return torch.tensor(mask, dtype=torch.bool, device=device)

    def _pair_logits(
        self,
        h_left: torch.Tensor,
        h_right: torch.Tensor,
        rel: torch.Tensor,
        psr_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._apply_head(self._pair_feat(h_left, h_right, rel), psr_mask)

    def score_pairs(
        self,
        left: Sequence[Dict],
        right: Optional[Sequence[Dict]] = None,
        pair_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(P,) logits for gated pairs. Used by _spatial_cluster."""
        if pair_index is None and torch.is_tensor(right):
            pair_index = right
            right = left
        if right is None:
            right = left
        if pair_index is None or pair_index.numel() == 0:
            return torch.zeros(0, device=next(self.parameters()).device)
        same = left is right
        ii = pair_index[:, 0].long()
        jj = pair_index[:, 1].long()
        if self.rel_only:
            device = next(self.parameters()).device
            h_l = torch.zeros((len(left), self.hidden_dim), device=device)
            h_r = h_l if same else torch.zeros((len(right), self.hidden_dim), device=device)
        elif self.gated_encode:
            h_l = self.encode_cliques(left, pair_index, role="meas")
            h_r = h_l if same else self.encode_cliques(right, pair_index, role="meas")
        else:
            h_l = self.encode(left, role="meas")
            h_r = h_l if same else self.encode(right, role="meas")
        rel_np = np.stack(
            [relative_feature_vec(left[int(i)], right[int(j)]) for i, j in zip(ii.tolist(), jj.tolist())]
        )
        rel = torch.from_numpy(rel_np).to(h_l.device)
        psr_mask = self._psr_mask_pairs(left, right, ii.tolist(), jj.tolist(), h_l.device) if self.dual_heads else None
        return self._pair_logits(h_l[ii], h_r[jj], rel, psr_mask)

    def score_assignment(
        self,
        tracks: Sequence[Dict],
        metas: Sequence[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S: (T, M) logits, dustbin: (T,) logits."""
        device = next(self.parameters()).device
        t_n, m_n = len(tracks), len(metas)
        if t_n == 0:
            return torch.zeros((0, m_n), device=device), torch.zeros((0,), device=device)
        h_t = self.encode(tracks, role="track")
        dust = self.dustbin_head(h_t).squeeze(-1) / float(self.temperature.clamp_min(1e-3).reshape(-1)[0].item())
        if m_n == 0:
            return torch.zeros((t_n, 0), device=device), dust
        h_m = self.encode(metas, role="meas")
        rel_np = np.zeros((t_n, m_n, REL_DIM), dtype=np.float32)
        psr = np.zeros((t_n, m_n), dtype=bool)
        for i, tr in enumerate(tracks):
            t1 = "SSR" if (tr.get("mode_3a") or tr.get("mode3a") or tr.get("mode_s")) else "PSR"
            for j, meta in enumerate(metas):
                mt = get_time(meta, None)
                proj = project_track_to_time(tr, mt)
                rel_np[i, j] = relative_feature_vec(proj, meta)
                t2 = get_meas_type(meta, "PSR")
                psr[i, j] = t1 == "PSR" and t2 == "PSR"
        rel = torch.from_numpy(rel_np).to(device)
        h_i = h_t.unsqueeze(1).expand(-1, m_n, -1)
        h_j = h_m.unsqueeze(0).expand(t_n, -1, -1)
        psr_mask = torch.from_numpy(psr).to(device).reshape(-1) if self.dual_heads else None
        scores = self._pair_logits(
            h_i.reshape(-1, self.hidden_dim),
            h_j.reshape(-1, self.hidden_dim),
            rel.reshape(-1, REL_DIM),
            psr_mask,
        )
        return scores.view(t_n, m_n), dust


def load_v8(path: str | Path, device: Optional[torch.device] = None, **kwargs) -> AssociationTransformerV8:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = {}
    state = ckpt
    extra = {}
    if isinstance(ckpt, dict):
        cfg = dict(ckpt.get("config") or {})
        state = ckpt.get("model_state_dict", ckpt)
        extra = {k: ckpt[k] for k in ("temperature",) if k in ckpt}
    if "d_model" in cfg and "hidden_dim" not in cfg:
        cfg["hidden_dim"] = cfg.pop("d_model")
    if "nhead" in cfg and "num_heads" not in cfg:
        cfg["num_heads"] = cfg.pop("nhead")
    allowed = {
        "hidden_dim", "num_heads", "num_layers", "dropout", "use_self_attn",
        "rel_only", "gated_encode", "dual_heads", "max_sensors",
    }
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    cfg.update(kwargs)
    model = AssociationTransformerV8(**cfg).to(device)
    model.load_state_dict(state, strict=False)
    if model.dual_heads:
        # If the ckpt was a shared head, clone it into both specialized heads.
        has_psr = any(k.startswith("psr_head") for k in (state.keys() if hasattr(state, "keys") else []))
        if not has_psr:
            model.copy_shared_to_dual()
    if "temperature" in extra:
        model.temperature.fill_(float(extra["temperature"]))
    model.eval()
    return model


def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    balance: bool = False,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    w = alpha * (1.0 - p_t).pow(gamma)
    if balance:
        n_pos = targets.sum().clamp(min=1.0)
        n_neg = (1.0 - targets).sum().clamp(min=1.0)
        # Equal total mass on pos vs neg (real pos_weight; alpha is no longer a class prior).
        w = w * torch.where(targets > 0.5, n_neg / n_pos, torch.ones_like(targets))
    return (w * bce).mean()

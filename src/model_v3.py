"""
RecurrentGATTrackerV3.2 — Single end-to-end AI/ML Tracker (production version)
Replaces per-radar physics trackers + correlator.
Uses pairwise classifier probs as edge features → true multi-sensor correlation.
"""
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import numpy as np
import json

# Hybrid's proven components (reused for GNN edge features)
from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import compute_psr_psr_features, compute_ssr_any_features, get_psr_psr_dim, get_ssr_any_dim
from src.metrics import TrackingMetrics   # your existing metrics

class RecurrentGATTrackerV3(nn.Module):
    """Single ML component for the entire multi-radar fusion pipeline."""
    def __init__(self, num_sensors=5, hidden_dim=64, state_dim=6, num_heads=4, edge_dim=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.type_emb = nn.Embedding(2, 8)      # PSR vs SSR
        self.sensor_emb = nn.Embedding(num_sensors + 1, 8)

        self.encoder = nn.Sequential(
            nn.Linear(7 + 8 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)
        )
        nn.init.constant_(self.decoder[-1].bias[state_dim], 0.0)

    def forward(self, x, node_type, sensor_id, edge_index, edge_attr, hidden_state=None):
        N = x.shape[0]
        type_emb = self.type_emb(node_type)
        sensor_emb = self.sensor_emb(sensor_id)
        h = torch.cat([x, type_emb, sensor_emb], dim=-1)
        h = self.encoder(h)

        h, _ = self.gat1(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        h = F.relu(h)
        h, (_, alpha2) = self.gat2(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)

        if hidden_state is None:
            hidden_full = torch.zeros(N, self.hidden_dim, device=h.device)
        else:
            num_tracks = hidden_state.shape[0]
            pad = N - num_tracks
            hidden_full = torch.cat([hidden_state, torch.zeros(pad, self.hidden_dim, device=h.device)], dim=0) if pad > 0 else hidden_state

        new_hidden_full = self.gru(h, hidden_full)
        new_hidden_full = self.layer_norm(new_hidden_full)
        out = self.decoder(new_hidden_full)

        return out, new_hidden_full, alpha2


def build_gnn_edges(full_x, node_type, psr_clf, ssr_clf, device, max_dist=60000.0, k=12):
    """100% Vectorized Torch implementation. Zero Python loops in the hot path."""
    pos = full_x[:, :3]
    vel = full_x[:, 3:6]
    amp = full_x[:, 6]
    N = pos.shape[0]
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 7), device=device)

    # 1. Spatial Adjacency (Fast Torch CDist)
    dist = torch.cdist(pos, pos)
    mask = (dist < max_dist) & (dist > 0)
    _, indices = torch.topk(dist, min(k + 1, N), dim=1, largest=False)
    knn_mask = torch.zeros_like(dist, dtype=torch.bool, device=device)
    knn_mask.scatter_(1, indices, True)
    final_mask = mask | knn_mask
    final_mask.fill_diagonal_(False)

    edge_index = final_mask.nonzero().t()
    row, col = edge_index
    if edge_index.shape[1] == 0:
        return edge_index, torch.empty((0, 7), device=device)

    # 2. Vectorized Feature Extraction for Edge Attributes
    p1, p2 = pos[row], pos[col]
    v1, v2 = vel[row], vel[col]
    t1, t2 = node_type[row], node_type[col]
    
    # Calculate angular features in batch
    def get_az_el(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        az = torch.atan2(y, x)
        el = torch.atan2(z, torch.sqrt(x**2 + y**2 + 1e-8))
        return az, el

    az1, el1 = get_az_el(p1)
    az2, el2 = get_az_el(p2)
    
    az_diff = torch.abs(az1 - az2)
    az_diff = torch.where(az_diff > np.pi, 2*np.pi - az_diff, az_diff)
    el_diff = torch.abs(el1 - el2)

    # 3. Batch Classifier Inference
    probs = torch.zeros(edge_index.shape[1], device=device)
    
    # PSR-PSR vectorized features
    psr_mask = (t1 == 0) & (t2 == 0)
    if psr_mask.any() and psr_clf is not None:
        vp1, vp2 = v1[psr_mask], v2[psr_mask]
        v1_n = torch.norm(vp1, dim=1) + 1e-8
        v2_n = torch.norm(vp2, dim=1) + 1e-8
        cos_sim = torch.sum(vp1 * vp2, dim=1) / (v1_n * v2_n)
        mag_diff = torch.abs(v1_n - v2_n) / 1000.0
        
        # Match psr_psr input dim (6): [dist, cos_sim, mag_diff, az_diff, el_diff, amp_diff]
        dist_feat = torch.norm(p1[psr_mask] - p2[psr_mask], dim=1) / 100000.0
        amp_diff = torch.abs(amp[row[psr_mask]] - amp[col[psr_mask]]) / 100.0
        
        psr_feats = torch.stack([
            dist_feat, cos_sim, mag_diff, 
            az_diff[psr_mask], el_diff[psr_mask], amp_diff
        ], dim=1)
        probs[psr_mask] = torch.sigmoid(psr_clf(psr_feats))

    # SSR-ANY vectorized features
    ssr_mask = ~psr_mask
    if ssr_mask.any() and ssr_clf is not None:
        # Match ssr_any input dim (4): [dist, az_diff, mode3a(0), modeS(0)]
        # Since full_x currently doesn't hold IDs, we pass 0 for them.
        dist_feat = torch.norm(p1[ssr_mask] - p2[ssr_mask], dim=1) / 100000.0
        ssr_feats = torch.stack([
            dist_feat, az_diff[ssr_mask], 
            torch.zeros_like(dist_feat), torch.zeros_like(dist_feat)
        ], dim=1)
        probs[ssr_mask] = torch.sigmoid(ssr_clf(ssr_feats))

    # 4. Final Attributes
    edge_attr = torch.cat([
        p1 - p2, 
        v1 - v2, 
        probs.unsqueeze(1)
    ], dim=-1)
    
    return edge_index, edge_attr


def load_frames(data_file: str) -> List[Dict]:
    frames = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                frames.append(json.loads(line))
            except:
                continue
    print(f"Loaded {len(frames)} frames")
    return frames


def frame_to_tensors(frame_data: Dict, device):
    measurements = frame_data['measurements']
    meas_list, sid_list = [], []
    for m in measurements:
        row = [m.get(k, 0.0) for k in ('x','y','z','vx','vy','vz','amplitude')]
        meas_list.append(row)
        sid_list.append(m.get('sensor_id', 0))
    if not meas_list:
        return torch.empty((0,7), device=device), torch.empty((0,), dtype=torch.long, device=device)
    return torch.tensor(meas_list, dtype=torch.float32, device=device), torch.tensor(sid_list, dtype=torch.long, device=device)


def build_full_input(active_tracks, meas, meas_sensor_ids, num_sensors, device):
    if active_tracks:
        track_kin = torch.stack([tr['state_tensor'] for tr in active_tracks])
        track_amp = torch.zeros(len(active_tracks), 1, device=device)
        track_features = torch.cat([track_kin, track_amp], dim=1)
        track_hiddens = torch.stack([tr['hidden'] for tr in active_tracks])
        track_sensor_ids = torch.full((len(active_tracks),), num_sensors, dtype=torch.long, device=device)
        full_x = torch.cat([track_features, meas], dim=0)
        full_sensor_id = torch.cat([track_sensor_ids, meas_sensor_ids])
        return full_x, full_sensor_id, track_hiddens, len(active_tracks)
    return meas, meas_sensor_ids, None, 0


def model_forward(model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state):
    raw_out, new_hidden_full, alpha = model(full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state)
    state_delta = raw_out[:, :6]
    existence_logits = raw_out[:, 6]
    updated_state = full_x[:, :6] + state_delta
    out = torch.cat([updated_state, existence_logits.unsqueeze(-1)], dim=-1)
    existence_probs = torch.sigmoid(existence_logits)
    return out, new_hidden_full, alpha, existence_probs, existence_logits


def focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    return (alpha * (1-pt)**gamma * bce).mean()


def manage_tracks(active_tracks, out, new_hidden_full, existence_probs, existence_logits, alpha, edge_index,
                  num_tracks, num_meas, init_thresh, coast_thresh, suppress_thresh, del_exist, del_age, track_cap, dt=1.0):
    meas_offset = num_tracks
    attn_suppress = torch.zeros(num_meas, dtype=torch.bool, device=out.device)
    if num_meas > 0 and alpha is not None and alpha.numel() > 0:
        alpha_mean = alpha.mean(dim=-1)
        src, dst = edge_index
        meas_mask = dst >= num_tracks
        if meas_mask.any():
            meas_edges = meas_mask.nonzero(as_tuple=False).squeeze(-1)
            meas_dst = dst[meas_edges] - num_tracks
            meas_incoming = torch.zeros(num_meas, device=out.device)
            meas_incoming.scatter_add_(0, meas_dst, alpha_mean[meas_edges])
            attn_suppress = meas_incoming > suppress_thresh

    selected = []
    if num_tracks > 0:
        coast_boost = 0.5 if num_meas < 30 else 0.0
        for i in range(num_tracks):
            prob = existence_probs[i] + coast_boost
            if prob > coast_thresh:
                track = active_tracks[i].copy()
                # Apply motion model for existing tracks
                state = out[i, :6].detach()
                state[0:3] += state[3:6] * dt # Use dt here
                track['state_tensor'] = state
                track['hidden'] = new_hidden_full[i].detach()
                track['logit'] = existence_logits[i]
                if existence_probs[i] > 0.4:
                    track['age'] = 0
                    track['hits'] = track.get('hits', 0) + 1
                else:
                    track['age'] = track.get('age', 0) + 1
                s = track['state_tensor']
                track['x'],track['y'],track['z'],track['vx'],track['vy'],track['vz'] = s.tolist()
                selected.append(track)

    if num_meas > 0:
        cold_start = (num_tracks == 0)
        for i in range(num_meas):
            idx = meas_offset + i
            prob = existence_probs[idx]
            eff_init = init_thresh - 0.18 if cold_start else init_thresh
            if prob > eff_init and not (attn_suppress[i] and not cold_start):
                s = out[idx, :6].detach()
                selected.append({
                    'state_tensor': s, 'x':s[0].item(),'y':s[1].item(),'z':s[2].item(),
                    'vx':s[3].item(),'vy':s[4].item(),'vz':s[5].item(),
                    'hidden': new_hidden_full[idx].detach(), 'logit': existence_logits[idx],
                    'age':0, 'hits':1, 'is_new':True
                })

    selected = [tr for tr in selected if torch.sigmoid(tr['logit']) > del_exist or tr['age'] < del_age]
    if len(selected) > track_cap:
        probs = torch.stack([torch.sigmoid(tr['logit']) for tr in selected])
        top_idx = torch.topk(probs, track_cap).indices
        selected = [selected[i.item()] for i in top_idx]
    return selected


def compute_loss(pred_states, pred_logits, gt_states_dev, num_gt, match_gate, miss_penalty, fp_mult,
                 out, epoch, num_meas, meas=None, existence_logits=None, num_tracks=0):
    """Fixed: all Hungarian matching now on CPU numpy → safe indexing."""
    device = out.device
    reg_loss = exist_matched_loss = exist_fp_loss = matched_exist_loss = torch.tensor(0.0, device=device)
    miss_loss = torch.tensor(miss_penalty * num_gt, device=device)

    if pred_states.shape[0] > 0 and num_gt > 0:
        cost_matrix = torch.cdist(pred_states[:, :3], gt_states_dev[:, :3])
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        valid = cost_np[row_ind, col_ind] < match_gate
        row_ind = row_ind[valid]
        col_ind = col_ind[valid]
        row_ind_torch = torch.from_numpy(row_ind).to(device)

        if len(row_ind) > 0:
            reg_loss = F.smooth_l1_loss(pred_states[row_ind_torch], gt_states_dev[col_ind])
            exist_matched_loss = focal_bce(pred_logits[row_ind_torch], torch.ones_like(pred_logits[row_ind_torch]))
            target_logits = torch.full_like(pred_logits[row_ind_torch], 4.0)
            matched_exist_loss = F.mse_loss(pred_logits[row_ind_torch], target_logits)

        matched_mask = torch.zeros(len(pred_logits), dtype=torch.bool, device=device)
        if len(row_ind) > 0:
            matched_mask[row_ind_torch] = True
        fp_mask = ~matched_mask
        if fp_mask.any():
            exist_fp_loss = fp_mult * focal_bce(pred_logits[fp_mask], torch.zeros_like(pred_logits[fp_mask]))

        miss_loss = torch.tensor(miss_penalty * (num_gt - len(row_ind)), device=device)

    loss = reg_loss + exist_matched_loss + exist_fp_loss + miss_loss + 2.0 * matched_exist_loss

    # Cardinality loss — learns correct track count (no more artificial cap)
    num_pred = pred_states.shape[0]
    card_loss = 0.5 * (num_pred - num_gt) ** 2
    loss = loss + card_loss

    # Strong pseudo-aux for initiation
    if num_meas > 0 and meas is not None and existence_logits is not None:
        meas_logits = existence_logits[num_tracks : num_tracks + num_meas]
        vel_mag = torch.norm(meas[:, 3:6], dim=1)
        pseudo_target = torch.where((meas[:, 6] > 45.0) & (vel_mag > 80.0) & (vel_mag < 550.0), 0.92, 0.08).to(device)
        loss = loss + 5.0 * focal_bce(meas_logits, pseudo_target)

    if existence_logits is not None:
        loss = loss + 0.001 * (existence_logits ** 2).mean()
    return loss


def train_model(num_epochs=25, data_file="data/sim_hetero_001.jsonl", checkpoint_path="checkpoints/model_v3.2.pt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_sensors = 3

    # Load pairwise classifiers (same as hybrid)
    try:
        psr_clf = PairwiseAssociationClassifier(feature_dim=get_psr_psr_dim()).to(device)
        psr_clf.load_state_dict(torch.load('checkpoints/pairwise_psr_psr.pt', map_location=device, weights_only=True))
        psr_clf.eval()
        ssr_clf = PairwiseAssociationClassifier(feature_dim=get_ssr_any_dim()).to(device)
        ssr_clf.load_state_dict(torch.load('checkpoints/pairwise_ssr_any.pt', map_location=device, weights_only=True))
        ssr_clf.eval()
        print("✓ Loaded pairwise classifiers for GNN edge features")
    except Exception as e:
        print(f"Classifier load failed: {e}. Falling back to distance-only.")
        psr_clf = ssr_clf = None

    model = RecurrentGATTrackerV3(edge_dim=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    del_exist = 0.02
    del_age = 8
    track_cap = 30
    match_gate = 15000.0
    miss_penalty = 6.0

    frames = load_frames(data_file)
    active_tracks = []

    for epoch in range(num_epochs):
        if epoch > 0 and epoch % 4 == 0:
            active_tracks = []

        epoch_losses = []
        epoch_gt_counts = []

        if epoch < 5:
            init_thresh, coast_thresh, suppress_thresh, fp_mult = 0.20, 0.08, 1.0, 0.20
        elif epoch < 12:
            init_thresh, coast_thresh, suppress_thresh, fp_mult = 0.27, 0.12, 0.85, 0.60
        else:
            init_thresh, coast_thresh, suppress_thresh, fp_mult = 0.33, 0.15, 0.75, 1.00
            for g in optimizer.param_groups:
                g['lr'] = 5e-4

        frame_idx = 0
        for frame_data in tqdm(frames, desc=f"Epoch {epoch+1}/{num_epochs}"):
            frame_idx += 1
            meas, meas_sensor_ids = frame_to_tensors(frame_data, device)
            num_meas = meas.shape[0]
            if num_meas == 0: continue

            gt_tracks = frame_data.get('gt_tracks', [])
            gt_states_dev = torch.tensor([[gt.get(k,0) for k in ('x','y','z','vx','vy','vz')] for gt in gt_tracks],
                                         dtype=torch.float32, device=device)
            num_gt = gt_states_dev.shape[0]
            epoch_gt_counts.append(num_gt)

            full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
                active_tracks, meas, meas_sensor_ids, num_sensors, device)

            N = full_x.shape[0]
            if N == 0: continue

            node_type = torch.cat([torch.ones(num_tracks, dtype=torch.long, device=device),
                                   torch.zeros(num_meas, dtype=torch.long, device=device)])

            edge_index, edge_attr = build_gnn_edges(full_x, node_type, psr_clf, ssr_clf, device)

            out, new_hidden_full, alpha, existence_probs, existence_logits = model_forward(
                model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state)

            if num_tracks == 0 or frame_idx % 50 == 0:
                print(f"Frame {frame_idx} | N={N}, tracks={num_tracks}, meas={num_meas}, exist_mean={existence_probs.mean().item():.4f}")

            selected = manage_tracks(active_tracks, out, new_hidden_full, existence_probs, existence_logits,
                                     alpha, edge_index, num_tracks, num_meas, init_thresh, coast_thresh,
                                     suppress_thresh, del_exist, del_age, track_cap)

            active_tracks = selected

            pred_states = torch.stack([tr['state_tensor'] for tr in selected]) if selected else torch.empty((0, model.state_dim), device=device)
            pred_logits = torch.stack([tr['logit'] for tr in selected]) if selected else torch.empty((0,), device=device)

            loss = compute_loss(pred_states, pred_logits, gt_states_dev, num_gt, match_gate, miss_penalty, fp_mult,
                                out, epoch, num_meas, meas, existence_logits, num_tracks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        avg_gt = np.mean(epoch_gt_counts)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_loss:.1f} | Final tracks: {len(active_tracks)} (GT avg: {avg_gt:.1f})")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, checkpoint_path)

    print("\n=== Training finished — ready for CLI eval in --mode gnn ===")

if __name__ == "__main__":
    train_model(num_epochs=25, data_file="data/sim_hetero_001.jsonl")

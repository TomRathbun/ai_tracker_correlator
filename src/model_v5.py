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

class RecurrentGATTrackerV5(nn.Module):
    """V5 Architecture: Includes Early Clutter Head and Bipartite Cross-Attention style fusion."""
    def __init__(self, num_sensors=5, hidden_dim=64, state_dim=6, num_heads=4, edge_dim=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.type_emb = nn.Embedding(2, 8)      # PSR vs SSR
        self.sensor_emb = nn.Embedding(num_sensors + 1, 8)

        # Now 8 physics features: [x, y, z, vx, vy, vz, amplitude, dt]
        self.encoder = nn.Sequential(
            nn.Linear(8 + 8 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.clutter_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
        # Initialize clutter head to output negative logits (presume real initially)
        nn.init.constant_(self.clutter_head[-1].bias, -2.0)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 2) # +2: survival, initiation
        )
        nn.init.constant_(self.decoder[-1].bias[state_dim], 0.0)
        nn.init.constant_(self.decoder[-1].bias[state_dim + 1], -2.0)

    def forward(self, x, node_type, sensor_id, edge_index, edge_attr, hidden_state=None, clutter_thresh=0.70):
        N = x.shape[0]
        type_emb = self.type_emb(node_type)
        sensor_emb = self.sensor_emb(sensor_id)
        h = torch.cat([x, type_emb, sensor_emb], dim=-1)
        h = self.encoder(h)

        # V5 Upgrade: Early Clutter Identification Head
        clutter_logits = self.clutter_head(h).squeeze(-1)
        clutter_probs = torch.sigmoid(clutter_logits)
        
        # HARD DROP MASKING: Completely sever noisy measurements from the GNN edge list so they cannot pollute attention
        clean_nodes = (node_type == 1) | ((node_type == 0) & (clutter_probs < clutter_thresh))
        
        src, dst = edge_index
        valid_edges = clean_nodes[src] & clean_nodes[dst]
        edge_index = edge_index[:, valid_edges]
        edge_attr = edge_attr[valid_edges]
        
        # Soft-gate multiplier acting as a final fail-safe for node residual addition
        keep_mask = 1.0 - (clutter_probs * (node_type == 0).float())
        h = h * keep_mask.unsqueeze(-1)
        
        # Valid bounds check for PyG structure integrity against index out-of-bounds exception
        if edge_index.numel() > 0:
            assert edge_index.max() < N, f"Graph Edge Out-Of-Bounds: edge_index={edge_index.max()} exceeds N={N}"
            assert (edge_index >= 0).all(), f"Graph Edge Negative Bounds: graph integrity comprised."

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

        return out, new_hidden_full, alpha2, clutter_logits, edge_index


def build_gnn_edges(full_x, node_type, device, max_dist=60000.0, k=12):
    """100% Vectorized Torch implementation. Zero Python loops in the hot path.
    V4: Replaces frozen classifier probs with raw features for end-to-end learning."""
    pos = full_x[:, :3]
    vel = full_x[:, 3:6]
    dt_feat = full_x[:, 7]
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
    dt1, dt2 = dt_feat[row], dt_feat[col]

    # 3. Final Attributes (End-to-End Edge Embedding)
    # The GATv2 layer will take these 7 raw features and learn the association
    # probabilities internally via its attention mechanism.
    edge_attr = torch.cat([
        p1 - p2, 
        v1 - v2, 
        (dt1 - dt2).unsqueeze(1)
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


def frame_to_tensors(frame_data: Dict, device, window_t=None):
    measurements = frame_data['measurements']
    meas_list, sid_list = [], []
    for m in measurements:
        # Calculate dt: how many seconds before the window end did this hit occur?
        t_offset = (window_t - m['t']) if window_t is not None else 0.0
        row = [m.get(k, 0.0) for k in ('x','y','z','vx','vy','vz','amplitude')]
        row.append(t_offset)
        meas_list.append(row)
        sid_list.append(m.get('sensor_id', 0))
    if not meas_list:
        return torch.empty((0,8), device=device), torch.empty((0,), dtype=torch.long, device=device)
    return torch.tensor(meas_list, dtype=torch.float32, device=device), torch.tensor(sid_list, dtype=torch.long, device=device)


def build_full_input(active_tracks, meas, meas_sensor_ids, num_sensors, device):
    if active_tracks:
        track_kin = torch.stack([tr['state_tensor'] for tr in active_tracks])
        # [x, y, z, vx, vy, vz, amp, dt]
        # Tracks are predicted to the window end, so dt = 0
        extra = torch.zeros(len(active_tracks), 2, device=device) 
        track_features = torch.cat([track_kin, extra], dim=1)
        track_hiddens = torch.stack([tr['hidden'] for tr in active_tracks])
        track_sensor_ids = torch.full((len(active_tracks),), num_sensors, dtype=torch.long, device=device)
        full_x = torch.cat([track_features, meas], dim=0)
        full_sensor_id = torch.cat([track_sensor_ids, meas_sensor_ids])
        return full_x, full_sensor_id, track_hiddens, len(active_tracks)
    return meas, meas_sensor_ids, None, 0


def model_forward(model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state, clutter_thresh=0.70):
    raw_out, new_hidden_full, alpha, clutter_logits, pruned_edge_index = model(full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state, clutter_thresh)
    state_delta = raw_out[:, :6]
    survival_logits = raw_out[:, 6]
    init_logits = raw_out[:, 7]
    
    existence_logits = torch.where(node_type == 1, survival_logits, init_logits)
    
    updated_state = full_x[:, :6] + state_delta
    out = torch.cat([updated_state, existence_logits.unsqueeze(-1)], dim=-1)
    existence_probs = torch.sigmoid(existence_logits)
    clutter_probs = torch.sigmoid(clutter_logits)
    return out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits, pruned_edge_index


def focal_bce(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1-pt)**gamma * bce
    if reduction == 'mean':
        return loss.mean()
    return loss


def manage_tracks(active_tracks, out, new_hidden_full, existence_probs, existence_logits, clutter_probs, alpha, edge_index,
                  num_tracks, num_meas, init_thresh, coast_thresh, suppress_thresh, del_exist, del_age, track_cap, dt=0.0, clutter_thresh=0.70):
    """
    Fixed: Removed automatic motion step during state update (dt=0).
    The GNN 'out' is now the refined state at the END of the window.
    The Pipeline handles the physics prediction between windows.
    """
    meas_offset = num_tracks
    actual_meas_nodes = out.shape[0] - num_tracks
    attn_suppress = torch.zeros(actual_meas_nodes, dtype=torch.bool, device=out.device)
    if actual_meas_nodes > 0 and alpha is not None and alpha.numel() > 0:
        alpha_mean = alpha.mean(dim=-1)
        src, dst = edge_index
        meas_mask = dst >= num_tracks
        if meas_mask.any():
            meas_edges = meas_mask.nonzero(as_tuple=False).squeeze(-1)
            meas_dst = dst[meas_edges] - num_tracks
            meas_incoming = torch.zeros(actual_meas_nodes, device=out.device)
            meas_incoming.scatter_add_(0, meas_dst, alpha_mean[meas_edges])
            attn_suppress = meas_incoming > suppress_thresh

    selected = []
    if num_tracks > 0:
        coast_boost = 0.5 if num_meas < 30 else 0.0
        for i in range(num_tracks):
            prob = existence_probs[i] + coast_boost
            if prob > coast_thresh:
                track = active_tracks[i].copy()
                # Apply motion model ONLY if explicit dt > 0 is provided (rarely used during update)
                state = out[i, :6].detach()
                if dt > 0:
                    state[0:3] += state[3:6] * dt
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
        
        # Strict bounds checking to pinpoint alignment failure
        total_required_length = meas_offset + num_meas
        assert existence_probs.dim() == 1, f"existence_probs has unexpected dimensions {existence_probs.shape}"
        assert total_required_length <= existence_probs.size(0), \
            f"V5 Alignment Error: len(existence_probs)={existence_probs.size(0)} but requires up to {total_required_length-1}. " \
            f"(num_tracks={num_tracks}, num_meas={num_meas}, dummy_added?)"

        # Sort measurements by probability to pick the best hit for initiation
        meas_indices = list(range(num_meas))
        meas_indices.sort(key=lambda i: existence_probs[meas_offset + i].item(), reverse=True)
        
        initiated_centroids = []
        for i in meas_indices:
            idx = meas_offset + i
            prob = existence_probs[idx]
            
            # Use lower threshold during cold start to ensure we catch initial targets
            eff_init = init_thresh - 0.18 if cold_start else init_thresh
            if prob > eff_init:
                if clutter_probs[idx] > clutter_thresh: continue # V5 Clutter Head Threshold
                s = out[idx, :6].detach()
                
                # Spatial Guard: Don't start multiple tracks for the same target in one window
                # Multi-sensor detections will all refine to roughly the same state
                already_covered = False
                for centroid in initiated_centroids:
                    dist = torch.norm(s[:3] - centroid[:3])
                    if dist < 7500.0:  # 7.5km clustering radius
                        already_covered = True
                        break
                
                if already_covered:
                    continue
                
                # Attention Guard: Existing tracks override new initiations
                suppressed = attn_suppress[i] and not cold_start
                if not suppressed:
                    # Proceed with initiation
                    next_id = 0
                    if active_tracks:
                        max_id = max([tr.get('id', -1) for tr in active_tracks])
                        next_id = max_id + 1
                    
                    selected.append({
                        'id': next_id + idx,
                        'state_tensor': s, 'x':s[0].item(),'y':s[1].item(),'z':s[2].item(),
                        'vx':s[3].item(),'vy':s[4].item(),'vz':s[5].item(),
                        'hidden': new_hidden_full[idx].detach(), 'logit': existence_logits[idx],
                        'age':0, 'hits':1, 'is_new':True
                    })
                    initiated_centroids.append(s)

    # Strict Deletion: Must have high probability OR be very young (to survive temporary drops)
    # Raising del_exist to 0.40 and reducing del_age to 2 to kill persistence
    selected = [tr for tr in selected if (torch.sigmoid(tr['logit']) > del_exist) or (tr['age'] < 1 and torch.sigmoid(tr['logit']) > 0.15)]
    if len(selected) > track_cap:
        probs = torch.stack([torch.sigmoid(tr['logit']) for tr in selected])
        top_idx = torch.topk(probs, track_cap).indices
        selected = [selected[i.item()] for i in top_idx]
    return selected


def compute_loss(pred_states, pred_logits, gt_states_dev, num_gt, match_gate, miss_penalty, fp_mult,
                 out, epoch, num_meas, meas=None, existence_logits=None, clutter_logits=None, num_tracks=0, 
                 pred_ages=None, aux_init_weight=5.0):
    """Fixed: all Hungarian matching now on CPU numpy → safe indexing."""
    device = out.device
    reg_loss = exist_matched_loss = exist_fp_loss = matched_exist_loss = torch.tensor(0.0, device=device)
    clutter_loss = torch.tensor(0.0, device=device)
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
            fp_logits = pred_logits[fp_mask]
            fp_bce = focal_bce(fp_logits, torch.zeros_like(fp_logits), reduction='none')
            
            if pred_ages is not None:
                # Ghost tracks (high age) get penalized more harshly to stop persistence
                # We only apply this to the track portion of the fp_mask
                fp_tracks_mask = fp_mask[:len(pred_ages)]
                if fp_tracks_mask.any():
                    age_penalty = 1.0 + (pred_ages[fp_tracks_mask].float() / 3.0).clamp(0, 3.0)
                    # We need to apply this selectively to the BCE values corresponding to tracks
                    track_fp_bce = fp_bce[:fp_tracks_mask.sum()]
                    exist_fp_loss = fp_mult * (track_fp_bce * age_penalty).mean()
                    # Add remaining (new seeds) without age penalty
                    if fp_bce.shape[0] > fp_tracks_mask.sum():
                        exist_fp_loss += fp_mult * fp_bce[fp_tracks_mask.sum():].mean()
                else:
                    exist_fp_loss = fp_mult * fp_bce.mean()
            else:
                exist_fp_loss = fp_mult * fp_bce.mean()

        miss_loss = torch.tensor(miss_penalty * (num_gt - len(row_ind)), device=device)

    loss = reg_loss + exist_matched_loss + exist_fp_loss + miss_loss + 2.0 * matched_exist_loss

    # Cardinality loss — learns correct track count (no more artificial cap)
    num_pred = pred_states.shape[0]
    card_loss = 0.5 * (num_pred - num_gt) ** 2
    loss = loss + card_loss

    # Strong pseudo-aux for initiation (Decays as training progresses to avoid over-initiation)
    if num_meas > 0 and meas is not None and existence_logits is not None and aux_init_weight > 0:
        meas_logits = existence_logits[num_tracks : num_tracks + num_meas]
        vel_mag = torch.norm(meas[:, 3:6], dim=1)
        pseudo_target = torch.where((meas[:, 6] > 45.0) & (vel_mag > 80.0) & (vel_mag < 550.0), 0.92, 0.08).to(device)
        loss = loss + aux_init_weight * focal_bce(meas_logits, pseudo_target)

    # Clutter Head Loss
    if num_meas > 0 and meas is not None and clutter_logits is not None:
        meas_pos = meas[:, :3]
        if num_gt > 0:
            dists = torch.cdist(meas_pos, gt_states_dev[:, :3])
            min_dist, _ = torch.min(dists, dim=1)
            is_true_tgt = min_dist < match_gate
        else:
            is_true_tgt = torch.zeros(num_meas, dtype=torch.bool, device=device)
            
        m_clutter_logits = clutter_logits[num_tracks : num_tracks + num_meas]
        
        # Target for clutter: 1.0 if NOT close to any ground truth
        clutter_target = (~is_true_tgt).float()
        
        # INCREASED AGGRESSION: Focal loss weighted heavily to suppress clutter ghosts
        clutter_focal = focal_bce(m_clutter_logits, clutter_target, alpha=0.35, gamma=2.5)
        clutter_loss = 10.0 * clutter_focal 
        loss = loss + clutter_loss

        # Telemetry metrics
        clutter_p = torch.sigmoid(m_clutter_logits)
        metrics_dict = {
            "clutter_loss": clutter_loss.item(),
            "clutter_reject_rate": ((clutter_p > 0.7) == (~is_true_tgt)).float().mean().item()
        }
    else:
        metrics_dict = {"clutter_loss": 0, "clutter_reject_rate": 0}

    if existence_logits is not None:
        loss = loss + 0.001 * (existence_logits ** 2).mean()
    return loss, metrics_dict


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

            out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits = model_forward(
                model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state)

            if num_tracks == 0 or frame_idx % 50 == 0:
                print(f"Frame {frame_idx} | N={N}, tracks={num_tracks}, meas={num_meas}, exist_mean={existence_probs.mean().item():.4f}")

            selected = manage_tracks(active_tracks, out, new_hidden_full, existence_probs, existence_logits, clutter_probs,
                                     alpha, edge_index, num_tracks, num_meas, init_thresh, coast_thresh,
                                     suppress_thresh, del_exist, del_age, track_cap)

            active_tracks = selected

            pred_states = torch.stack([tr['state_tensor'] for tr in selected]) if selected else torch.empty((0, model.state_dim), device=device)
            pred_logits = torch.stack([tr['logit'] for tr in selected]) if selected else torch.empty((0,), device=device)

            loss = compute_loss(pred_states, pred_logits, gt_states_dev, num_gt, match_gate, miss_penalty, fp_mult,
                                out, epoch, num_meas, meas, existence_logits, clutter_logits, num_tracks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        avg_gt = np.mean(epoch_gt_counts)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_loss:.1f} | Final tracks: {len(active_tracks)} (GT avg: {avg_gt:.1f})")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss}, checkpoint_path)

    print("\n=== Training finished - ready for CLI eval in --mode gnn ===")

def build_gnn_edges(full_x, node_type, psr_clf, ssr_clf, device, max_dist=60000.0, k=12):
    """V5 Implementation (Edge building logic)"""
    pos = full_x[:, :3]
    N = pos.shape[0]
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 7), device=device)

    dist = torch.cdist(pos, pos)
    mask = (dist < max_dist) & (dist > 0)
    _, indices = torch.topk(dist, min(k + 1, N), dim=1, largest=False)
    knn_mask = torch.zeros_like(dist, dtype=torch.bool, device=device)
    knn_mask.scatter_(1, indices, True)
    final_mask = mask | knn_mask
    final_mask.fill_diagonal_(False)

    edge_index = final_mask.nonzero().t()
    row, col = edge_index
    
    # Static edge attributes (will be replaced by learned embeddings in forward)
    edge_attr = torch.zeros(edge_index.shape[1], 7, device=device)
    edge_attr[:, 0] = dist[row, col] / 1000.0 # Scale to km
    
    return edge_index, edge_attr


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
class RecurrentGATTrackerV3(nn.Module):
    def __init__(self, num_sensors=3, hidden_dim=64, state_dim=6, num_heads=4, edge_dim=6, emb_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.type_emb = nn.Embedding(2, emb_dim)
        self.sensor_emb = nn.Embedding(num_sensors + 1, emb_dim)

        self.encoder = nn.Sequential(
            nn.Linear(7 + emb_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)
        )

        # Initialize existence bias to 0.0 for neutral starting point (~0.5 probability)
        nn.init.constant_(self.decoder[-1].bias[state_dim], 0.0)

    def forward(self, x, node_type, sensor_id, edge_index, edge_attr, hidden_state=None):
        N = x.shape[0]

        type_emb = self.type_emb(node_type)
        sensor_emb = self.sensor_emb(sensor_id)

        h = torch.cat([x, type_emb, sensor_emb], dim=-1)
        h = self.encoder(h)

        h, (edge_index1, alpha1) = self.gat1(h, edge_index, edge_attr=edge_attr,
                                             return_attention_weights=True)
        h = F.relu(h)
        h, (edge_index2, alpha2) = self.gat2(h, edge_index, edge_attr=edge_attr,
                                             return_attention_weights=True)

        if hidden_state is None:
            hidden_full = torch.zeros(N, self.hidden_dim, device=h.device)
        else:
            num_tracks = hidden_state.shape[0]
            pad = N - num_tracks
            if pad > 0:
                zeros_pad = torch.zeros(pad, self.hidden_dim, device=h.device)
                hidden_full = torch.cat([hidden_state, zeros_pad], dim=0)
            elif pad < 0:
                raise ValueError("hidden_state larger than total nodes")
            else:
                hidden_full = hidden_state

        new_hidden_full = self.gru(h, hidden_full)
        out = self.decoder(new_hidden_full)

        return out, new_hidden_full, alpha2  # Return final alpha for suppression

def build_sparse_edges(x, max_dist=60000.0, k=10):
    pos = x[:, :3]
    vel = x[:, 3:6]
    N = pos.shape[0]
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=x.device), torch.empty((0, 6), device=x.device)

    dist = torch.cdist(pos, pos)

    mask = (dist < max_dist) & (dist > 0)

    effective_k = min(k + 1, N)
    _, indices = torch.topk(dist, effective_k, dim=1, largest=False)
    knn_mask = torch.zeros_like(dist, dtype=torch.bool)
    knn_mask.scatter_(1, indices, True)

    final_mask = mask | knn_mask
    final_mask.fill_diagonal_(False)

    edge_index = final_mask.nonzero().t()
    row, col = edge_index
    delta_pos = pos[row] - pos[col]
    delta_vel = vel[row] - vel[col]
    edge_attr = torch.cat([delta_pos, delta_vel], dim=-1)

    return edge_index, edge_attr

def generate_synthetic_frame(t_step, true_trajectories, num_sensors, sensor_noises,
                             p_detect=0.85, clutter_rate=10):
    meas_list = []
    sensor_list = []
    is_true_list = []
    gt_states = []

    for traj in true_trajectories:
        true_pos = traj['initial_pos'] + traj['vel'] * t_step
        true_vel = traj['vel']
        gt_states.append(torch.cat([true_pos, true_vel]))

        for s in range(num_sensors):
            if torch.rand(1) < p_detect:
                pos_noise = torch.randn(3) * sensor_noises[s]
                vel_noise = torch.randn(3) * 8.0
                amp = 60.0 + torch.randn(1) * 15.0
                m = torch.cat([true_pos + pos_noise, true_vel + vel_noise, amp])
                meas_list.append(m)
                sensor_list.append(s)
                is_true_list.append(True)

    for _ in range(clutter_rate):
        clutter_pos = torch.randn(3) * 50000.0
        clutter_vel = torch.randn(3) * 30.0
        amp = 35.0 + torch.randn(1) * 10.0
        m = torch.cat([clutter_pos, clutter_vel, amp])
        meas_list.append(m)
        sensor_list.append(torch.randint(0, num_sensors, (1,)).item())
        is_true_list.append(False)

    if meas_list:
        meas = torch.stack(meas_list)
        sensor_ids = torch.tensor(sensor_list, dtype=torch.long)
        is_true_meas = torch.tensor(is_true_list, dtype=torch.bool)
    else:
        meas = torch.empty((0, 7))
        sensor_ids = torch.empty((0,), dtype=torch.long)
        is_true_meas = torch.empty((0,), dtype=torch.bool)

    gt_states = torch.stack(gt_states) if gt_states else torch.empty((0, 6))

    return meas, sensor_ids, gt_states, is_true_meas


# 1. Data loading / preparation (one-time or per epoch)
def load_frames(data_file: str) -> List[Dict]:
    """Load all frames from jsonl into memory."""
    frames = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(frames)} frames")
    return frames

# 2. Convert one frame's measurements → tensors
def frame_to_tensors(frame_data: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse measurements → [N, 7] feature tensor + sensor_id tensor"""
    measurements = frame_data['measurements']
    meas_list = []
    sensor_ids_list = []

    for m in measurements:
        row = [
            m.get('x', 0.0), m.get('y', 0.0), m.get('z', 0.0),
            m.get('vx', 0.0), m.get('vy', 0.0), m.get('vz', 0.0),
            m.get('amplitude', 0.0)
        ]
        meas_list.append(row)
        sensor_ids_list.append(m.get('sensor_id', 0))

    if not meas_list:
        return torch.empty((0, 7), device=device), torch.empty((0,), dtype=torch.long, device=device)

    meas = torch.tensor(meas_list, dtype=torch.float32, device=device)
    sensor_ids = torch.tensor(sensor_ids_list, dtype=torch.long, device=device)
    return meas, sensor_ids

# 3. Build full input (tracks + measurements)
def build_full_input(active_tracks: List[Dict], meas: torch.Tensor, meas_sensor_ids: torch.Tensor,
                     num_sensors: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Combine persistent tracks + new measurements into full_x, full_sensor_id, hidden_state, num_tracks"""
    if active_tracks:
        track_kin_list = []
        for tr in active_tracks:
            if 'state_tensor' in tr:
                track_kin_list.append(tr['state_tensor'])
            else:
                # Reconstruct from scalar fields if state_tensor is missing (e.g. from KF mode)
                s = torch.tensor([
                    tr.get('x', 0.0), tr.get('y', 0.0), tr.get('z', 0.0),
                    tr.get('vx', 0.0), tr.get('vy', 0.0), tr.get('vz', 0.0)
                ], device=device)
                track_kin_list.append(s)
        
        track_kin = torch.stack(track_kin_list)
        track_amp = torch.zeros(len(active_tracks), 1, device=device)
        track_features = torch.cat([track_kin, track_amp], dim=1)

        track_hiddens = torch.stack([tr['hidden'] for tr in active_tracks])
        track_sensor_ids = torch.full((len(active_tracks),), num_sensors, dtype=torch.long, device=device)

        full_x = torch.cat([track_features, meas], dim=0)
        full_sensor_id = torch.cat([track_sensor_ids, meas_sensor_ids])
        hidden_state = track_hiddens
        num_tracks = len(active_tracks)
    else:
        full_x = meas
        full_sensor_id = meas_sensor_ids
        hidden_state = None
        num_tracks = 0

    return full_x, full_sensor_id, hidden_state, num_tracks

# 4. Model forward + existence probs
def model_forward(model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state):
    raw_out, new_hidden_full, alpha = model(full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state)
    
    # Residual logic: Add input state to model's predicted deltas
    # full_x features [0:6] are [x, y, z, vx, vy, vz]
    state_delta = raw_out[:, :6]
    existence_logits = raw_out[:, 6]
    
    updated_state = full_x[:, :6] + state_delta
    
    # Reassemble output
    out = torch.cat([updated_state, existence_logits.unsqueeze(-1)], dim=-1)
    
    existence_probs = torch.sigmoid(existence_logits)
    return out, new_hidden_full, alpha, existence_probs, existence_logits

# 5. Track management (selection, suppression, deletion, cap)
def manage_tracks(active_tracks, out, new_hidden_full, existence_probs, existence_logits, alpha, edge_index,
                  num_tracks, num_meas, init_thresh, coast_thresh, suppress_thresh, del_exist, del_age, track_cap):
    meas_offset = num_tracks

    # Softer suppression (only strong associations block)
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

    # Coast/Update existing
    if num_tracks > 0:
        coast_boost = 0.5 if num_meas < 30 else 0.0  # help coasting in sparse frames
        for i in range(num_tracks):
            prob = existence_probs[i] + coast_boost
            if prob > coast_thresh:
                track = active_tracks[i].copy()
                track['state_tensor'] = out[i, :6].detach()
                track['hidden'] = new_hidden_full[i].detach()
                track['logit'] = existence_logits[i]
                
                # Check if it was updated by a measurement
                # Simple check: was the probability strong?
                # Lowered from 0.8 for legacy GNN compatibility (which peaks around 0.6-0.7)
                if existence_probs[i] > 0.4:
                    track['age'] = 0
                    track['hits'] = track.get('hits', 0) + 1
                else:
                    track['age'] = track.get('age', 0) + 1
                
                # Sync tensor state to x,y,z if needed for metrics
                s = track['state_tensor']
                track['x'], track['y'], track['z'] = s[0].item(), s[1].item(), s[2].item()
                track['vx'], track['vy'], track['vz'] = s[3].item(), s[4].item(), s[5].item()
                
                selected.append(track)

    # Initiate new
    if num_meas > 0:
        for i in range(num_meas):
            idx = meas_offset + i
            prob = existence_probs[idx]
            if prob > init_thresh and not attn_suppress[i]:
                s = out[idx, :6].detach()
                selected.append({
                    'state_tensor': s,
                    'x': s[0].item(), 'y': s[1].item(), 'z': s[2].item(),
                    'vx': s[3].item(), 'vy': s[4].item(), 'vz': s[5].item(),
                    'hidden': new_hidden_full[idx].detach(),
                    'logit': existence_logits[idx],
                    'age': 0,
                    'hits': 1,
                    'is_new': True
                })

    # Softer deletion: keep young tracks even if low prob
    selected = [tr for tr in selected if torch.sigmoid(tr['logit']) > del_exist or tr['age'] < del_age]

    # Cap
    if len(selected) > track_cap:
        probs = torch.stack([torch.sigmoid(tr['logit']) for tr in selected])
        top_idx = torch.topk(probs, track_cap).indices
        selected = [selected[i.item()] for i in top_idx]

    return selected


# 6. Compute loss (using GT from frame)
def compute_loss(
    pred_states: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_states_dev: torch.Tensor,
    num_gt: int,
    match_gate: float,
    miss_penalty: float,
    fp_mult: float,
    out: torch.Tensor,
    epoch: int,
    num_meas: int,
    meas: torch.Tensor = None,          # optional for pseudo-aux
    existence_logits: torch.Tensor = None,  # optional but useful for pseudo-aux
    num_tracks: int = 0,                # optional for slicing
):
    """
    Compute the full tracking loss.
    Works even when pred_states is empty.
    """
    device = out.device

    reg_loss = torch.tensor(0.0, device=device)
    exist_matched_loss = torch.tensor(0.0, device=device)
    exist_fp_loss = torch.tensor(0.0, device=device)
    miss_loss = torch.tensor(miss_penalty * num_gt, device=device)
    matched_exist_loss = torch.tensor(0.0, device=device)

    row_ind_torch = None
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
            exist_matched_loss = F.binary_cross_entropy_with_logits(
                pred_logits[row_ind_torch],
                torch.ones_like(pred_logits[row_ind_torch])
            )

            # Matched existence target: push matched tracks' logits high
            target_logits = torch.full_like(pred_logits[row_ind_torch], 4.0)
            matched_exist_loss = F.mse_loss(pred_logits[row_ind_torch], target_logits)

        matched_mask = torch.zeros(len(pred_logits), dtype=torch.bool, device=device)
        if len(row_ind) > 0:
            matched_mask[row_ind_torch] = True
        fp_mask = ~matched_mask
        if fp_mask.any():
            exist_fp_loss = fp_mult * F.binary_cross_entropy_with_logits(
                pred_logits[fp_mask],
                torch.zeros_like(pred_logits[fp_mask])
            )

        miss_loss = torch.tensor(miss_penalty * (num_gt - len(row_ind)), device=device)

    loss = reg_loss + exist_matched_loss + exist_fp_loss + miss_loss + 2.0 * matched_exist_loss

    # Pseudo-aux on measurements (amplitude > 40 → likely true)
    if num_meas > 0 and meas is not None and existence_logits is not None:
        meas_logits = existence_logits[num_tracks : num_tracks + num_meas]
        pseudo_target = torch.where(meas[:, 5] > 40.0, 0.8, 0.2).to(device)
        pseudo_aux = F.binary_cross_entropy_with_logits(meas_logits, pseudo_target)
        loss = loss + 3.0 * pseudo_aux

    # Prevent extreme negative logits
    if existence_logits is not None:
        logit_reg = 0.001 * (existence_logits ** 2).mean()
        loss = loss + logit_reg

    return loss

# 7. Main training loop (now clean and readable)
def train_model(num_epochs=30, data_file="data/sim_realistic_003.jsonl"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_sensors = 3
    model = RecurrentGATTrackerV3(num_sensors=num_sensors).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Hyperparameters (easy to change or sweep)
    del_exist = 0.05
    del_age = 8
    track_cap = 150
    match_gate = 15000.0
    miss_penalty = 10.0

    frames = load_frames(data_file)

    for epoch in range(num_epochs):
        active_tracks = []
        epoch_losses = []
        epoch_gt_counts = []

        if epoch < 3:
            suppress_thresh = 1.0  # off
        else:
            suppress_thresh = 0.8

        if epoch < 5:
            init_thresh = 0.25
            coast_thresh = 0.1
            fp_mult = 0.2
        else:
            init_thresh = 0.35
            coast_thresh = 0.15
            fp_mult = 0.8


        frame_idx = 0
        for frame_data in tqdm(frames, desc=f"Epoch {epoch+1}/{num_epochs}"):
            frame_idx += 1
            meas, meas_sensor_ids = frame_to_tensors(frame_data, device)
            num_meas = meas.shape[0]
            if num_meas == 0:
                continue

            gt_tracks = frame_data.get('gt_tracks', [])
            gt_states_dev = torch.tensor(
                [[gt.get('x', 0.0), gt.get('y', 0.0), gt.get('z', 0.0), 
                  gt.get('vx', 0.0), gt.get('vy', 0.0), gt.get('vz', 0.0)] for gt in gt_tracks],
                dtype=torch.float32, device=device
            )
            num_gt = gt_states_dev.shape[0]
            epoch_gt_counts.append(num_gt)

            full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
                active_tracks, meas, meas_sensor_ids, num_sensors, device
            )

            N = full_x.shape[0]
            if N == 0:
                continue

            node_type = torch.cat([torch.ones(num_tracks, dtype=torch.long, device=device),
                                   torch.zeros(num_meas, dtype=torch.long, device=device)])

            edge_index, edge_attr = build_sparse_edges(full_x)

            out, new_hidden_full, alpha, existence_probs, existence_logits = model_forward(
                model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state
            )

            print(
                f"Frame {frame_idx} | N={N}, num_tracks={num_tracks}, num_meas={num_meas}, existence_probs.mean()={existence_probs.mean().item():.4f}")

            selected = manage_tracks(
                active_tracks=active_tracks,
                out=out,
                new_hidden_full=new_hidden_full,
                existence_probs=existence_probs,
                existence_logits=existence_logits,
                alpha=alpha,
                edge_index=edge_index,
                num_tracks=num_tracks,
                num_meas=num_meas,
                init_thresh=init_thresh,
                coast_thresh=coast_thresh,
                suppress_thresh=suppress_thresh,
                del_exist=del_exist,
                del_age=del_age,
                track_cap=track_cap
            )

            active_tracks = selected

            # Get pred tensors
            if len(selected) == 0:
                pred_states = torch.empty((0, model.state_dim), device=device)
                pred_logits = torch.empty((0,), device=device)
            else:
                pred_states = torch.stack([tr['state'] for tr in selected])
                pred_logits = torch.stack([tr['logit'] for tr in selected])

            loss = compute_loss(
                pred_states=pred_states,
                pred_logits=pred_logits,
                gt_states_dev=gt_states_dev,
                num_gt=num_gt,
                match_gate=match_gate,
                miss_penalty=miss_penalty,
                fp_mult=fp_mult,
                out=out,
                existence_logits=existence_logits,  # added
                num_tracks=num_tracks,  # added
                epoch=epoch,
                num_meas=num_meas,
                meas=meas  # for pseudo-aux
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        avg_gt = np.mean(epoch_gt_counts)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_loss:.4f} | Final tracks: {len(active_tracks)} "
              f"(Avg GT unique: {avg_gt:.1f})")

if __name__ == "__main__":
    train_model(num_epochs=5, data_file="data/sim_realistic_003.jsonl")
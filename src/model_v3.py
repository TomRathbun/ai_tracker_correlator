import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import numpy as np

# Model unchanged
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

    def forward(self, x, node_type, sensor_id, edge_index, edge_attr, hidden_state=None):
        N = x.shape[0]

        type_emb = self.type_emb(node_type)
        sensor_emb = self.sensor_emb(sensor_id)

        h = torch.cat([x, type_emb, sensor_emb], dim=-1)
        h = self.encoder(h)

        h, (_, alpha1) = self.gat1(h, edge_index, edge_attr=edge_attr,
                                   return_attention_weights=True)
        h = F.relu(h)
        h, (_, alpha2) = self.gat2(h, edge_index, edge_attr=edge_attr,
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

        return out, new_hidden_full, [alpha1, alpha2]

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
    gt_states = []

    for traj in true_trajectories:  # Fixed: direct iteration
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

    # Clutter (independent)
    for _ in range(clutter_rate):
        clutter_pos = torch.randn(3) * 50000.0
        clutter_vel = torch.randn(3) * 30.0
        amp = 35.0 + torch.randn(1) * 10.0
        m = torch.cat([clutter_pos, clutter_vel, amp])
        meas_list.append(m)
        sensor_list.append(torch.randint(0, num_sensors, (1,)).item())

    if meas_list:
        meas = torch.stack(meas_list)
        sensor_ids = torch.tensor(sensor_list, dtype=torch.long)
    else:
        meas = torch.empty((0, 7))
        sensor_ids = torch.empty((0,), dtype=torch.long)

    gt_states = torch.stack(gt_states) if gt_states else torch.empty((0, 6))

    return meas, sensor_ids, gt_states

def train_model(num_epochs=20, steps_per_epoch=100, num_objects=4, num_sensors=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecurrentGATTrackerV3(num_sensors=num_sensors).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    match_gate = 15000.0
    miss_penalty = 2.0

    for epoch in range(num_epochs):
        true_trajectories = []
        for _ in range(num_objects):
            init_pos = torch.randn(3) * 30000.0
            vel = torch.randn(3) * 100.0 + torch.tensor([100.0, 0.0, 0.0])
            true_trajectories.append({'initial_pos': init_pos, 'vel': vel})

        sensor_noises = [20.0 + 30.0 * i for i in range(num_sensors)]

        active_tracks = []
        epoch_losses = []

        for t in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}"):
            meas, meas_sensor_ids, gt_states = generate_synthetic_frame(
                t, true_trajectories, num_sensors, sensor_noises, clutter_rate=10)

            num_meas = meas.shape[0]

            # Build graph input
            if active_tracks:
                track_states = torch.stack([tr['state'] for tr in active_tracks])
                track_hiddens = torch.stack([tr['hidden'] for tr in active_tracks])
                track_sensor_ids = torch.full((len(active_tracks),), num_sensors, dtype=torch.long)
                full_x = torch.cat([track_states, meas.to(device)], dim=0)
                full_sensor_id = torch.cat([track_sensor_ids, meas_sensor_ids.to(device)])
                hidden_state = track_hiddens
                num_tracks = len(active_tracks)
            else:
                full_x = meas.to(device)
                full_sensor_id = meas_sensor_ids.to(device)
                hidden_state = None
                num_tracks = 0

            N = full_x.shape[0]
            if N == 0:
                # Empty scene: log miss penalty but no gradient update (no information)
                num_gt = gt_states.shape[0]
                loss_val = miss_penalty * num_gt
                epoch_losses.append(loss_val)
                continue

            node_type = torch.cat([torch.ones(num_tracks, dtype=torch.long, device=device),
                                   torch.zeros(num_meas, dtype=torch.long, device=device)])

            edge_index, edge_attr = build_sparse_edges(full_x)

            out, new_hidden_full, _ = model(full_x, node_type, full_sensor_id,
                                            edge_index.to(device), edge_attr.to(device),
                                            hidden_state)

            existence_logits = out[:, 6]
            existence_probs = torch.sigmoid(existence_logits)

            # Clean track management
            selected_states = []
            selected_hiddens = []
            selected_logits = []

            if num_tracks > 0:
                for i in range(num_tracks):
                    if existence_probs[i] > 0.3:
                        selected_states.append(out[i, :6])
                        selected_hiddens.append(new_hidden_full[i])
                        selected_logits.append(existence_logits[i])

            meas_offset = num_tracks
            if num_meas > 0:
                for i in range(num_meas):
                    idx = meas_offset + i
                    if existence_probs[idx] > 0.75:
                        selected_states.append(out[idx, :6])
                        selected_hiddens.append(new_hidden_full[idx])
                        selected_logits.append(existence_logits[idx])

            active_tracks = [{'state': s, 'hidden': h} 
                             for s, h in zip(selected_states, selected_hiddens)]

            # Supervised loss
            gt_states_dev = gt_states.to(device)
            num_gt = gt_states_dev.shape[0]

            if len(selected_states) == 0:
                pred_states = torch.empty((0, model.state_dim), device=device)
                pred_logits = torch.empty((0,), device=device)
            else:
                pred_states = torch.stack(selected_states)
                pred_logits = torch.stack(selected_logits)

            reg_loss = torch.tensor(0.0, device=device)
            exist_matched_loss = torch.tensor(0.0, device=device)
            exist_fp_loss = torch.tensor(0.0, device=device)
            miss_loss = miss_penalty * num_gt

            if pred_states.shape[0] > 0 and num_gt > 0:
                cost_matrix = torch.cdist(pred_states[:, :3], gt_states_dev[:, :3])
                cost_np = cost_matrix.cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)

                valid = cost_np[row_ind, col_ind] < match_gate
                row_ind = row_ind[valid]
                col_ind = col_ind[valid]
                row_ind_torch = torch.from_numpy(row_ind).to(device)

                if len(row_ind) > 0:
                    reg_loss = F.mse_loss(pred_states[row_ind_torch], gt_states_dev[col_ind])
                    exist_matched_loss = F.binary_cross_entropy_with_logits(
                        pred_logits[row_ind_torch], torch.ones_like(pred_logits[row_ind_torch]))

                matched_mask = torch.zeros(len(pred_logits), dtype=torch.bool, device=device)
                matched_mask[row_ind_torch] = True
                fp_mask = ~matched_mask
                if fp_mask.any():
                    exist_fp_loss = F.binary_cross_entropy_with_logits(
                        pred_logits[fp_mask], torch.zeros_like(pred_logits[fp_mask]))

                miss_loss = miss_penalty * (num_gt - len(row_ind))

            loss = reg_loss + exist_matched_loss + exist_fp_loss + miss_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        print(f"Epoch {epoch+1} complete | Avg loss: {np.mean(epoch_losses):.4f} | Final tracks: {len(active_tracks)} (GT: {num_objects})")

if __name__ == "__main__":
    train_model(num_epochs=30, steps_per_epoch=120, num_objects=5, num_sensors=4) 


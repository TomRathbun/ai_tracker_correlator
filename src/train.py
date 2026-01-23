import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import IterableDataset, DataLoader
from schema import BatchFrame, RadarPlot, RadarBeacon
from src.model_v2 import RecurrentGATTracker, build_fully_connected_edge_index
from tqdm import tqdm

class RadarDataset(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file
        
    def __iter__(self):
        with open(self.data_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Parse back to objects if needed, or just use dict
                # Here we reconstruct for clarity but could stream dicts
                yield data

def collate_fn(batch_list):
    # Since we are processing sequential frames in a loop inside train(), 
    # and "batch" here usually implies parallel samples, but for RNN we want sequence.
    # However, standard DataLoader collates into a batch of items.
    # Our "item" is one Frame (which contains N measurements).
    # We will just return the list of frames and handle tensor construction in the loop
    # or construct a batch of graphs here.
    
    # Simple approach: Return the single frame's data as tensors
    # Assuming batch_size=1 for streaming sequence
    frame_dict = batch_list[0] 
    measurements = frame_dict['measurements']
    
    tensor_data = []
    # Vector: [x, y, z, vx, vy, amplitude, code, sensor_id]
    
    for m in measurements:
        is_beacon = (m['type'] == 'beacon')
        
        row = [
            m['x'], m['y'], m['z'],
            m['vx'], m['vy'],
            0.0 if is_beacon else m.get('amplitude', 0.0),
            float(m.get('identity_code', 0)) if is_beacon else 0.0,
            float(m['sensor_id'])
        ]
        tensor_data.append(row)
        
    if not tensor_data:
        return torch.empty(0, 8)
        
    return torch.tensor(tensor_data, dtype=torch.float32)

def train():
    data_file = "data/sim_001.jsonl"
    if not os.path.exists(data_file):
        print(f"File {data_file} not found. Run data_generation.py first.")
        return

    # Setup
    dataset = RadarDataset(data_file)
    # batch_size=1 because each line is already a "batch" of simultaneous readings (a frame)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn) 
    
    # Input dim 8 now
    model = RecurrentGATTracker(input_dim=8, hidden_dim=64, output_dim=7)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    hidden_state = None
    losses = []
    
    print("Starting training from file...")
    for reports in tqdm(loader):
        # reports is [N, 8] tensor
        if reports.size(0) == 0:
            continue
            
        num_items = reports.size(0)
        edge_index = build_fully_connected_edge_index(num_items)
        
        # Hidden state management (naive reset if size changes)
        if hidden_state is not None and hidden_state.shape[0] != num_items:
            hidden_state = None
            
        out, hidden_state = model(reports, edge_index, hidden_state)
        hidden_state = hidden_state.detach()
        
        # Self-supervised dummy target: predict input [x,y,z,vx,vy,amp,code] (ignoring sensor_id for loss maybe?)
        # Only first 7 dims match output_dim=7
        target = reports[:, :7] 
        
        loss = criterion(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    print(f"Training complete. Final Loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    train()

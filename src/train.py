import torch
import torch.nn as nn
import torch.optim as optim
from data_generation import create_default_scenario
from model import RecurrentGATTracker, build_fully_connected_edge_index
from tqdm import tqdm

def train():
    # Setup
    gen = create_default_scenario()
    model = RecurrentGATTracker(input_dim=7, hidden_dim=64, output_dim=7)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training Loop
    model.train()
    hidden_state = None
    
    losses = []
    
    # Simulate a sequence of 50 steps
    print("Starting training sequence...")
    for t in tqdm(range(50)):
        # 1. Generate data
        # Note: in a real loop we'd have a dataloader, here we simulate on the fly
        reports, labels = gen.generate_batch(t * 3.0)
        
        if len(reports) == 0:
            continue
            
        # 2. Build graph (fully connected for now)
        num_items = reports.shape[0]
        edge_index = build_fully_connected_edge_index(num_items)
        
        # 3. Forward
        # We need to manage hidden state size mismatch if objects appear/disappear
        # For this simple feasibility demo, we'll reset hidden state if size changes (coarse approximation)
        # A real implementation requires ID-based matching or max-prop sizing.
        
        if hidden_state is not None and hidden_state.shape[0] != num_items:
            # Re-init hidden for new items or just zero out for simplicity in this demo
            # In production: strict ID matching required
            hidden_state = None # Force reset for demo
            
        out, hidden_state = model(reports, edge_index, hidden_state)
        
        # Detach hidden state to prevent backprop through entire history (TBPTT could be used)
        hidden_state = hidden_state.detach()
        
        # 4. Loss (Self-supervised / Auto-regressive reconstruction for this demo)
        # We try to predict the input itself (auto-encoder style) or next step
        # Ideally we have GT labels. Let's assume 'reports' contains GT for training if we treat it as such
        # or we generate next step GT.
        # For simplicity: reconstruction loss
        target = reports # Simple AE objective
        
        loss = criterion(out, target)
        
        # 5. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    print(f"Training complete. Final Loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    train()

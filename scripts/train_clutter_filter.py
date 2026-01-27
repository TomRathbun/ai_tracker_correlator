"""
Train the Clutter Classifier using heterogeneous simulation data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.clutter_classifier import ClutterClassifier, extract_clutter_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClutterDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def load_clutter_data(path):
    print(f"Loading data from {path}...")
    all_features = []
    all_labels = []
    
    with open(path) as f:
        for line in f:
            frame = json.loads(line)
            for m in frame.get('measurements', []):
                feats = extract_clutter_features(m)
                all_features.append(feats.numpy())
                
                # Label 1 if clutter (track_id == -1), 0 if real signal
                # Wait, user said: "Discard measurements where P(clutter) > 0.7"
                # So let's train where 1 = Clutter, 0 = Real
                is_clutter = 1.0 if m.get('track_id', -1) == -1 else 0.0
                all_labels.append(is_clutter)
                
    return np.array(all_features), np.array(all_labels)

def train():
    features, labels = load_clutter_data('data/sim_hetero_001.jsonl')
    
    # Shuffle and split
    indices = np.random.permutation(len(features))
    split = int(0.8 * len(features))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_features, train_labels = features[train_idx], labels[train_idx]
    val_features, val_labels = features[val_idx], labels[val_idx]
    
    print(f"Total: {len(features)}, Train: {len(train_features)}, Val: {len(val_features)}")
    print(f"Clutter ratio: {labels.mean():.2%}")
    
    train_loader = DataLoader(ClutterDataset(train_features, train_labels), batch_size=256, shuffle=True)
    val_loader = DataLoader(ClutterDataset(val_features, val_labels), batch_size=256, shuffle=False)
    
    writer = SummaryWriter('runs/clutter_filter_v1')
    
    model = ClutterClassifier(feature_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Weight the loss because clutter might be outnumbered or vice versa
    pos_weight = torch.tensor([float((1-train_labels).sum() / (train_labels.sum() + 1e-6))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    Path('checkpoints').mkdir(exist_ok=True)
    best_f1 = 0.0
    
    print("\nTraining Clutter Classifier...")
    for epoch in range(1, 21):
        model.train()
        train_loss = 0
        for feats, labs in train_loader:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for feats, labs in val_loader:
                feats = feats.to(device)
                probs = torch.sigmoid(model(feats))
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labs.numpy())
        
        preds = (np.array(all_probs) > 0.5).astype(float)
        labels_arr = np.array(all_labels)
        
        tp = ((preds == 1) & (labels_arr == 1)).sum()
        fp = ((preds == 1) & (labels_arr == 0)).sum()
        fn = ((preds == 0) & (labels_arr == 1)).sum()
        
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Loss: {train_loss/len(train_loader):.4f} | P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}")
            
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Metrics/Precision', p, epoch)
        writer.add_scalar('Metrics/Recall', r, epoch)
        writer.add_scalar('Metrics/F1', f1, epoch)
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'checkpoints/clutter_classifier.pt')

    writer.close()
    print(f"\n✓ Training complete. Best F1: {best_f1:.3f}")

if __name__ == "__main__":
    train()

"""
Training script for pairwise association classifier - CLEAN DATA VERSION
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import extract_all_pairs, get_feature_dim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PairwiseDataset(Dataset):
    """Dataset of pairwise features and labels"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(path):
    """Load JSONL data"""
    frames = []
    with open(path) as f:
        for line in f:
            frames.append(json.loads(line))
    return frames

def extract_pairs_from_frames(frames):
    """Extract all pairwise features and labels from frames"""
    all_features = []
    all_labels = []
    
    print(f"Extracting pairs from {len(frames)} frames...")
    for frame in tqdm(frames):
        measurements = frame.get('measurements', [])
        if len(measurements) < 2:
            continue
        
        # Extract all pairs from this frame
        feats, labels = extract_all_pairs(measurements)
        all_features.append(feats)
        all_labels.append(labels)
    
    # Concatenate all
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Total pairs: {len(all_features)}")
    print(f"Positive pairs (same object): {int(all_labels.sum())}")
    print(f"Negative pairs (different objects): {int((1-all_labels).sum())}")
    print(f"Positive ratio: {all_labels.mean():.3f}")
    
    return all_features, all_labels

def compute_metrics(preds, labels, probs):
    """Compute classification metrics"""
    preds_tensor = torch.tensor(preds)
    labels_tensor = torch.tensor(labels)
    
    # Precision, Recall, F1
    tp = ((preds_tensor == 1) & (labels_tensor == 1)).sum().float()
    fp = ((preds_tensor == 1) & (labels_tensor == 0)).sum().float()
    fn = ((preds_tensor == 0) & (labels_tensor == 1)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Simple AUC computation
    probs_tensor = torch.tensor(probs)
    sorted_indices = torch.argsort(probs_tensor, descending=True)
    sorted_labels = labels_tensor[sorted_indices]
    
    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos
    auc = 0.0
    if n_pos > 0 and n_neg > 0:
        cumsum_pos = torch.cumsum(sorted_labels, dim=0)
        auc = (cumsum_pos * (1 - sorted_labels)).sum() / (n_pos * n_neg)
        auc = auc.item()
    
    return precision.item(), recall.item(), f1.item(), auc

def train_pairwise_classifier():
    """Train pairwise association classifier on CLEAN data"""
    
    # Load CLEAN data
    print("Loading CLEAN dataset (sim_clean_001.jsonl)...")
    all_frames = load_data('data/sim_clean_001.jsonl')
    
    # Split
    train_frames = all_frames[:240]
    val_frames = all_frames[240:270]
    
    # Extract pairs
    print("\n===Training Set===")
    train_features, train_labels = extract_pairs_from_frames(train_frames)
    
    print("\n===Validation Set===")
    val_features, val_labels = extract_pairs_from_frames(val_frames)
    
    # Create datasets
    train_dataset = PairwiseDataset(train_features, train_labels)
    val_dataset = PairwiseDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Model
    feature_dim = get_feature_dim()
    model = PairwiseAssociationClassifier(
        feature_dim=feature_dim,
        hidden_dims=[64, 32]
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Use weighted BCE to handle class imbalance
    pos_weight = torch.tensor([float((1-train_labels).sum() / train_labels.sum())], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"Positive weight (for class balance): {pos_weight.item():.2f}")
    
    best_f1 = 0.0
    
    for epoch in range(1, 51):
        # Training
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(features)
        
        train_loss /= len(train_dataset)
        scheduler.step()
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels_val = []
            all_probs = []
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    
                    logits = model(features)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(features)
                    
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels_val.extend(labels.cpu().numpy())
            
            val_loss /= len(val_dataset)
            
            # Metrics
            precision, recall, f1, auc = compute_metrics(all_preds, all_labels_val, all_probs)
            
            print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'checkpoints/pairwise_classifier_best.pt')
                print(f"  ✓ Saved best model (F1: {best_f1:.3f})")
        else:
            print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f}")
    
    print(f"\nTraining complete. Best F1: {best_f1:.3f}")
    print(f"Model saved to checkpoints/pairwise_classifier_best.pt")

if __name__ == "__main__":
    Path('checkpoints').mkdir(exist_ok=True)
    train_pairwise_classifier()

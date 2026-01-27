"""
Training script for DUAL pairwise association classifiers (PSR-PSR and SSR-ANY).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import extract_specialized_pairs, get_psr_psr_dim, get_ssr_any_dim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PairwiseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def load_data(path):
    frames = []
    with open(path) as f:
        for line in f: frames.append(json.loads(line))
    return frames

def compute_metrics(preds, labels, probs):
    preds_tensor = torch.tensor(preds)
    labels_tensor = torch.tensor(labels)
    tp = ((preds_tensor == 1) & (labels_tensor == 1)).sum().float()
    fp = ((preds_tensor == 1) & (labels_tensor == 0)).sum().float()
    fn = ((preds_tensor == 0) & (labels_tensor == 1)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision.item(), recall.item(), f1.item()

def train_classifier(classifier_type, train_frames, val_frames, output_path, writer):
    print(f"\n\n>>> Training {classifier_type} Classifier")
    
    # Extract pairs
    print("Extracting training pairs...")
    train_features = []
    train_labels = []
    for f in tqdm(train_frames):
        feats, labels = extract_specialized_pairs(f.get('measurements', []), classifier_type)
        if len(labels) > 0:
            train_features.append(feats)
            train_labels.append(labels)
    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)
    
    print("Extracting validation pairs...")
    val_features = []
    val_labels = []
    for f in tqdm(val_frames):
        feats, labels = extract_specialized_pairs(f.get('measurements', []), classifier_type)
        if len(labels) > 0:
            val_features.append(feats)
            val_labels.append(labels)
    val_features = np.concatenate(val_features)
    val_labels = np.concatenate(val_labels)
    
    print(f"Dataset Size: {len(train_features)} train, {len(val_features)} val")
    print(f"Positive ratio: {train_labels.mean():.4f}")
    
    train_loader = DataLoader(PairwiseDataset(train_features, train_labels), batch_size=512, shuffle=True)
    val_loader = DataLoader(PairwiseDataset(val_features, val_labels), batch_size=512, shuffle=False)
    
    # Model
    fdim = get_psr_psr_dim() if classifier_type == 'PSR-PSR' else get_ssr_any_dim()
    model = PairwiseAssociationClassifier(feature_dim=fdim, hidden_dims=[64, 32]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([float((1-train_labels).sum() / (train_labels.sum() + 1e-6))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_f1 = 0.0
    for epoch in range(1, 31):
        model.train()
        for feats, labs in train_loader:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labs)
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for feats, labs in val_loader:
                    feats = feats.to(device)
                    probs = torch.sigmoid(model(feats))
                    all_preds.extend((probs > 0.5).float().cpu().numpy())
                    all_labels.extend(labs.numpy())
            
            p, r, f1 = compute_metrics(np.array(all_preds), np.array(all_labels), None)
            print(f"Epoch {epoch:2d} | P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}")
            
            writer.add_scalar(f'{classifier_type}/Precision', p, epoch)
            writer.add_scalar(f'{classifier_type}/Recall', r, epoch)
            writer.add_scalar(f'{classifier_type}/F1', f1, epoch)
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), output_path)
    
    print(f"✓ {classifier_type} Training complete. Best F1: {best_f1:.3f}")

if __name__ == "__main__":
    Path('checkpoints').mkdir(exist_ok=True)
    print("Loading heterogeneous data...")
    all_frames = load_data('data/sim_hetero_001.jsonl')
    train_frames = all_frames[:240]
    val_frames = all_frames[240:300]
    
    writer = SummaryWriter('runs/pairwise_classifiers_v1')
    
    train_classifier('PSR-PSR', train_frames, val_frames, 'checkpoints/pairwise_psr_psr.pt', writer)
    train_classifier('SSR-ANY', train_frames, val_frames, 'checkpoints/pairwise_ssr_any.pt', writer)
    
    writer.close()

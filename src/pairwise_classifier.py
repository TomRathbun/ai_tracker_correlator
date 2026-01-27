"""
Pairwise Association Classifier.

Simple MLP that predicts whether two measurements represent the same object.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseAssociationClassifier(nn.Module):
    """
    MLP classifier for pairwise association.
    
    Input: Pairwise features [batch, feature_dim]
    Output: Association probability [batch] in [0, 1]
    """
    def __init__(self, feature_dim=12, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Args:
            features: [batch, feature_dim]
            
        Returns:
            logits: [batch] - logits for association (apply sigmoid for probability)
        """
        return self.mlp(features).squeeze(-1)
    
    def predict_proba(self, features):
        """
        Predict association probabilities.
        
        Args:
            features: [batch, feature_dim]
            
        Returns:
            probs: [batch] - probabilities in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(features)
            probs = torch.sigmoid(logits)
        return probs

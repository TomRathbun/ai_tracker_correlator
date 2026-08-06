"""
V7 package entry for factory: src.model_v7 → RecurrentGATTrackerV7 alias.
Implementation lives in model_v7_transformer.py.
"""
from src.model_v7_transformer import (  # noqa: F401
    TransformerTrackerV7,
    RecurrentGATTrackerV7,
    frame_to_tensors,
    build_full_input,
    build_gnn_edges,
    model_forward,
    manage_tracks,
    compute_loss,
    focal_bce,
)

import importlib
import torch
import os

def get_model_suite(version: str):
    """
    Dynamically resolves all model-specific components for a given version.
    Expects modules: src.model_v{version} and src.train_streaming_v{version}
    """
    v_str = version.lower()
    model_mod_name = f"src.model_{v_str}"
    train_mod_name = f"src.train_streaming_{v_str}"
    
    try:
        model_module = importlib.import_module(model_mod_name)
    except ImportError:
        raise ImportError(f"Model module {model_mod_name} not found.")

    # Optional training module
    train_fn = None
    try:
        train_module = importlib.import_module(train_mod_name)
        train_fn = getattr(train_module, "train_streaming", None)
    except ImportError:
        pass

    # Extract all components
    suite = {
        "model_class": getattr(model_module, f"RecurrentGATTracker{v_str.upper()}", None),
        "build_edges": getattr(model_module, "build_gnn_edges", None),
        "model_forward": getattr(model_module, "model_forward", None),
        "manage_tracks": getattr(model_module, "manage_tracks", None),
        "train_streaming": train_fn
    }
    
    # Validation
    missing = [k for k, v in suite.items() if v is None and k != "train_streaming"]
    if missing:
        raise AttributeError(f"Model {v_str} is missing required components: {missing}")
        
    return suite

def detect_model_version(checkpoint_path: str):
    """
    Inspects a checkpoint's state_dict to automatically determine the model version.
    """
    if not os.path.exists(checkpoint_path):
        return "v4" # Default fallback
        
    try:
        state_dict = torch.load(checkpoint_path, map_state_dict="cpu", weights_only=True)
        if "clutter_head.0.weight" in state_dict:
            return "v5"
        # Logic for future versions
        if "cross_attn" in str(state_dict.keys()):
            return "v6"
    except:
        pass
        
    return "v4" # Default

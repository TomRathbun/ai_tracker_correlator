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
        "model_class": getattr(model_module, f"RecurrentGATTracker{v_str.upper()}", None)
        or getattr(model_module, f"TransformerTracker{v_str.upper()}", None),
        "build_edges": getattr(model_module, "build_gnn_edges", None),
        "model_forward": getattr(model_module, "model_forward", None),
        "manage_tracks": getattr(model_module, "manage_tracks", None),
        "build_input": getattr(model_module, "build_full_input", None),
        "frame_to_tensors": getattr(model_module, "frame_to_tensors", None),
        "train_streaming": train_fn,
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
        return "v4"  # Default fallback

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        keys = (
            ckpt["model_state_dict"].keys()
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt.keys()
        )
        keys = list(keys)

        # V7 pure Transformer (meas_encoder / cross_layers / state_head)
        if any(k.startswith("meas_encoder") or k.startswith("cross_layers") for k in keys):
            return "v7"
        if any("arch" in str(ckpt) for _ in [0]) and isinstance(ckpt, dict):
            if ckpt.get("arch") == "TransformerTrackerV7":
                return "v7"

        # Phase 2: V6 (Bipartite Cross Attention module name)
        if any("cross_attn" in k for k in keys) and any("gat1" in k for k in keys):
            return "v6"
        if any("cross_attn" in k for k in keys):
            return "v6"

        if "clutter_head.0.weight" in keys or any(k.startswith("clutter_head") for k in keys):
            if "v5" in str(checkpoint_path).lower():
                return "v5"
            return "v4"
    except Exception as e:
        print(f"Version detection failed: {e}")

    return "v4"  # Default fallback

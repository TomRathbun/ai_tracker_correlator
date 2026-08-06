#!/usr/bin/env python3
"""Smoke test: V7 Transformer tracker forward pass + short training run."""
from __future__ import annotations

import os
import sys
import time

import torch

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_forward():
    from src.model_v7_transformer import TransformerTrackerV7, model_forward

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTrackerV7(hidden_dim=64, num_heads=4).to(device)
    model.eval()

    # 2 tracks + 5 measurements
    n_t, n_m = 2, 5
    n = n_t + n_m
    x = torch.randn(n, 8, device=device)
    # Place meas near tracks
    x[:n_t, :3] = torch.tensor([[0.0, 0.0, 1000.0], [5000.0, 3000.0, 2000.0]], device=device)
    x[n_t:, :3] = x[:1, :3].repeat(n_m, 1) + torch.randn(n_m, 3, device=device) * 200
    node_type = torch.zeros(n, dtype=torch.long, device=device)
    node_type[:n_t] = 1
    sensor_id = torch.randint(0, 5, (n,), device=device)
    hidden = torch.zeros(n_t, 64, device=device)

    with torch.no_grad():
        out, new_h, attn, exist_p, exist_l, clut_p, clut_l, _ = model_forward(
            model, x, node_type, sensor_id, None, None, hidden
        )

    assert out.shape == (n, 7), out.shape
    assert new_h.shape == (n, 64), new_h.shape
    assert exist_p.shape == (n,), exist_p.shape
    print("[OK] forward pass")
    print(f"     out={tuple(out.shape)} hidden={tuple(new_h.shape)} attn={None if attn is None else tuple(attn.shape)}")
    print(f"     exist_prob mean={exist_p.mean().item():.3f} clutter mean={clut_p.mean().item():.3f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"     parameters={n_params:,}")
    return True


def test_factory():
    from src.factory import get_model_suite

    suite = get_model_suite("v7")
    assert suite["model_class"] is not None
    assert suite["frame_to_tensors"] is not None
    assert suite["model_forward"] is not None
    m = suite["model_class"](hidden_dim=32, num_heads=4)
    print("[OK] factory suite v7")
    print(f"     class={suite['model_class'].__name__} train_fn={suite['train_streaming'] is not None}")
    return True


def test_train_smoke():
    from src.train_streaming_v7 import train_streaming

    data = os.path.join(ROOT, "data", "sim_hetero_001.jsonl")
    if not os.path.exists(data):
        data = os.path.join(ROOT, "data", "canonical", "sim_batch_hetero.jsonl")
    ckpt = os.path.join(ROOT, "checkpoints", "model_v7_transformer_smoke.pt")

    t0 = time.time()
    result = train_streaming(
        num_epochs=1,
        data_file=data,
        window_size=2.0,
        max_windows=40,
        checkpoint_path=ckpt,
        lr=1e-4,
        hidden_dim=64,
        num_heads=4,
    )
    dt = time.time() - t0
    print("[OK] train smoke")
    print(f"     elapsed={dt:.1f}s history={result['history']} ckpt={result['checkpoint']}")
    assert os.path.exists(ckpt), "checkpoint missing"
    # detect version
    from src.factory import detect_model_version

    ver = detect_model_version(ckpt)
    print(f"     detect_model_version={ver}")
    assert ver == "v7", ver
    return True


def main():
    print("=== V7 Transformer Tracker smoke tests ===")
    test_forward()
    test_factory()
    test_train_smoke()
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()

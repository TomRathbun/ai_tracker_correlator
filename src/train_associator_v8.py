"""
Train V8 association transformer on track_id labels.

Supervised matching only — no residual state, no existence, no GRU.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.model_v8_associator import (
    ASSOC_GATE_M,
    CLUSTER_GATE_M,
    AssociationTransformerV8,
    focal_bce,
    project_track_to_time,
)
from src.stream_utils import load_stream_and_truth


def make_split(all_track_ids, split_ratio: float = 0.8, seed: int = 42) -> Tuple[Set[int], Set[int]]:
    ids = list(all_track_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * split_ratio))
    return set(ids[:n_train]), set(ids[n_train:])


def iter_windows(measurements: List[Dict], window_size: float) -> List[List[Dict]]:
    if not measurements:
        return []
    measurements = sorted(measurements, key=lambda m: m.get("t", 0.0))
    t0 = measurements[0]["t"]
    t1 = measurements[-1]["t"]
    windows: List[List[Dict]] = []
    idx = 0
    t = t0
    n = len(measurements)
    while t < t1:
        nxt = t + window_size
        chunk = []
        while idx < n and measurements[idx]["t"] < nxt:
            chunk.append(measurements[idx])
            idx += 1
        if chunk:
            windows.append(chunk)
        t = nxt
    return windows


def _tid(m: Dict) -> int:
    try:
        return int(m.get("track_id", -1))
    except (TypeError, ValueError):
        return -1


def cluster_pairs(window: List[Dict], allowed: Set[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gated meas-meas pairs + labels (same track_id)."""
    pairs: List[Tuple[int, int]] = []
    labels: List[float] = []
    n = len(window)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = window[i], window[j]
            dx = float(a.get("x", 0) or 0) - float(b.get("x", 0) or 0)
            dy = float(a.get("y", 0) or 0) - float(b.get("y", 0) or 0)
            if dx * dx + dy * dy > CLUSTER_GATE_M ** 2:
                continue
            pairs.append((i, j))
            ia, ib = _tid(a), _tid(b)
            pos = ia != -1 and ia == ib and ia in allowed
            labels.append(1.0 if pos else 0.0)
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
    return torch.tensor(pairs, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)


def assign_sample(
    last: Dict[int, Dict],
    window: List[Dict],
    allowed: Set[int],
) -> Tuple[List[Dict], List[Dict], np.ndarray, np.ndarray]:
    """Teacher-forced tracks = last plot per id; metas = this window."""
    tracks = [last[tid] for tid in last if tid in allowed]
    metas = window
    if not tracks or not metas:
        return tracks, metas, np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    t_n, m_n = len(tracks), len(metas)
    pair_y = np.zeros((t_n, m_n), dtype=np.float32)
    dust_y = np.ones((t_n,), dtype=np.float32)
    for i, tr in enumerate(tracks):
        tid = _tid(tr)
        for j, meta in enumerate(metas):
            mt = meta.get("t")
            proj = project_track_to_time(tr, mt)
            dx = float(proj.get("x", 0) or 0) - float(meta.get("x", 0) or 0)
            dy = float(proj.get("y", 0) or 0) - float(meta.get("y", 0) or 0)
            if dx * dx + dy * dy > ASSOC_GATE_M ** 2:
                continue
            if tid != -1 and tid == _tid(meta):
                pair_y[i, j] = 1.0
                dust_y[i] = 0.0
    return tracks, metas, pair_y, dust_y


def train_associator(
    data_file: str = "data/sim_hetero_001.jsonl",
    num_epochs: int = 20,
    window_size: float = 2.0,
    split_ratio: float = 0.8,
    lr: float = 1e-3,
    checkpoint_path: str = "checkpoints/model_v8_assoc.pt",
    hidden_dim: int = 64,
    num_heads: int = 4,
    use_self_attn: bool = True,
    max_windows: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    measurements, _truth, all_ids = load_stream_and_truth(data_file)
    train_ids, test_ids = make_split(all_ids, split_ratio=split_ratio)
    print(f"V8 split: {len(train_ids)} train ids, {len(test_ids)} holdout ids")

    windows = iter_windows(measurements, window_size)
    if max_windows is not None:
        windows = windows[: max(1, max_windows)]
    print(f"Windows: {len(windows)} @ {window_size}s")

    model = AssociationTransformerV8(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        use_self_attn=use_self_attn,
    ).to(device)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            print(f"Resumed V8 from {checkpoint_path}")
        except Exception as exc:
            print(f"Could not load checkpoint ({exc}); training from scratch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        last: Dict[int, Dict] = {}
        losses = []
        pbar = tqdm(windows, desc=f"Epoch {epoch + 1} (V8 associator)")
        for window in pbar:
            pair_idx, pair_y = cluster_pairs(window, train_ids)
            tracks, metas, assign_y, dust_y = assign_sample(last, window, train_ids)

            optimizer.zero_grad()
            loss = torch.zeros((), device=device)
            n_terms = 0

            if pair_idx.numel() > 0:
                logits = model.score_pairs(window, window, pair_idx.to(device))
                y = pair_y.to(device)
                loss = loss + focal_bce(logits, y)
                n_terms += 1

            if tracks and metas:
                S, dust = model.score_assignment(tracks, metas)
                ay = torch.from_numpy(assign_y).to(device)
                dy = torch.from_numpy(dust_y).to(device)
                gate = torch.zeros_like(ay, dtype=torch.bool)
                for i, tr in enumerate(tracks):
                    for j, meta in enumerate(metas):
                        proj = project_track_to_time(tr, meta.get("t"))
                        dx = float(proj.get("x", 0) or 0) - float(meta.get("x", 0) or 0)
                        dyv = float(proj.get("y", 0) or 0) - float(meta.get("y", 0) or 0)
                        if dx * dx + dyv * dyv <= ASSOC_GATE_M ** 2:
                            gate[i, j] = True
                if gate.any():
                    loss = loss + focal_bce(S[gate], ay[gate])
                    n_terms += 1
                loss = loss + focal_bce(dust, dy)
                n_terms += 1
                # peaked rows (optional, light)
                if S.numel() > 0:
                    logits_row = torch.cat([S, dust.unsqueeze(-1)], dim=-1)
                    p = torch.softmax(logits_row, dim=-1).clamp_min(1e-8)
                    ent = -(p * p.log()).sum(dim=-1).mean()
                    loss = loss + 0.1 * ent

            if n_terms == 0 or not torch.isfinite(loss):
                _update_last(last, window, train_ids)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            pbar.set_postfix(loss=np.mean(losses[-20:]))
            _update_last(last, window, train_ids)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"Epoch {epoch + 1}: mean_loss={mean_loss:.4f} steps={len(losses)}")
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss, "steps": len(losses)})

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "history": history,
            "arch": "AssociationTransformerV8",
            "config": {
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "use_self_attn": use_self_attn,
            },
        }
        torch.save(payload, checkpoint_path)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = checkpoint_path.replace(".pt", "_best.pt")
            torch.save(payload, best_path)
            print(f"Saved {checkpoint_path} (best {best_path})")
        else:
            print(f"Saved {checkpoint_path}")

    return {"history": history, "checkpoint": checkpoint_path, "device": str(device)}


def _update_last(last: Dict[int, Dict], window: List[Dict], allowed: Set[int]) -> None:
    for m in window:
        tid = _tid(m)
        if tid != -1 and tid in allowed:
            last[tid] = m


def main():
    p = argparse.ArgumentParser(description="Train V8 association transformer")
    p.add_argument("--data", default="data/sim_hetero_001.jsonl")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--window", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint", "--out", dest="checkpoint", default="checkpoints/model_v8_assoc.pt")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--no-self-attn", action="store_true")
    args = p.parse_args()
    train_associator(
        data_file=args.data,
        num_epochs=args.epochs,
        window_size=args.window,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        use_self_attn=not args.no_self_attn,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()

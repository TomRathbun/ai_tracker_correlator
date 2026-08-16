"""
Train V8 association transformer on track_id labels.

Supervised matching only — no residual state, no existence, no GRU.
Negatives are capped (~8× positives) and loss is class-balanced.
Early-stop on holdout pair F1, not train loss or precision.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Set, Tuple

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


def subsample_binary(
    y: torch.Tensor, max_neg_ratio: float, rng: np.random.RandomState
) -> torch.Tensor:
    """Boolean keep-mask: all positives + up to max_neg_ratio * n_pos negatives."""
    yf = y.reshape(-1)
    pos = (yf > 0.5).detach().cpu().numpy()
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    cap = int(max(n_pos, 1) * max_neg_ratio)
    keep = np.ones(yf.numel(), dtype=bool)
    if n_neg > cap:
        drop = rng.choice(np.flatnonzero(~pos), size=n_neg - cap, replace=False)
        keep[drop] = False
    return torch.from_numpy(keep).to(y.device)


def pair_prf(pred: torch.Tensor, y: torch.Tensor) -> Tuple[int, int, int]:
    pos = y > 0.5
    pr = pred > 0.5
    tp = int((pr & pos).sum().item())
    fp = int((pr & ~pos).sum().item())
    fn = int((~pr & pos).sum().item())
    return tp, fp, fn


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


@torch.no_grad()
def eval_pair_metrics(
    model: AssociationTransformerV8,
    windows: List[List[Dict]],
    allowed: Set[int],
    device: torch.device,
    thr: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    last: Dict[int, Dict] = {}
    for window in windows:
        # Holdout plots + clutter only, so train-id pairs are not counted as false positives.
        sliced = [m for m in window if _tid(m) == -1 or _tid(m) in allowed]
        pair_idx, pair_y = cluster_pairs(sliced, allowed)
        if pair_idx.numel() > 0:
            logits = model.score_pairs(window, window, pair_idx.to(device))
            pred = torch.sigmoid(logits) > thr
            t, f, n = pair_prf(pred, pair_y.to(device))
            tp += t
            fp += f
            fn += n
        _update_last(last, window, allowed)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


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
    init_path: Optional[str] = None,
    neg_ratio: float = 8.0,
    patience: int = 6,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed)
    print(f"Device: {device}  neg_ratio={neg_ratio}  patience={patience}")

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
    warm = init_path or (checkpoint_path if os.path.exists(checkpoint_path) else None)
    if warm and os.path.exists(warm):
        try:
            ckpt = torch.load(warm, map_location=device, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            print(f"Resumed V8 from {warm}")
        except Exception as exc:
            print(f"Could not load checkpoint ({exc}); training from scratch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    best_val_f1 = -1.0
    stale = 0

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
                keep = subsample_binary(pair_y, neg_ratio, rng)
                pair_idx_b = pair_idx[keep]
                pair_y_b = pair_y[keep]
                if pair_idx_b.numel() > 0:
                    logits = model.score_pairs(window, window, pair_idx_b.to(device))
                    loss = loss + focal_bce(logits, pair_y_b.to(device), balance=True)
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
                    s_g, y_g = S[gate], ay[gate]
                    keep = subsample_binary(y_g, neg_ratio, rng)
                    loss = loss + focal_bce(s_g[keep], y_g[keep], balance=True)
                    n_terms += 1
                # Dustbin: keep all rows (rare positives / coasts)
                loss = loss + 2.0 * focal_bce(dust, dy, balance=True)
                n_terms += 1
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
        val = eval_pair_metrics(model, windows, test_ids, device)
        print(
            f"Epoch {epoch + 1}: mean_loss={mean_loss:.4f} steps={len(losses)} "
            f"val_prec={val['precision']:.3f} val_rec={val['recall']:.3f} val_f1={val['f1']:.3f} "
            f"tp={int(val['tp'])} fp={int(val['fp'])} fn={int(val['fn'])}"
        )
        row = {"epoch": epoch + 1, "mean_loss": mean_loss, "steps": len(losses), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)

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
            "metrics": {"val_pair_precision": val["precision"], "val_pair_f1": val["f1"]},
            "schema_version": 1,
        }
        torch.save(payload, checkpoint_path)
        if val["f1"] > best_val_f1 + 1e-4:
            best_val_f1 = val["f1"]
            stale = 0
            best_path = checkpoint_path.replace(".pt", "_best.pt")
            torch.save(payload, best_path)
            print(f"Saved {checkpoint_path} (best val_f1={best_val_f1:.3f} → {best_path})")
        else:
            stale += 1
            print(f"Saved {checkpoint_path} (stale={stale}/{patience})")
            if stale >= patience:
                print(f"Early stop: val pair F1 did not improve for {patience} epochs")
                break

    return {"history": history, "checkpoint": checkpoint_path, "device": str(device), "best_val_f1": best_val_f1}


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
    p.add_argument("--init", default=None, help="Warm-start weights (does not have to be --checkpoint)")
    p.add_argument("--neg-ratio", type=float, default=8.0, help="Max negatives per positive after gating")
    p.add_argument("--patience", type=int, default=6, help="Early-stop epochs without val pair-F1 gain")
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
        init_path=args.init,
        neg_ratio=args.neg_ratio,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()

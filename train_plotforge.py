#!/usr/bin/env python3
"""Fine-tune hybrid clutter + pairwise classifiers on a PlotForge stream."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.clutter_classifier import ClutterClassifier, extract_clutter_features
from src.data_schema import get_meas_type, normalize_measurement_dict
from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import (
    compute_psr_psr_features,
    compute_ssr_any_features,
    get_psr_psr_dim,
    get_ssr_any_dim,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GATE_M = 8000.0


class XYDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def load_stream(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            rows.append(normalize_measurement_dict(obj))
    rows.sort(key=lambda m: float(m.get("t", 0.0)))
    return rows


def windows(rows: list[dict], dt: float = 1.0) -> list[list[dict]]:
    if not rows:
        return []
    out: list[list[dict]] = []
    t0 = float(rows[0]["t"])
    t1 = float(rows[-1]["t"])
    i = 0
    t = t0
    n = len(rows)
    while t < t1:
        nxt = t + dt
        chunk = []
        while i < n and float(rows[i]["t"]) < nxt:
            chunk.append(rows[i])
            i += 1
        if chunk:
            out.append(chunk)
        t = nxt
    return out


def clutter_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for m in rows:
        feats.append(extract_clutter_features(m).numpy())
        if "is_clutter" in m:
            lab = 1.0 if m["is_clutter"] else 0.0
        else:
            lab = 1.0 if int(m.get("track_id", -1)) == -1 else 0.0
        labels.append(lab)
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def gated_pairs(frames: list[list[dict]], kind: str, neg_ratio: float, rng: np.random.RandomState):
    pos, neg = [], []
    gate2 = GATE_M * GATE_M
    for chunk in frames:
        n = len(chunk)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = chunk[i], chunk[j]
                dx = float(a["x"]) - float(b["x"])
                dy = float(a["y"]) - float(b["y"])
                if dx * dx + dy * dy > gate2:
                    continue
                t1, t2 = get_meas_type(a), get_meas_type(b)
                if kind == "PSR-PSR":
                    if not (t1 == "PSR" and t2 == "PSR"):
                        continue
                    feat = compute_psr_psr_features(a, b)
                else:
                    if not (t1 == "SSR" or t2 == "SSR"):
                        continue
                    feat = compute_ssr_any_features(a, b)
                tid1 = int(a.get("track_id", -1))
                tid2 = int(b.get("track_id", -1))
                y = 1.0 if (tid1 == tid2 and tid1 != -1) else 0.0
                (pos if y > 0.5 else neg).append(feat)
    if not pos:
        dim = get_psr_psr_dim() if kind == "PSR-PSR" else get_ssr_any_dim()
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    cap = int(max(len(pos), 1) * neg_ratio)
    if len(neg) > cap:
        pick = rng.choice(len(neg), size=cap, replace=False)
        neg = [neg[k] for k in pick]
    x = np.asarray(pos + neg, dtype=np.float32)
    y = np.concatenate(
        [np.ones(len(pos), dtype=np.float32), np.zeros(len(neg), dtype=np.float32)]
    )
    return x, y


def prf(probs: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> dict:
    pred = (probs > thr).astype(np.float32)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def train_mlp(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    out_path: Path,
    name: str,
) -> dict:
    n = len(y)
    idx = np.random.permutation(n)
    split = max(1, int(0.85 * n))
    tr, va = idx[:split], idx[split:]
    train_loader = DataLoader(XYDataset(x[tr], y[tr]), batch_size=256, shuffle=True)
    val_x = torch.tensor(x[va], dtype=torch.float32, device=DEVICE)
    val_y = y[va]
    pos = float(y[tr].sum())
    neg = float(len(tr) - pos)
    pos_weight = torch.tensor([neg / (pos + 1e-6)], device=DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best = -1.0
    best_metrics = {}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for feats, labs in train_loader:
            feats, labs = feats.to(DEVICE), labs.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(feats), labs)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(val_x)).cpu().numpy()
        m = prf(probs, val_y)
        m["loss"] = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, **{k: round(float(v), 4) for k, v in m.items() if k != "epoch"}})
        if m["f1"] > best:
            best = m["f1"]
            best_metrics = m
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"  {name} epoch {epoch:02d} loss={m['loss']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
            )
    return {
        "name": name,
        "nTrain": int(len(tr)),
        "nVal": int(len(va)),
        "posRate": round(float(y.mean()), 4),
        "bestF1": round(float(best), 4),
        "best": {k: round(float(v), 4) for k, v in best_metrics.items()},
        "epochs": epochs,
        "path": str(out_path),
    }


def maybe_load(model: nn.Module, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"  warm start {path}")
        return True
    except Exception as exc:
        print(f"  could not warm-start {path}: {exc}")
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", default="checkpoints/plotforge")
    p.add_argument("--epochs-clutter", type=int, default=20)
    p.add_argument("--epochs-pair", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--neg-ratio", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--init-clutter", default="checkpoints/clutter_classifier.pt")
    p.add_argument("--init-psr", default="checkpoints/pairwise_psr_psr.pt")
    p.add_argument("--init-ssr", default="checkpoints/pairwise_ssr_any.pt")
    args = p.parse_args()
    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.perf_counter()
    print(f"Device {DEVICE}")
    print(f"Loading {args.data}")
    rows = load_stream(args.data)
    frames = windows(rows, 1.0)
    print(f"  {len(rows)} hits, {len(frames)} 1s windows")

    out_dir = Path(args.out_dir)
    report: dict = {"data": args.data, "nHits": len(rows), "nWindows": len(frames), "models": []}

    print("\n=== Clutter classifier ===")
    cx, cy = clutter_xy(rows)
    print(f"  clutter rate {cy.mean():.3%}")
    clutter = ClutterClassifier(feature_dim=8).to(DEVICE)
    maybe_load(clutter, Path(args.init_clutter))
    report["models"].append(
        train_mlp(clutter, cx, cy, args.epochs_clutter, args.lr, out_dir / "clutter_classifier.pt", "clutter")
    )

    print("\n=== PSR-PSR pairwise ===")
    px, py = gated_pairs(frames, "PSR-PSR", args.neg_ratio, rng)
    print(f"  pairs {len(py)} posRate {py.mean() if len(py) else 0:.3%}")
    psr = PairwiseAssociationClassifier(feature_dim=get_psr_psr_dim(), hidden_dims=[64, 32]).to(DEVICE)
    maybe_load(psr, Path(args.init_psr))
    report["models"].append(
        train_mlp(psr, px, py, args.epochs_pair, args.lr, out_dir / "pairwise_psr_psr.pt", "psr-psr")
    )

    print("\n=== SSR-ANY pairwise ===")
    sx, sy = gated_pairs(frames, "SSR-ANY", args.neg_ratio, rng)
    print(f"  pairs {len(sy)} posRate {sy.mean() if len(sy) else 0:.3%}")
    ssr = PairwiseAssociationClassifier(feature_dim=get_ssr_any_dim(), hidden_dims=[64, 32]).to(DEVICE)
    maybe_load(ssr, Path(args.init_ssr))
    report["models"].append(
        train_mlp(ssr, sx, sy, args.epochs_pair, args.lr, out_dir / "pairwise_ssr_any.pt", "ssr-any")
    )

    report["elapsedS"] = round(time.perf_counter() - t0, 2)
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'train_report.json'} in {report['elapsedS']}s")


if __name__ == "__main__":
    main()

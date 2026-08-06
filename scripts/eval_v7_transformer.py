#!/usr/bin/env python3
"""
Evaluate V7 Transformer tracker with MOTA / MOTP on a streaming JSONL.

Uses the same track-ID split as training (seed=42, 80/20) so metrics reflect
holdout aircraft when --holdout is set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.factory import detect_model_version, get_model_suite
from src.metrics import TrackingMetrics
from src.model_v7_transformer import (
    TransformerTrackerV7,
    build_full_input,
    frame_to_tensors,
    manage_tracks,
    model_forward,
)
from src.stream_utils import get_truth_at_time, load_stream_and_truth


def make_split(all_track_ids, split_ratio=0.8, seed=42):
    ids = list(all_track_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * split_ratio))
    return set(ids[:n_train]), set(ids[n_train:])


def evaluate(
    data_file: str,
    model_path: str,
    window_size: float = 2.0,
    holdout: bool = True,
    split_ratio: float = 0.8,
    init_thresh: float = 0.55,
    coast_thresh: float = 0.25,
    suppress_thresh: float = 0.25,
    del_exist: float = 0.20,
    track_cap: int = 150,
    match_threshold: float = 15000.0,
    results_path: str = "artifacts/v7_eval_results.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {model_path}")
    print(f"Data:   {data_file}")

    measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)
    train_ids, test_ids = make_split(all_track_ids, split_ratio=split_ratio)
    eval_ids = test_ids if holdout else set(all_track_ids)
    print(f"Tracks total={len(all_track_ids)} train={len(train_ids)} test={len(test_ids)}")
    print(f"Evaluating against {'holdout test' if holdout else 'all'} IDs ({len(eval_ids)})")

    # Filter measurements: keep clutter + eval-relevant labels for context;
    # for holdout mode still feed all sensors (realistic), score only eval GT.
    measurements = [m for m in measurements_all if isinstance(m, dict)]
    measurements.sort(key=lambda x: x["t"])

    model = TransformerTrackerV7(hidden_dim=64, num_heads=4).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint (detect={detect_model_version(model_path)})")

    # Also sanity-check factory wiring
    suite = get_model_suite("v7")
    assert suite["model_class"] is not None

    metrics = TrackingMetrics(match_threshold=match_threshold)
    active_tracks = []
    t_start = measurements[0]["t"]
    t_end = measurements[-1]["t"]
    current_t = t_start
    meas_idx = 0
    windows = 0
    peak_tracks = 0

    cfg_note = {
        "init_thresh": init_thresh,
        "coast_thresh": coast_thresh,
        "suppress_thresh": suppress_thresh,
        "del_exist": del_exist,
        "track_cap": track_cap,
        "window_size": window_size,
        "holdout": holdout,
    }
    print(f"Inference config: {cfg_note}")

    with torch.no_grad():
        pbar = tqdm(total=max(1, int(t_end - t_start)), desc="V7 eval")
        while current_t < t_end:
            next_t = current_t + window_size
            window_meas = []
            while meas_idx < len(measurements) and measurements[meas_idx]["t"] < next_t:
                window_meas.append(measurements[meas_idx])
                meas_idx += 1

            meas_tensor, meas_sensor_ids = frame_to_tensors(window_meas, device, window_t=next_t)
            full_x, full_sensor_id, track_hiddens, num_tracks = build_full_input(
                active_tracks, meas_tensor, meas_sensor_ids, num_sensors=6, device=device
            )
            num_meas = full_x.shape[0] - num_tracks

            if full_x.shape[0] == 0:
                current_t = next_t
                pbar.update(int(window_size))
                continue

            node_type = torch.zeros(full_x.shape[0], dtype=torch.long, device=device)
            if num_tracks > 0:
                node_type[:num_tracks] = 1

            out, new_hidden_full, attn, exist_p, exist_l, clut_p, clut_l, _ = model_forward(
                model, full_x, node_type, full_sensor_id, None, None, track_hiddens
            )

            active_tracks = manage_tracks(
                active_tracks,
                out,
                new_hidden_full,
                exist_p,
                exist_l,
                clut_p,
                attn,
                None,
                num_tracks,
                num_meas,
                init_thresh,
                coast_thresh,
                suppress_thresh,
                del_exist,
                del_age=8,
                track_cap=track_cap,
                dt=window_size,
                clutter_thresh=0.70,
            )
            peak_tracks = max(peak_tracks, len(active_tracks))

            gt_list = get_truth_at_time(truth_trajectories, next_t, allowed_ids=eval_ids)
            if active_tracks or gt_list:
                if active_tracks:
                    pred_states = torch.stack([tr["state_tensor"] for tr in active_tracks])
                    pred_np = pred_states[:, :6].detach().cpu().numpy()
                    pred_ids = [int(tr.get("id", -1)) for tr in active_tracks]
                else:
                    pred_np = np.zeros((0, 6), dtype=np.float32)
                    pred_ids = []
                if gt_list:
                    gt_np = np.array(
                        [
                            [
                                g.get("x", 0),
                                g.get("y", 0),
                                g.get("z", 0),
                                g.get("vx", 0),
                                g.get("vy", 0),
                                g.get("vz", 0),
                            ]
                            for g in gt_list
                            if isinstance(g, dict)
                        ],
                        dtype=np.float32,
                    )
                else:
                    gt_np = np.zeros((0, 6), dtype=np.float32)
                metrics.update(pred_np, gt_np, pred_ids=pred_ids)

            windows += 1
            current_t = next_t
            pbar.update(int(window_size))
            pbar.set_postfix(tracks=len(active_tracks), peak=peak_tracks)
        pbar.close()

    results = metrics.compute()
    results_out = {
        "model": model_path,
        "data": data_file,
        "device": str(device),
        "windows": windows,
        "peak_tracks": peak_tracks,
        "final_tracks": len(active_tracks),
        "eval_ids": len(eval_ids),
        "config": cfg_note,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in results.items()},
    }

    print("\n=== V7 Evaluation Results ===")
    for k, v in results_out["metrics"].items():
        if isinstance(v, float):
            print(f"  {k:>16}: {v:.4f}")
        else:
            print(f"  {k:>16}: {v}")
    print(f"  {'peak_tracks':>16}: {peak_tracks}")
    print(f"  {'windows':>16}: {windows}")

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nWrote {results_path}")
    return results_out


def main():
    p = argparse.ArgumentParser(description="Evaluate V7 Transformer tracker")
    p.add_argument("--data", default="data/stream_radar_001.jsonl")
    p.add_argument("--model", default="checkpoints/model_v7_transformer.pt")
    p.add_argument("--window", type=float, default=2.0)
    p.add_argument("--holdout", action="store_true", default=True)
    p.add_argument("--all-ids", action="store_true", help="Score all track IDs (not holdout only)")
    p.add_argument("--init-thresh", type=float, default=0.55)
    p.add_argument("--coast-thresh", type=float, default=0.25)
    p.add_argument("--track-cap", type=int, default=150)
    p.add_argument("--results", default="artifacts/v7_eval_results.json")
    args = p.parse_args()
    evaluate(
        data_file=args.data,
        model_path=args.model,
        window_size=args.window,
        holdout=not args.all_ids,
        init_thresh=args.init_thresh,
        coast_thresh=args.coast_thresh,
        track_cap=args.track_cap,
        results_path=args.results,
    )


if __name__ == "__main__":
    main()

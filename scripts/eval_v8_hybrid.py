#!/usr/bin/env python3
"""Evaluate Hybrid + V8 associator on a streaming JSONL (same contract as run_cli hybrid)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_schemas import PipelineConfig
from src.pipeline import Pipeline
from src.metrics import TrackingMetrics
from src.stream_utils import load_stream_and_truth, get_truth_at_time


def evaluate(
    data_file: str,
    assoc: str = "transformer",
    v8_path: str = "checkpoints/model_v8_assoc.pt",
    use_dustbin: bool = False,
    window_size: float = 1.0,
    match_threshold: float = 7000.0,
    min_hits: int = 3,
    max_age: int = 10,
):
    cfg = PipelineConfig()
    cfg.state_updater.type = "hybrid"
    cfg.pairwise.backend = assoc
    cfg.pairwise.v8_model_path = Path(v8_path)
    cfg.pairwise.use_dustbin = use_dustbin
    cfg.track_manager.min_hits = min_hits
    cfg.track_manager.max_age = max_age
    cfg.clutter_filter.enabled = True

    print(f"Assoc backend: {assoc}  dustbin={use_dustbin}  v8={v8_path}")
    pipeline = Pipeline(cfg)

    measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)
    measurements_all.sort(key=lambda x: x["t"])
    t_start, t_end = measurements_all[0]["t"], measurements_all[-1]["t"]
    metrics = TrackingMetrics(match_threshold=match_threshold)

    current_t = t_start
    meas_idx = 0
    windows = 0
    from tqdm import tqdm

    pbar = tqdm(total=max(1, int(t_end - t_start)), desc="V8 hybrid eval")
    while current_t < t_end:
        window_meas = []
        while meas_idx < len(measurements_all) and measurements_all[meas_idx]["t"] < current_t + window_size:
            window_meas.append(measurements_all[meas_idx])
            meas_idx += 1
        predicted = pipeline.process_frame(window_meas, t=current_t + window_size)
        gt = get_truth_at_time(truth_trajectories, current_t + window_size, set(all_track_ids))
        metrics.update(predicted, gt)
        current_t += window_size
        windows += 1
        pbar.update(1)
        pbar.set_postfix(tracks=len(predicted))
    pbar.close()

    results = metrics.compute()
    print("\n=== Hybrid + V8 Evaluation ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:>16}: {v:.4f}")
        else:
            print(f"  {k:>16}: {v}")
    print(f"  {'windows':>16}: {windows}")
    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate Hybrid tracker with V8 associator")
    p.add_argument("--data", default="data/stream_radar_001.jsonl")
    p.add_argument("--assoc", choices=["mlp", "transformer", "ensemble"], default="transformer")
    p.add_argument("--v8-model", default="checkpoints/model_v8_assoc.pt")
    p.add_argument("--dustbin", action="store_true")
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--max-age", type=int, default=2)
    p.add_argument("--min-hits", type=int, default=3)
    args = p.parse_args()
    if args.assoc != "mlp" and not os.path.exists(args.v8_model):
        print(f"Warning: {args.v8_model} missing — Hybrid will fall back to MLP.")
    evaluate(
        args.data,
        assoc=args.assoc,
        v8_path=args.v8_model,
        use_dustbin=args.dustbin,
        window_size=args.window,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )


if __name__ == "__main__":
    main()

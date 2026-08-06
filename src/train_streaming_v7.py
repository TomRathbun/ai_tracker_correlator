"""
V7 Transformer Tracker training on streaming multi-radar JSONL.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.model_v7_transformer import (
    TransformerTrackerV7,
    build_full_input,
    compute_loss,
    frame_to_tensors,
    manage_tracks,
    model_forward,
)
from src.stream_utils import get_truth_at_time, load_stream_and_truth


def train_streaming(
    num_epochs: int = 2,
    data_file: str = "data/sim_hetero_001.jsonl",
    window_size: float = 2.0,
    split_ratio: float = 0.8,
    start_epoch: int = 0,
    max_windows: int | None = None,
    checkpoint_path: str = "checkpoints/model_v7_transformer.pt",
    lr: float = 1e-4,
    hidden_dim: int = 64,
    num_heads: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)
    all_track_ids = list(all_track_ids)
    np.random.seed(42)
    np.random.shuffle(all_track_ids)
    num_train = max(1, int(len(all_track_ids) * split_ratio))
    train_ids = set(all_track_ids[:num_train])
    test_ids = set(all_track_ids[num_train:])
    print(f"V7 split: {len(train_ids)} train tracks, {len(test_ids)} test tracks")

    measurements = [
        m
        for m in measurements_all
        if isinstance(m, dict)
        and (m.get("track_id", -1) in train_ids or m.get("track_id", -1) == -1)
    ]
    measurements.sort(key=lambda x: x["t"])
    if not measurements:
        raise RuntimeError(f"No measurements after split from {data_file}")

    model = TransformerTrackerV7(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        use_radius_mask=True,
        max_assoc_m=50_000.0,
    ).to(device)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            print(f"Resumed V7 from {checkpoint_path}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}); training from scratch")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    cfg = {}
    try:
        with open("src/training_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        pass

    init_thresh = cfg.get("init_thresh", 0.45)
    coast_thresh = cfg.get("coast_thresh", 0.20)
    suppress_thresh = cfg.get("suppress_thresh", 0.15)
    fp_mult = cfg.get("fp_mult", 8.0)
    match_gate = cfg.get("match_gate", 5000.0)
    miss_penalty = cfg.get("miss_penalty", 0.05)
    del_exist = cfg.get("del_exist", 0.05)
    track_cap = cfg.get("track_cap", 200)

    history: List[dict] = []

    for epoch_idx in range(num_epochs):
        epoch = start_epoch + epoch_idx
        model.train()
        active_tracks: List[dict] = []
        epoch_losses = []
        t_start = measurements[0]["t"]
        t_end = measurements[-1]["t"]
        current_t = t_start
        meas_idx = 0
        windows_done = 0

        pbar = tqdm(total=max(1, int(t_end - t_start)), desc=f"Epoch {epoch + 1} (V7 Transformer)")

        while current_t < t_end:
            next_t = current_t + window_size
            window_meas_list = []
            while meas_idx < len(measurements) and measurements[meas_idx]["t"] < next_t:
                window_meas_list.append(measurements[meas_idx])
                meas_idx += 1

            meas_node_t, sensor_ids = frame_to_tensors(window_meas_list, device, window_t=next_t)
            full_x, full_sensor_id, track_hiddens, num_tracks = build_full_input(
                active_tracks, meas_node_t, sensor_ids, num_sensors=6, device=device
            )
            num_meas = full_x.shape[0] - num_tracks
            node_type = torch.zeros(full_x.shape[0], dtype=torch.long, device=device)
            if num_tracks > 0:
                node_type[:num_tracks] = 1

            if full_x.shape[0] == 0:
                current_t = next_t
                pbar.update(int(window_size))
                continue

            optimizer.zero_grad()
            res = model_forward(
                model,
                full_x,
                node_type,
                full_sensor_id,
                None,
                None,
                track_hiddens,
            )
            out, new_hidden_full, attn_weights, existence_probs, existence_logits, clutter_probs, clutter_logits, _ = res

            gt_states = get_truth_at_time(truth_trajectories, next_t, allowed_ids=train_ids)
            if gt_states:
                gt_tensor = torch.tensor(
                    [
                        [
                            g.get("x", 0),
                            g.get("y", 0),
                            g.get("z", 0),
                            g.get("vx", 0),
                            g.get("vy", 0),
                            g.get("vz", 0),
                        ]
                        for g in gt_states
                        if isinstance(g, dict)
                    ],
                    device=device,
                    dtype=torch.float32,
                )
            else:
                gt_tensor = torch.empty((0, 6), device=device)

            loss, metrics = compute_loss(
                pred_states=out[:, :6],
                pred_logits=existence_logits,
                gt_states_dev=gt_tensor,
                num_gt=gt_tensor.shape[0],
                match_gate=match_gate,
                miss_penalty=miss_penalty,
                fp_mult=fp_mult,
                out=out,
                epoch=epoch,
                num_meas=num_meas,
                existence_logits=existence_logits,
                clutter_logits=clutter_logits,
                num_tracks=num_tracks,
                attn_weights=attn_weights,
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            else:
                logging.warning("Non-finite loss; skipping step")

            active_tracks = manage_tracks(
                active_tracks,
                out,
                new_hidden_full,
                existence_probs,
                existence_logits,
                clutter_probs,
                attn_weights,
                None,
                num_tracks,
                num_meas,
                init_thresh,
                coast_thresh,
                suppress_thresh,
                del_exist,
                del_age=5,
                track_cap=track_cap,
                dt=window_size,
            )

            current_t = next_t
            windows_done += 1
            pbar.update(int(window_size))
            pbar.set_postfix(loss=np.mean(epoch_losses[-20:]) if epoch_losses else 0.0, tracks=len(active_tracks))

            if max_windows is not None and windows_done >= max_windows:
                break

        pbar.close()
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"Epoch {epoch + 1}: mean_loss={mean_loss:.4f} windows={windows_done} tracks_end={len(active_tracks)}")
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss, "windows": windows_done})

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "arch": "TransformerTrackerV7",
            },
            checkpoint_path,
        )
        print(f"Saved {checkpoint_path}")

    return {"history": history, "checkpoint": checkpoint_path, "device": str(device)}


def main():
    parser = argparse.ArgumentParser(description="Train V7 Transformer Tracker")
    parser.add_argument("--data", default="data/sim_hetero_001.jsonl")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="checkpoints/model_v7_transformer.pt")
    args = parser.parse_args()
    train_streaming(
        num_epochs=args.epochs,
        data_file=args.data,
        window_size=args.window,
        max_windows=args.max_windows,
        checkpoint_path=args.checkpoint,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

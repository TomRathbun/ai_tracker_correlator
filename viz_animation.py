import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.config_schemas import PipelineConfig
from src.pipeline import Pipeline
from src.stream_utils import load_stream_and_truth, get_truth_at_time

def create_animation(
    data_file: str,
    mode: str = "hybrid",
    gnn_path: str = None,
    duration: float = 120.0,
    window_size: float = 2.0,
    max_age: int = 10,
    min_hits: int = 2,
    out_path: str = "artifacts/tracker_simulation.gif",
    fps: int = 5,
    trail_len: int = 25,
):
    """
    Run the tracker correlator over a stream and render a time-lapse animation.

    Shows: clutter, current measurements, ground-truth (rings), confirmed tracks (stars + trails).
    """
    print(f"Creating tracker simulation (mode={mode}, duration={duration}s)...")

    config = PipelineConfig()
    config.state_updater.type = mode
    config.track_manager.max_age = max_age
    config.track_manager.min_hits = min_hits
    config.state_updater.del_age = max_age
    if gnn_path:
        config.state_updater.gnn_model_path = Path(gnn_path)

    pipeline = Pipeline(config)

    measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)

    t_start = measurements_all[0]["t"]
    t_end = t_start + duration
    current_t = t_start
    meas_idx = 0

    frames_data = []

    pbar = tqdm(total=int(duration), desc="Simulating tracker")
    while current_t < t_end:
        window_meas = []
        while (
            meas_idx < len(measurements_all)
            and measurements_all[meas_idx]["t"] < current_t + window_size
        ):
            window_meas.append(measurements_all[meas_idx])
            meas_idx += 1

        confirmed_tracks = pipeline.process_frame(window_meas, t=current_t + window_size)

        def _is_clutter(m):
            tid = m.get("track_id", -1)
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                tid = -1
            return tid == -1 or m.get("is_clutter")

        meas_pts = [
            (m["x"] / 1000.0, m["y"] / 1000.0, m.get("sensor_id", m.get("radar_id", 0)))
            for m in window_meas
            if not _is_clutter(m)
        ]
        clutter_pts = [
            (m["x"] / 1000.0, m["y"] / 1000.0) for m in window_meas if _is_clutter(m)
        ]

        track_pts = []
        for t in confirmed_tracks:
            tid = t.get("track_id", t.get("id", -1))
            track_pts.append((tid, t["x"] / 1000.0, t["y"] / 1000.0))

        truth_pts = []
        gt_tracks = get_truth_at_time(
            truth_trajectories, current_t + window_size, set(all_track_ids)
        )
        for gt in gt_tracks:
            truth_pts.append((gt["track_id"], gt["x"] / 1000.0, gt["y"] / 1000.0))

        frames_data.append(
            {
                "time": current_t + window_size,
                "meas": meas_pts,
                "clutter": clutter_pts,
                "tracks": track_pts,
                "truth": truth_pts,
                "n_tracks": len(track_pts),
                "n_meas": len(meas_pts),
            }
        )

        current_t += window_size
        pbar.update(int(window_size))
    pbar.close()

    print(f"Rendering {len(frames_data)} animation frames...")
    fig, ax = plt.subplots(figsize=(11, 10), facecolor="#0B1220")
    ax.set_facecolor("#0B1220")
    ax.set_title(
        f"Hybrid Tracker Correlator Simulation ({mode.upper()})\n"
        f"max-age={max_age}, min-hits={min_hits}  |  {Path(data_file).name}",
        color="white",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("East (km)", color="#A8C5E2")
    ax.set_ylabel("North (km)", color="#A8C5E2")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#475569")

    all_x = [p[0] for f in frames_data for p in (f["meas"] + f["clutter"] + [(t[1], t[2]) for t in f["tracks"]])]
    all_y = [p[1] for f in frames_data for p in (f["meas"] + f["clutter"] + [(t[1], t[2]) for t in f["tracks"]])]
    if all_x and all_y:
        pad = 15
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    else:
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
    ax.set_aspect("equal", adjustable="box")

    clutter_scatter = ax.scatter(
        [], [], c="#F87171", s=18, alpha=0.45, marker="x", label="Clutter", zorder=2
    )
    meas_scatter = ax.scatter(
        [], [], c="#94A3B8", s=22, alpha=0.7, marker="o", label="Measurements", zorder=3
    )
    truth_scatter = ax.scatter(
        [],
        [],
        facecolors="none",
        edgecolors="#34D399",
        s=90,
        linewidths=1.5,
        alpha=0.7,
        label="Ground truth",
        zorder=4,
    )
    track_scatter = ax.scatter(
        [],
        [],
        c="#38BDF8",
        s=100,
        marker="*",
        edgecolors="white",
        linewidths=0.6,
        zorder=6,
        label="Confirmed tracks",
    )

    time_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="white",
        va="top",
        bbox=dict(facecolor="#172033", alpha=0.9, edgecolor="#38BDF8", boxstyle="round,pad=0.4"),
    )
    legend = ax.legend(
        loc="upper right",
        facecolor="#172033",
        edgecolor="#334155",
        labelcolor="white",
        fontsize=9,
    )
    legend.get_frame().set_alpha(0.9)

    track_history = {}
    lines = {}
    cmap = plt.get_cmap("tab20")

    def init():
        meas_scatter.set_offsets(np.empty((0, 2)))
        clutter_scatter.set_offsets(np.empty((0, 2)))
        truth_scatter.set_offsets(np.empty((0, 2)))
        track_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return meas_scatter, clutter_scatter, truth_scatter, track_scatter, time_text

    def update(frame_idx):
        frame = frames_data[frame_idx]
        artists = [meas_scatter, clutter_scatter, truth_scatter, track_scatter, time_text]

        if frame["meas"]:
            meas_scatter.set_offsets(np.array([[m[0], m[1]] for m in frame["meas"]]))
        else:
            meas_scatter.set_offsets(np.empty((0, 2)))

        if frame["clutter"]:
            clutter_scatter.set_offsets(np.array([[c[0], c[1]] for c in frame["clutter"]]))
        else:
            clutter_scatter.set_offsets(np.empty((0, 2)))

        if frame["truth"]:
            truth_scatter.set_offsets(np.array([[t[1], t[2]] for t in frame["truth"]]))
        else:
            truth_scatter.set_offsets(np.empty((0, 2)))

        current_tids = set()
        if frame["tracks"]:
            track_coords = []
            for tid, tx, ty in frame["tracks"]:
                current_tids.add(tid)
                track_coords.append([tx, ty])

                if tid not in track_history:
                    track_history[tid] = []
                    (line,) = ax.plot(
                        [],
                        [],
                        color=cmap(int(tid) % 20),
                        linewidth=2.0,
                        alpha=0.85,
                        zorder=5,
                    )
                    lines[tid] = line

                track_history[tid].append((tx, ty))
                track_history[tid] = track_history[tid][-trail_len:]
                pts = np.array(track_history[tid])
                lines[tid].set_data(pts[:, 0], pts[:, 1])

            track_scatter.set_offsets(np.array(track_coords))
            track_colors = [cmap(int(tid) % 20) for tid, _, _ in frame["tracks"]]
            track_scatter.set_color(track_colors)
            track_scatter.set_edgecolor("white")
        else:
            track_scatter.set_offsets(np.empty((0, 2)))

        for tid, line in lines.items():
            if tid not in current_tids:
                # keep last trail visible faintly? clear for cleanliness
                line.set_data([], [])
            artists.append(line)

        time_text.set_text(
            f"t = {frame['time']:.1f} s   |   "
            f"tracks = {frame['n_tracks']}   |   "
            f"meas = {frame['n_meas']}"
        )
        return artists

    anim = FuncAnimation(
        fig, update, frames=len(frames_data), init_func=init, blit=False, interval=1000 // fps
    )

    os.makedirs(os.path.dirname(out_path) or "artifacts", exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Animation saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate tracker correlator over time")
    parser.add_argument(
        "--data",
        type=str,
        default="data/canonical/stream_sweden_30min_holdout.jsonl",
    )
    parser.add_argument("--mode", type=str, default="hybrid")
    parser.add_argument("--gnn", type=str, default=None)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--max-age", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--out", type=str, default="artifacts/tracker_simulation.gif")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    create_animation(
        args.data,
        args.mode,
        args.gnn,
        args.duration,
        window_size=args.window,
        max_age=args.max_age,
        min_hits=args.min_hits,
        out_path=args.out,
        fps=args.fps,
    )

"""
Visual validation for stream JSONL: plot detections over time.

Checks realism by eye:
  - Do tracks look continuous (not teleporting)?
  - Multi-radar co-detection clusters near same aircraft?
  - Clutter scattered vs structured?
  - Speeds / spacing look aviation-like?

Usage:
  uv run python scripts/data/plot_stream.py data/canonical/stream_sweden_15min.jsonl
  uv run python scripts/data/plot_stream.py data/canonical/stream_sweden_15min.jsonl --t0 0 --t1 120 --out artifacts/sweden_stream_qa.png
  uv run python scripts/data/plot_stream.py data/canonical/stream_uae_2min.jsonl --animate --duration 60
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_stream(path: Path, t0: float | None, t1: float | None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if not isinstance(m, dict) or "t" not in m:
                continue
            t = float(m["t"])
            if t0 is not None and t < t0:
                continue
            if t1 is not None and t > t1:
                continue
            rows.append(m)
    rows.sort(key=lambda r: r["t"])
    return rows


def summarize(rows):
    tids = set()
    clutter = 0
    sensors = set()
    types = defaultdict(int)
    for m in rows:
        tid = int(m.get("track_id", -1))
        if tid == -1 or m.get("is_clutter"):
            clutter += 1
        else:
            tids.add(tid)
        sid = m.get("sensor_id", m.get("radar_id"))
        if sid is not None:
            sensors.add(int(sid))
        types[str(m.get("meas_type", m.get("type", "?")))] += 1
    ts = [m["t"] for m in rows]
    return {
        "n": len(rows),
        "t0": min(ts) if ts else None,
        "t1": max(ts) if ts else None,
        "n_tracks": len(tids),
        "clutter": clutter,
        "clutter_ratio": clutter / max(len(rows), 1),
        "sensors": sorted(sensors),
        "types": dict(types),
    }


def plot_static(rows, out_path: Path, title: str, max_tracks: int = 40):
    """XY plot: GT trails + noisy detections colored by sensor; clutter in gray."""
    by_tid = defaultdict(list)
    clutter_xy = []
    for m in rows:
        tid = int(m.get("track_id", -1))
        if tid == -1 or m.get("is_clutter"):
            clutter_xy.append((m["x"] / 1000.0, m["y"] / 1000.0))
        else:
            by_tid[tid].append(m)

    # Prefer busiest tracks for clarity
    top = sorted(by_tid.keys(), key=lambda t: len(by_tid[t]), reverse=True)[:max_tracks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- Left: spatial ---
    ax = axes[0]
    if clutter_xy:
        cx, cy = zip(*clutter_xy)
        ax.scatter(cx, cy, s=4, c="lightgray", alpha=0.4, label="clutter", zorder=1)

    cmap = plt.cm.tab20
    for i, tid in enumerate(top):
        ms = sorted(by_tid[tid], key=lambda m: m["t"])
        xs = [m["x"] / 1000.0 for m in ms]
        ys = [m["y"] / 1000.0 for m in ms]
        # GT if present
        gxs = [m["gt_x"] / 1000.0 for m in ms if m.get("gt_x") is not None]
        gys = [m["gt_y"] / 1000.0 for m in ms if m.get("gt_y") is not None]
        color = cmap(i % 20)
        if gxs:
            ax.plot(gxs, gys, "-", color=color, lw=1.5, alpha=0.9, zorder=3)
        ax.scatter(xs, ys, s=8, c=[color], alpha=0.35, zorder=2)

    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_title("Spatial: GT trails + detections")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # --- Right: time vs range from origin (or y) ---
    ax2 = axes[1]
    for i, tid in enumerate(top[:15]):
        ms = sorted(by_tid[tid], key=lambda m: m["t"])
        ts = [m["t"] for m in ms]
        # ground speed estimate from consecutive GT if available
        if len(ms) >= 2 and ms[0].get("gt_x") is not None:
            speeds = []
            t_mid = []
            for a, b in zip(ms[:-1], ms[1:]):
                dt = b["t"] - a["t"]
                if dt <= 0:
                    continue
                dx = (b.get("gt_x", b["x"]) - a.get("gt_x", a["x"]))
                dy = (b.get("gt_y", b["y"]) - a.get("gt_y", a["y"]))
                speeds.append(np.hypot(dx, dy) / dt)
                t_mid.append(0.5 * (a["t"] + b["t"]))
            if speeds:
                ax2.plot(t_mid, speeds, "-", color=cmap(i % 20), alpha=0.8, label=f"id {tid}")
        else:
            ax2.scatter(ts, [np.hypot(m["x"], m["y"]) / 1000.0 for m in ms], s=6, alpha=0.5)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s) if GT else range (km)")
    ax2.set_title("Temporal: track speed (or range)")
    ax2.grid(True, alpha=0.3)
    if top:
        ax2.legend(fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_animation(rows, out_path: Path, window_s: float = 2.0, fps: int = 5):
    """Simple animated XY over time (GIF)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not rows:
        print("No rows to animate")
        return

    t0, t1 = rows[0]["t"], rows[-1]["t"]
    xs_all = [m["x"] / 1000.0 for m in rows]
    ys_all = [m["y"] / 1000.0 for m in rows]
    pad = 5
    xlim = (min(xs_all) - pad, max(xs_all) + pad)
    ylim = (min(ys_all) - pad, max(ys_all) + pad)

    fig, ax = plt.subplots(figsize=(9, 9))
    scat_tgt = ax.scatter([], [], s=20, c="C0", label="targets")
    scat_fa = ax.scatter([], [], s=10, c="0.7", label="clutter")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    title = ax.set_title("")

    times = np.arange(t0, t1, window_s)
    # Pre-index by time for speed
    ts = np.array([m["t"] for m in rows])

    def frame(i):
        tw = times[i]
        mask = (ts >= tw) & (ts < tw + window_s)
        idxs = np.where(mask)[0]
        tx, ty, fx, fy = [], [], [], []
        for j in idxs:
            m = rows[j]
            if int(m.get("track_id", -1)) == -1 or m.get("is_clutter"):
                fx.append(m["x"] / 1000.0)
                fy.append(m["y"] / 1000.0)
            else:
                tx.append(m["x"] / 1000.0)
                ty.append(m["y"] / 1000.0)
        scat_tgt.set_offsets(np.column_stack([tx, ty]) if tx else np.empty((0, 2)))
        scat_fa.set_offsets(np.column_stack([fx, fy]) if fx else np.empty((0, 2)))
        title.set_text(f"t = {tw:.1f} – {tw + window_s:.1f} s  |  targets={len(tx)} clutter={len(fx)}")
        return scat_tgt, scat_fa, title

    anim = FuncAnimation(fig, frame, frames=len(times), interval=1000 // fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Visual QA for stream training data")
    ap.add_argument("path", help="Stream JSONL path")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--window", type=float, default=2.0, help="Animation window seconds")
    ap.add_argument("--max-tracks", type=int, default=40)
    args = ap.parse_args()

    path = Path(args.path)
    rows = load_stream(path, args.t0, args.t1)
    stats = summarize(rows)
    print("=== Stream summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if not rows:
        print("No measurements in window.")
        return

    stem = path.stem
    if args.animate:
        out = Path(args.out or f"artifacts/{stem}_qa.gif")
        plot_animation(rows, out, window_s=args.window)
    else:
        out = Path(args.out or f"artifacts/{stem}_qa.png")
        title = f"{path.name}  |  n={stats['n']} tracks={stats['n_tracks']} clutter={stats['clutter_ratio']:.1%}"
        plot_static(rows, out, title, max_tracks=args.max_tracks)


if __name__ == "__main__":
    main()

"""
Scenario difficulty analysis for stream JSONL training data.

Reports: track counts, altitude/speed distributions, nearest-neighbor distances,
multi-sensor overlap, concurrent traffic, and a simple difficulty score.

Usage:
  uv run python scripts/data/difficulty_report.py data/canonical/stream_sweden_15min.jsonl
  uv run python scripts/data/difficulty_report.py data/canonical --out artifacts/data_difficulty_report.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_stream(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if isinstance(m, dict) and "t" in m:
                rows.append(m)
    return rows


def analyze(path: Path) -> Dict[str, Any]:
    rows = load_stream(path)
    if not rows:
        return {"path": str(path), "error": "empty"}

    by_tid: Dict[int, List[dict]] = defaultdict(list)
    clutter = 0
    sensors = set()
    types = defaultdict(int)
    for m in rows:
        tid = int(m.get("track_id", -1))
        if tid < 0 or m.get("is_clutter"):
            clutter += 1
        else:
            by_tid[tid].append(m)
        sid = m.get("sensor_id", m.get("radar_id"))
        if sid is not None:
            sensors.add(int(sid))
        types[str(m.get("meas_type", m.get("type", "?")))] += 1

    ts = [m["t"] for m in rows]
    t0, t1 = min(ts), max(ts)

    # Per-track stats
    alts, speeds, durs = [], [], []
    samples = []  # (tid, t, x, y, z)
    for tid, ms in by_tid.items():
        ms = sorted(ms, key=lambda x: x["t"])
        durs.append(ms[-1]["t"] - ms[0]["t"])
        zs = [
            float(m["gt_z"]) if m.get("gt_z") is not None else float(m.get("z", 0))
            for m in ms
        ]
        if zs:
            alts.append(float(np.median(zs)))
        if len(ms) >= 2 and ms[0].get("gt_x") is not None:
            dt = max(ms[-1]["t"] - ms[0]["t"], 1e-6)
            dx = float(ms[-1]["gt_x"]) - float(ms[0]["gt_x"])
            dy = float(ms[-1]["gt_y"]) - float(ms[0]["gt_y"])
            speeds.append(math.hypot(dx, dy) / dt)
        # sample every ~10s
        last_t = -1e9
        for m in ms:
            if m["t"] - last_t >= 10:
                x = float(m["gt_x"]) if m.get("gt_x") is not None else float(m["x"])
                y = float(m["gt_y"]) if m.get("gt_y") is not None else float(m["y"])
                z = float(m["gt_z"]) if m.get("gt_z") is not None else float(m.get("z", 0))
                samples.append((tid, m["t"], x, y, z))
                last_t = m["t"]

    # Nearest neighbor distances (different tracks, |dt|<15s)
    nn_dists = []
    nn_dz = []
    close_pairs = 0  # < 5 km
    close_diff_alt = 0  # < 5 km and |dz| > 500 m
    by_time_bin: Dict[int, List[Tuple]] = defaultdict(list)
    for s in samples:
        by_time_bin[int(s[1] // 15)].append(s)

    for b, group in by_time_bin.items():
        # also check adjacent bins
        cand = group + by_time_bin.get(b - 1, []) + by_time_bin.get(b + 1, [])
        for i in range(len(group)):
            tid1, t1, x1, y1, z1 = group[i]
            best = None
            for tid2, t2, x2, y2, z2 in cand:
                if tid2 == tid1:
                    continue
                if abs(t2 - t1) > 15:
                    continue
                d = math.hypot(x1 - x2, y1 - y2)
                if best is None or d < best[0]:
                    best = (d, abs(z1 - z2))
            if best:
                nn_dists.append(best[0])
                nn_dz.append(best[1])
                if best[0] < 5000:
                    close_pairs += 1
                    if best[1] > 500:
                        close_diff_alt += 1

    # Concurrent tracks per 1s bin
    conc = defaultdict(set)
    for m in rows:
        tid = int(m.get("track_id", -1))
        if tid < 0:
            continue
        conc[int(m["t"])].add(tid)
    conc_counts = [len(v) for v in conc.values()] if conc else [0]

    # Multi-sensor: same track, same 1s bin, multiple sensors
    multi_num = multi_den = 0
    track_sec_sensors: Dict[Tuple[int, int], set] = defaultdict(set)
    for m in rows:
        tid = int(m.get("track_id", -1))
        if tid < 0:
            continue
        sid = m.get("sensor_id", m.get("radar_id"))
        if sid is None:
            continue
        track_sec_sensors[(tid, int(m["t"]))].add(int(sid))
    for sids in track_sec_sensors.values():
        multi_den += 1
        if len(sids) >= 2:
            multi_num += 1

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else None

    # Difficulty score 0-100 (heuristic)
    score = 0.0
    if conc_counts:
        score += min(30.0, float(np.median(conc_counts)) * 5)  # concurrency
    if nn_dists:
        med_nn = float(np.median(nn_dists))
        # closer neighbors => harder
        score += max(0.0, min(30.0, 30.0 * (10000 - med_nn) / 10000))
    score += min(20.0, close_pairs / max(len(nn_dists), 1) * 100)  # fraction close
    score += min(20.0, multi_num / max(multi_den, 1) * 40)  # multi-sensor

    return {
        "path": str(path).replace("\\", "/"),
        "n_meas": len(rows),
        "duration_s": t1 - t0,
        "duration_min": round((t1 - t0) / 60, 2),
        "n_tracks": len(by_tid),
        "clutter_ratio": round(clutter / len(rows), 4),
        "n_sensors": len(sensors),
        "types": dict(types),
        "concurrent_median": float(np.median(conc_counts)),
        "concurrent_max": int(max(conc_counts)),
        "alt_m_p10": pct(alts, 10),
        "alt_m_p50": pct(alts, 50),
        "alt_m_p90": pct(alts, 90),
        "speed_mps_p50": pct(speeds, 50),
        "track_duration_s_p50": pct(durs, 50),
        "nn_dist_m_p10": pct(nn_dists, 10),
        "nn_dist_m_p50": pct(nn_dists, 50),
        "nn_samples": len(nn_dists),
        "close_pairs_lt_5km": close_pairs,
        "close_pairs_diff_alt": close_diff_alt,
        "multi_sensor_frac": round(multi_num / max(multi_den, 1), 4),
        "difficulty_score_0_100": round(score, 1),
        "difficulty_band": (
            "hard" if score >= 60 else "medium" if score >= 35 else "easy"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="JSONL file or directory of streams")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    target = Path(args.path)
    files = sorted(target.glob("stream_*.jsonl")) if target.is_dir() else [target]
    # also top-level canonical streams
    if target.is_dir():
        files = sorted(set(files) | set(target.glob("*.jsonl")))
        files = [f for f in files if f.name.startswith("stream_") or "hetero" in f.name]

    results = []
    for f in files:
        print(f"Analyzing {f} ...")
        # skip batch hetero for stream metrics or handle lightly
        with open(f, encoding="utf-8") as fh:
            first = fh.readline()
        if not first:
            continue
        obj = json.loads(first)
        if "measurements" in obj:
            print("  skip batch frame file")
            continue
        results.append(analyze(f))

    for r in results:
        if "error" in r:
            print(r)
            continue
        print(
            f"  {Path(r['path']).name}: tracks={r['n_tracks']} "
            f"dur={r['duration_min']}m conc_med={r['concurrent_median']} "
            f"nn_p50={r['nn_dist_m_p50']} score={r['difficulty_score_0_100']} ({r['difficulty_band']})"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Training Data Difficulty Report",
            "",
            "Heuristic score (0–100): concurrency, nearest-neighbor closeness, multi-sensor overlap.",
            "",
            "| File | Dur (min) | Tracks | Conc med/max | NN p50 (m) | Close &lt;5km | Multi-sensor | Score | Band |",
            "|------|----------:|-------:|-------------:|-----------:|-------------:|-------------:|------:|------|",
        ]
        for r in results:
            if "error" in r:
                continue
            nn = r.get("nn_dist_m_p50")
            nn_s = f"{nn:.0f}" if nn is not None else "—"
            lines.append(
                f"| `{Path(r['path']).name}` | {r['duration_min']} | {r['n_tracks']} | "
                f"{r['concurrent_median']:.0f}/{r['concurrent_max']} | "
                f"{nn_s} | {r['close_pairs_lt_5km']} | "
                f"{r['multi_sensor_frac']:.2f} | {r['difficulty_score_0_100']} | {r['difficulty_band']} |"
            )
        lines += ["", "## Details", "", "```json", json.dumps(results, indent=2), "```", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the hybrid correlator on a PlotForge canonical stream."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config_schemas import PipelineConfig
from src.metrics import TrackingMetrics
from src.pipeline import Pipeline
from src.stream_utils import get_truth_at_time, load_stream_and_truth

ORIGIN_LAT = 24.4539
ORIGIN_LON = 54.3773
LAT_M = 111320.0


def enu_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    lat = ORIGIN_LAT + y / LAT_M
    lon = ORIGIN_LON + x / (LAT_M * np.cos(np.radians(ORIGIN_LAT)))
    return float(lat), float(lon), float(z)


def run(args: argparse.Namespace) -> dict:
    cfg = PipelineConfig()
    cfg.state_updater.type = "hybrid"
    cfg.pairwise.backend = args.assoc
    cfg.pairwise.cluster_backend = args.cluster_assoc or args.assoc
    cfg.pairwise.assign_backend = args.assign_assoc or args.assoc
    cfg.pairwise.v8_model_path = Path(args.v8_model_path)
    cfg.pairwise.cluster_threshold = args.cluster_threshold
    cfg.pairwise.assign_threshold = args.assign_threshold
    cfg.pairwise.use_dustbin = bool(args.dustbin)
    cfg.track_manager.min_hits = args.min_hits
    cfg.track_manager.max_age = args.max_age
    cfg.state_updater.del_age = args.max_age
    cfg.clutter_filter.enabled = not args.no_clutter_filter
    cfg.clutter_filter.threshold = args.clutter_threshold
    if args.clutter_model:
        cfg.clutter_filter.model_path = Path(args.clutter_model)
    if args.psr_model:
        cfg.pairwise.psr_model_path = Path(args.psr_model)
    if args.ssr_model:
        cfg.pairwise.ssr_model_path = Path(args.ssr_model)

    pipeline = Pipeline(cfg)
    v8 = getattr(pipeline.state_updater, "v8", None)
    if v8 is not None:
        if args.v8_gated_encode:
            v8.gated_encode = True
        if args.v8_temperature is not None:
            v8.temperature.fill_(float(args.v8_temperature))
    measurements, truth_trajectories, all_ids = load_stream_and_truth(args.data)
    measurements.sort(key=lambda m: m["t"])
    t_start = measurements[0]["t"]
    t_end = measurements[-1]["t"]
    window = 1.0

    metrics = TrackingMetrics(match_threshold=args.match_threshold)
    snapshots = []
    collect_snaps = not args.no_snapshots
    idx = 0
    current = t_start
    frames = 0
    t0 = time.perf_counter()

    while current < t_end:
        window_meas = []
        while idx < len(measurements) and measurements[idx]["t"] < current + window:
            window_meas.append(measurements[idx])
            idx += 1
        frame_t = current + window
        pred = pipeline.process_frame(window_meas, t=frame_t)
        gt = get_truth_at_time(truth_trajectories, frame_t, set(all_ids))
        pred_ids = [int(p.get("track_id", i)) for i, p in enumerate(pred)]
        metrics.update(pred, gt, pred_ids=pred_ids)
        frames += 1
        if collect_snaps and frames % 2 == 0:
            snap_tracks = []
            for p in pred:
                lat, lon, alt_m = enu_to_lla(float(p["x"]), float(p["y"]), float(p.get("z", 0.0)))
                snap_tracks.append(
                    {
                        "tn": int(p.get("track_id", -1)),
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "altFt": round(alt_m / 0.3048),
                        "gsKt": round(float(np.hypot(p.get("vx") or 0, p.get("vy") or 0)) / 0.514444, 1),
                        "mode3a": p.get("mode_3a") or p.get("mode3a"),
                        "callsign": p.get("callsign") or p.get("mode_s"),
                    }
                )
            snapshots.append({"t": round(frame_t, 3), "nPred": len(pred), "nGt": len(gt), "tracks": snap_tracks})
        current += window

    elapsed = time.perf_counter() - t0
    m = metrics.compute()
    result = {
        "ok": True,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "data": args.data,
        "mode": "hybrid",
        "assoc": args.assoc,
        "clusterBackend": args.cluster_assoc or args.assoc,
        "assignBackend": args.assign_assoc or args.assoc,
        "minHits": args.min_hits,
        "maxAge": args.max_age,
        "matchThresholdM": args.match_threshold,
        "clusterThreshold": args.cluster_threshold,
        "assignThreshold": args.assign_threshold,
        "dustbin": bool(args.dustbin),
        "v8GatedEncode": bool(args.v8_gated_encode),
        "v8Temperature": args.v8_temperature,
        "checkpoints": {
            "clutter": args.clutter_model,
            "psr": args.psr_model,
            "ssr": args.ssr_model,
            "v8": args.v8_model_path if args.assoc == "transformer" else None,
        },
        "durationS": round(t_end - t_start, 3),
        "nMeasurements": len(measurements),
        "nTruthTracks": len(all_ids),
        "nFrames": frames,
        "elapsedS": round(elapsed, 2),
        "metrics": {
            "mota": round(float(m["mota"]), 4),
            "motpM": round(float(m["motp"]), 1),
            "precision": round(float(m["precision"]), 4),
            "recall": round(float(m["recall"]), 4),
            "f1": round(float(m["f1"]), 4),
            "idSwitches": int(m["id_switches"]),
            "fp": int(m.get("fp", 0)),
            "fn": int(m.get("fn", 0)),
            "matches": int(m.get("total_matches", 0)),
            "fpPerFrame": round(float(m["fp_rate"]), 2),
            "fnPerFrame": round(float(m["fn_rate"]), 2),
        },
        "baselineSwedenHoldout": {
            "source": "artifacts/TRAINING_DATA_CHAPTER.md hybrid max-age=10 min-hits=2",
            "mota": 0.976,
            "motpM": 105,
            "precision": 0.979,
            "recall": 0.998,
            "idSwitches": 0,
        },
        "snapshots": snapshots,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--assoc", default="mlp", choices=["mlp", "transformer", "ensemble"])
    p.add_argument("--cluster-assoc", default=None, choices=["mlp", "transformer", "ensemble"])
    p.add_argument("--assign-assoc", default=None, choices=["mlp", "transformer", "ensemble"])
    p.add_argument("--v8-model-path", default="checkpoints/model_v8_assoc_best.pt")
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--max-age", type=int, default=10)
    p.add_argument("--match-threshold", type=float, default=7000.0)
    p.add_argument("--clutter-threshold", type=float, default=0.70)
    p.add_argument("--clutter-model", default="checkpoints/clutter_classifier.pt")
    p.add_argument("--psr-model", default="checkpoints/pairwise_psr_psr.pt")
    p.add_argument("--ssr-model", default="checkpoints/pairwise_ssr_any.pt")
    p.add_argument("--cluster-threshold", type=float, default=0.5)
    p.add_argument("--assign-threshold", type=float, default=0.0)
    p.add_argument("--dustbin", action="store_true")
    p.add_argument("--v8-gated-encode", action="store_true")
    p.add_argument("--v8-temperature", type=float, default=None)
    p.add_argument("--no-snapshots", action="store_true")
    p.add_argument("--no-clutter-filter", action="store_true")
    args = p.parse_args()
    res = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    m = res["metrics"]
    print("=" * 60)
    print("PLOTFORGE × HYBRID CORRELATOR")
    print("=" * 60)
    print(f"assoc={res['assoc']} cluster={res.get('clusterBackend')} assign={res.get('assignBackend')}  frames={res['nFrames']}  meas={res['nMeasurements']}  elapsed={res['elapsedS']}s")
    print(f"MOTA:      {m['mota']:.4f}")
    print(f"MOTP:      {m['motpM']:.1f} m")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1:        {m['f1']:.4f}")
    print(f"ID Switch: {m['idSwitches']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

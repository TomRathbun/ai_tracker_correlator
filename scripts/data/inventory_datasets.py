"""
Phase 0: Inventory all data/*.jsonl files — format, region, quality signals.

Usage:
  uv run python scripts/data/inventory_datasets.py
  uv run python scripts/data/inventory_datasets.py --out data/canonical/DATA_MANIFEST.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# Rough geo bounds for region tagging
REGION_BOUNDS = {
    "uae": {"lat": (15.0, 35.0), "lon": (45.0, 65.0)},
    "sweden": {"lat": (54.0, 70.0), "lon": (8.0, 28.0)},
}


def detect_format(obj: dict) -> str:
    if "measurements" in obj and isinstance(obj.get("measurements"), list):
        return "batch"
    if "t" in obj or "radar_id" in obj or "meas_type" in obj:
        return "stream"
    if "timestamp" in obj and "x" in obj:
        return "stream_like"
    return "unknown"


def sample_lines(path: Path, max_lines: int = 2000, stride: int = 1) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if stride > 1 and i % stride != 0:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= max_lines:
                break
    return rows


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def collect_lat_lon(rows: List[dict], fmt: str) -> Tuple[List[float], List[float]]:
    lats, lons = [], []
    for r in rows:
        if fmt == "batch":
            for m in r.get("measurements", []) or []:
                if not isinstance(m, dict):
                    continue
                lat, lon = m.get("source_lat"), m.get("source_lon")
                if lat is not None and lon is not None:
                    try:
                        lats.append(float(lat))
                        lons.append(float(lon))
                    except (TypeError, ValueError):
                        pass
        else:
            lat, lon = r.get("source_lat"), r.get("source_lon")
            if lat is not None and lon is not None:
                try:
                    lats.append(float(lat))
                    lons.append(float(lon))
                except (TypeError, ValueError):
                    pass
    return lats, lons


def infer_region(lats: List[float], lons: List[float]) -> str:
    if not lats:
        return "unknown_or_synthetic"
    med_lat = sorted(lats)[len(lats) // 2]
    med_lon = sorted(lons)[len(lons) // 2]
    for name, b in REGION_BOUNDS.items():
        if b["lat"][0] <= med_lat <= b["lat"][1] and b["lon"][0] <= med_lon <= b["lon"][1]:
            return name
    return f"other(lat={med_lat:.2f},lon={med_lon:.2f})"


def field_presence_stream(rows: List[dict]) -> Dict[str, float]:
    keys = [
        "t", "radar_id", "sensor_id", "meas_type", "type",
        "x", "y", "z", "vx", "vy", "amplitude",
        "mode3a", "mode_3a", "mode_s", "track_id",
        "gt_x", "gt_y", "gt_z", "gt_vx", "gt_vy",
        "source_lat", "source_lon", "is_clutter", "schema_version",
    ]
    n = max(len(rows), 1)
    return {k: sum(1 for r in rows if k in r and r[k] is not None) / n for k in keys}


def field_presence_batch(rows: List[dict]) -> Dict[str, float]:
    meas = []
    for r in rows:
        for m in r.get("measurements", []) or []:
            if isinstance(m, dict):
                meas.append(m)
    return field_presence_stream(meas) if meas else {}


def stream_stats(rows: List[dict]) -> Dict[str, Any]:
    ts, tids, sensors, types = [], [], [], []
    clutter = 0
    float_tid = 0
    for r in rows:
        t = r.get("t", r.get("timestamp"))
        if t is not None:
            try:
                ts.append(float(t))
            except (TypeError, ValueError):
                pass
        tid = r.get("track_id", -1)
        if isinstance(tid, float) and not float(tid).is_integer():
            float_tid += 1
        try:
            tid_i = int(tid) if tid is not None else -1
        except (TypeError, ValueError):
            tid_i = -1
        if tid_i == -1:
            clutter += 1
        else:
            tids.append(tid_i)
        sid = r.get("radar_id", r.get("sensor_id"))
        if sid is not None:
            sensors.append(sid)
        mt = r.get("meas_type", r.get("type"))
        if mt is not None:
            types.append(mt)
    n = max(len(rows), 1)
    return {
        "t_min": min(ts) if ts else None,
        "t_max": max(ts) if ts else None,
        "duration_s": (max(ts) - min(ts)) if ts else None,
        "unique_tracks_sample": len(set(tids)),
        "clutter_ratio_sample": clutter / n,
        "float_track_id_sample": float_tid,
        "sensors": dict(Counter(str(s) for s in sensors)),
        "types": dict(Counter(str(t) for t in types)),
    }


def batch_stats(rows: List[dict]) -> Dict[str, Any]:
    n_meas = 0
    clutter = 0
    tids = set()
    for r in rows:
        for m in r.get("measurements", []) or []:
            if not isinstance(m, dict):
                continue
            n_meas += 1
            tid = m.get("track_id", -1)
            try:
                tid_i = int(tid)
            except (TypeError, ValueError):
                tid_i = -1
            if tid_i == -1:
                clutter += 1
            else:
                tids.add(tid_i)
    n = max(n_meas, 1)
    return {
        "frames_sampled": len(rows),
        "meas_sampled": n_meas,
        "unique_tracks_sample": len(tids),
        "clutter_ratio_sample": clutter / n,
        "has_gt_tracks": sum(1 for r in rows if r.get("gt_tracks")) / max(len(rows), 1),
    }


def recommend_status(name: str, region: str, fmt: str) -> Tuple[str, str]:
    """Return (status, note)."""
    lname = name.lower()
    if "sweden" in lname and region == "uae":
        return "misnamed", "Filename says Sweden but source_lat/lon are UAE — do not train as Sweden"
    if region == "sweden" and fmt == "stream":
        return "ok_sweden", "True Sweden geo — good candidate for canonical regen/eval"
    if region == "uae" and fmt == "stream":
        if "sweden" in lname:
            return "misnamed", "UAE traffic under Sweden name"
        return "legacy_uae", "UAE stream; keep as legacy until canonical/stream_uae_* exists"
    if fmt == "batch":
        if "hetero" in lname:
            return "ok_batch", "Batch hetero sim — primary pairwise/clutter train source today"
        return "legacy_batch", "Batch sim — verify schema (type/mode fields) before training"
    if fmt == "unknown":
        return "do_not_train", "Unrecognized format"
    return "legacy", "Review before use"


def inventory_file(path: Path) -> Dict[str, Any]:
    n_lines = count_lines(path)
    # Adaptive sampling for large files
    stride = max(1, n_lines // 2000) if n_lines > 2000 else 1
    rows = sample_lines(path, max_lines=2000, stride=stride)
    if not rows:
        return {"path": str(path.relative_to(ROOT)), "error": "empty or unreadable", "status": "do_not_train"}

    fmt = detect_format(rows[0])
    lats, lons = collect_lat_lon(rows, fmt)
    region = infer_region(lats, lons)
    status, note = recommend_status(path.name, region, fmt)

    rec: Dict[str, Any] = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "n_lines": n_lines,
        "format": fmt,
        "region_inferred": region,
        "status": status,
        "note": note,
        "lat_min": min(lats) if lats else None,
        "lat_max": max(lats) if lats else None,
        "lon_min": min(lons) if lons else None,
        "lon_max": max(lons) if lons else None,
        "sample_stride": stride,
    }
    if fmt == "batch":
        rec["stats"] = batch_stats(rows)
        rec["field_presence"] = field_presence_batch(rows)
    else:
        rec["stats"] = stream_stats(rows)
        rec["field_presence"] = field_presence_stream(rows)
    return rec


def write_manifest(records: List[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Data Manifest (auto-generated)",
        "",
        "Generated by `scripts/data/inventory_datasets.py`.",
        "",
        "## Status legend",
        "",
        "| Status | Meaning |",
        "|--------|---------|",
        "| `ok_batch` | Safe batch training source |",
        "| `ok_sweden` | True Sweden geography |",
        "| `legacy_uae` | UAE stream; use until canonical UAE file exists |",
        "| `legacy_batch` | Older batch sim; check fields |",
        "| `misnamed` | **Do not treat as labeled region** |",
        "| `do_not_train` | Unusable / empty |",
        "",
        "## Files",
        "",
        "| Path | Format | Region | Lines | Status | Note |",
        "|------|--------|--------|------:|--------|------|",
    ]
    for r in sorted(records, key=lambda x: x.get("path", "")):
        if "error" in r:
            lines.append(
                f"| `{r['path']}` | — | — | — | `{r.get('status')}` | {r['error']} |"
            )
            continue
        note = (r.get("note") or "").replace("|", "/")
        lines.append(
            f"| `{r['path']}` | {r['format']} | {r['region_inferred']} | "
            f"{r['n_lines']:,} | `{r['status']}` | {note} |"
        )

    lines += [
        "",
        "## Training script defaults (freeze snapshot)",
        "",
        "| Script | Default data | Expected format |",
        "|--------|--------------|-----------------|",
        "| `scripts/train_hetero_pairwise.py` | `data/sim_hetero_001.jsonl` | batch |",
        "| `scripts/train_clutter_filter.py` | `data/sim_hetero_001.jsonl` | batch |",
        "| `scripts/train_gnn_tracker.py` | `data/sim_hetero_001.jsonl` | batch |",
        "| `src/train_streaming_v3–v6.py` | `data/stream_radar_001.jsonl` | stream |",
        "| `scripts/pretrain_v6_clutter.py` | `data/sweden_v2_60min.jsonl` | stream (**misnamed UAE**) |",
        "| `scripts/auto_eval.py` | `data/sweden_radar_subset.jsonl` | stream |",
        "",
        "## Canonical targets (Phase 2+)",
        "",
        "| Path | Purpose |",
        "|------|---------|",
        "| `data/canonical/sim_batch_hetero.jsonl` | Normalized batch hetero |",
        "| `data/canonical/stream_uae_2min.jsonl` | Clean UAE multi-radar stream |",
        "| `data/canonical/stream_sweden_15min.jsonl` | Clean Sweden multi-radar stream |",
        "| `data/canonical/episodes/` | 60–180s clips + manifests |",
        "",
        "## Per-file detail (JSON)",
        "",
        "See companion `DATA_INVENTORY.json` next to this file.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Inventory radar training JSONL datasets")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "canonical" / "DATA_MANIFEST.md"),
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(DATA_DIR / "canonical" / "DATA_INVENTORY.json"),
    )
    args = parser.parse_args()

    files = sorted(DATA_DIR.glob("*.jsonl"))
    records = []
    print(f"Scanning {len(files)} jsonl files under {DATA_DIR} ...")
    for path in files:
        print(f"  {path.name} ...", end=" ", flush=True)
        rec = inventory_file(path)
        records.append(rec)
        print(rec.get("status", rec.get("error", "?")))

    write_manifest(records, Path(args.out))
    Path(args.json_out).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

"""
Build a longer multi-target CAT-62-like text file by tiling a dense short scenario.

The Sweden subset spans hours/days but is mostly 1–2 concurrent tracks. The mini
extract (~10 min, ~55 tracks, up to ~8 concurrent) is the dense multi-target core.

This script tiles that dense core into a longer scenario for 30/60 min stream
generation, with remapped track numbers and small spatial offsets per tile so
tracks do not perfectly overlap.

Usage:
  uv run python scripts/data/build_dense_cat62.py \\
    --input data/cat_62_sweden_mini.txt \\
    --output data/canonical/cat62_sweden_dense_30min.txt \\
    --tiles 3 --tile-gap 5
"""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path


def load_cat62(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = ast.literal_eval(line)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("category") == 62:
                rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tiles", type=int, default=3, help="Number of tiled copies")
    ap.add_argument(
        "--tile-gap",
        type=float,
        default=5.0,
        help="Seconds between end of one tile and start of next",
    )
    ap.add_argument(
        "--offset-m",
        type=float,
        default=15000.0,
        help="Spatial offset (m) per tile along a diagonal to reduce perfect overlap",
    )
    args = ap.parse_args()

    rows = load_cat62(Path(args.input))
    if not rows:
        raise SystemExit(f"No CAT-62 records in {args.input}")

    times = [float(r["time"]) for r in rows]
    t_min, t_max = min(times), max(times)
    duration = t_max - t_min
    print(f"Source: {len(rows)} records, duration={duration:.1f}s, "
          f"unique tracks={len({str(r['track_number']) for r in rows})}")

    # Map original track numbers to integers
    orig_ids = sorted({str(r["track_number"]) for r in rows}, key=str)
    base_map = {oid: i + 1 for i, oid in enumerate(orig_ids)}

    out_rows = []
    for tile in range(args.tiles):
        t_shift = tile * (duration + args.tile_gap)
        # spatial offset: spiral-ish so tiles don't sit on top of each other
        ang = tile * 2.2
        dx = args.offset_m * tile * math.cos(ang)
        dy = args.offset_m * tile * math.sin(ang)
        id_shift = tile * 100000

        for r in rows:
            nr = dict(r)
            nr["time"] = float(r["time"]) - t_min + t_shift
            nr["x"] = float(r["x"]) + dx
            nr["y"] = float(r["y"]) + dy
            if r.get("lat") is not None and r.get("lon") is not None:
                # rough ENU inverse for geo tags (display only)
                # leave lat/lon offset proportionally ~111.32 km/deg
                nr["lat"] = float(r["lat"]) + dy / 111320.0
                nr["lon"] = float(r["lon"]) + dx / (111320.0 * math.cos(math.radians(float(r["lat"]))))
            tid = base_map[str(r["track_number"])] + id_shift
            nr["track_number"] = tid
            out_rows.append(nr)

    out_rows.sort(key=lambda r: r["time"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(repr(r) + "\n")

    print(f"Wrote {len(out_rows)} records -> {out}")
    print(f"  tiles={args.tiles} total_span={(out_rows[-1]['time'] - out_rows[0]['time']):.1f}s")
    print(f"  unique tracks={len({r['track_number'] for r in out_rows})}")


if __name__ == "__main__":
    main()

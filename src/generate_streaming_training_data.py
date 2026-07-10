"""
generate_streaming_training_data.py

Reads real ASTERIX CAT 62 track data (cat_62_data.txt) and generates a
streaming radar measurement dataset that mimics how a real tracker receives
data — one measurement at a time, from multiple radars, with independent,
staggered scan timings (not perfectly aligned).

Output format (JSONL) — one measurement (dict) per line:
{
  "t":          <float>  wall-clock time in seconds from epoch start,
  "radar_id":   <int>    0–4 (5 simulated radar sites),
  "meas_type":  "PSR" | "SSR",
  "x":          <float>  metres east  (from UAE reference origin),
  "y":          <float>  metres north (from UAE reference origin),
  "z":          <float>  metres altitude (simulated, see below),
  "vx":         <float>  m/s east  (PSR only; absent for SSR),
  "vy":         <float>  m/s north (PSR only; absent for SSR),
  "amplitude":  <float>  radar return amplitude dBZ (PSR only),
  "mode3a":     <str>    squawk code in octal (SSR only; absent for PSR),
  "mode_s":     <str>    24-bit hex ICAO address (SSR only; absent for PSR),
  "track_id":   <int>    ground-truth aircraft track_number (-1 = false alarm),
  "source_lat": <float>  original lat from CAT-62 record,
  "source_lon": <float>  original lon from CAT-62 record
}

Design decisions
----------------
* 5 radar sites are placed around the UAE coverage area.  Each radar has an
  independent rotation rate (approx 6–12 rpm → 5–10 s per scan).
* The raw CAT-62 data covers one large ASTERIX feed that spans different
  geographic sectors in the same second-of-day time-band.  We interpret
  each unique (time, track_number) pair as one aircraft at one epoch.
* For each radar we compute whether the aircraft is "visible" (within
  max_range_m) and, if so, emit a PSR and/or SSR return with realistic
  Gaussian noise and staggered timing.
* Altitude is estimated from mode-C (not available in raw data) so we
  assign a synthetic cruise altitude based on speed magnitude with light
  noise, giving a plausible z value.
* False-alarm clutter returns are injected at a Poisson rate per radar.
* The output timeline starts at t=0 and advances over the CAT-62 recording
  duration converted to wall-clock seconds.
"""

import ast
import json
import math
import os
import random
import re
import sys
from typing import List, Dict
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate multi-radar streaming detections from CAT-62 scenario truth."
    )
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "cat_62_data.txt"))
    # Never default to sim_hetero_001.jsonl (that path is the batch hetero dataset).
    parser.add_argument(
        "--output", type=str, default=str(DATA_DIR / "canonical" / "stream_uae_2min.jsonl")
    )
    parser.add_argument("--region", type=str, default=os.environ.get("TRACKER_REGION", "uae"))
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="Cap output wall-clock duration in seconds (e.g. 120 for 2 min, 900 for 15 min).",
    )
    parser.add_argument(
        "--dataset-id", type=str, default=None,
        help="Optional dataset_id tag written on every record.",
    )
    return parser.parse_args()

# ── reproducibility ─────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
random.seed(RNG_SEED)

# ── regional configurations ──────────────────────────────────────────────────
REGIONS = {
    "uae": {
        "origin_lat": 24.4539,
        "origin_lon": 54.3773,
        "radar_sites": [
            {"id": 0, "x": 0.0,      "y": 0.0,      "max_range": 450_000, "scan_period": 7.0,  "psr_prob": 0.92, "ssr_prob": 0.88},
            {"id": 1, "x": 150_000,  "y": 100_000,  "max_range": 400_000, "scan_period": 5.5,  "psr_prob": 0.90, "ssr_prob": 0.85},
            {"id": 2, "x": -200_000, "y": 200_000,  "max_range": 420_000, "scan_period": 8.0,  "psr_prob": 0.88, "ssr_prob": 0.83},
            {"id": 3, "x": 300_000,  "y": -50_000,  "max_range": 380_000, "scan_period": 6.5,  "psr_prob": 0.87, "ssr_prob": 0.80},
            {"id": 4, "x": -100_000, "y": -150_000, "max_range": 410_000, "scan_period": 9.0,  "psr_prob": 0.85, "ssr_prob": 0.82},
        ]
    },
    "sweden": {
        "origin_lat": 59.6519, # Arlanda
        "origin_lon": 17.9186,
        "radar_sites": [
            {"id": 0, "x": 0.0,      "y": 0.0,       "max_range": 400_000, "scan_period": 7.0,  "psr_prob": 0.92, "ssr_prob": 0.88}, # Arlanda
            {"id": 1, "x": -300_000, "y": -350_000,  "max_range": 400_000, "scan_period": 5.0,  "psr_prob": 0.90, "ssr_prob": 0.85}, # Landvetter area
            {"id": 2, "x": 100_000,  "y": -250_000,  "max_range": 400_000, "scan_period": 8.0,  "psr_prob": 0.88, "ssr_prob": 0.83}, # Gotland area
            {"id": 3, "x": -50_000,  "y": 250_000,   "max_range": 400_000, "scan_period": 6.0,  "psr_prob": 0.87, "ssr_prob": 0.80}, # North of Arlanda
            {"id": 4, "x": -250_000, "y": 100_000,   "max_range": 350_000, "scan_period": 9.0,  "psr_prob": 0.85, "ssr_prob": 0.82}, # West
        ]
    }
}

def get_config(region_name):
    region_name = region_name.lower()
    if region_name not in REGIONS:
        region_name = "uae"
    return REGIONS[region_name], region_name

# ── noise parameters ─────────────────────────────────────────────────────────
POS_NOISE_STD_M  = 150.0   # 1-sigma position error (metres)
VEL_NOISE_STD    = 3.0     # 1-sigma velocity error (m/s)
ALT_NOISE_STD_M  = 200.0   # 1-sigma altitude error (metres)
AMP_MEAN_DBZ     = 55.0    # mean PSR amplitude
AMP_STD_DBZ      = 15.0    # std-dev of PSR amplitude

# ── altitude model ───────────────────────────────────────────────────────────
# Speed magnitude → approximate cruise altitude in metres (rough mapping)
# Commercial aircraft at ~250 m/s → ~10 000 m
# Slow movers / ground vehicles → low altitude
def estimate_altitude_m(speed_ms: float) -> float:
    """Heuristic: map ground speed to a plausible flight level."""
    if speed_ms < 20:
        return rng.uniform(0, 300)          # possibly ground / helo
    elif speed_ms < 100:
        return rng.uniform(500, 4000)       # slow GA
    elif speed_ms < 180:
        return rng.uniform(3000, 8000)      # regional
    else:
        return rng.uniform(8000, 12500)     # commercial jet cruise

# ── false-alarm (clutter) parameters ─────────────────────────────────────────
FA_RATE_PER_SCAN = 2.0   # average number of false-alarm PSR/SSR per scan per radar

# ── ICAO mode-S address pool for false alarms ─────────────────────────────
def rand_hex6() -> str:
    return f"{rng.integers(0, 0xFF_FFFF):06X}"

def rand_squawk() -> str:
    return f"{rng.integers(0, 0o7777):04o}"

# ── helper: range from radar site to target ──────────────────────────────────
def slant_range(radar: dict, x: float, y: float) -> float:
    dx = x - radar["x"]
    dy = y - radar["y"]
    return math.hypot(dx, dy)

# ── load CAT-62 data ─────────────────────────────────────────────────────────

def load_cat62(path: Path) -> list[dict]:
    """
    Parse cat_62_data.txt.  Each line is a Python-dict literal (category 62
    records only) or a short category-65/63 sentinel.  We keep only
    category=62 lines that have the full set of fields.
    """
    records = []
    # mode3a optional — some Sweden extracts omit it
    required = {"track_number", "time", "lat", "lon", "x", "y", "vx", "vy"}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = ast.literal_eval(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("category") != 62:
                continue
            if not required.issubset(rec.keys()):
                continue
            records.append(rec)
    return records


def build_track_epochs(records: List[Dict]) -> Dict[float, List[Dict]]:
    by_time: Dict[float, Dict[str, Dict]] = defaultdict(dict)
    for rec in records:
        t = float(rec["time"])
        tn = str(rec["track_number"])
        if tn not in by_time[t]:
            by_time[t][tn] = rec
    return {t: list(tracks.values()) for t, tracks in sorted(by_time.items())}


# ── assign Mode-S addresses once per track_number ────────────────────────────
_modes_map: Dict[str, str] = {}

def get_mode_s(track_number, source_rec: dict = None) -> tuple:
    """Return (mode_s, source) preferring real ICAO/callsign-linked ids when present."""
    tn_str = str(track_number)
    if source_rec:
        for key in ("mode_s", "modes", "icao", "target_address"):
            if source_rec.get(key):
                return str(source_rec[key]), "cat62"
    if tn_str not in _modes_map:
        _modes_map[tn_str] = rand_hex6()
    return _modes_map[tn_str], "synthetic"


# ── generate false-alarm cluster per scan ────────────────────────────────────
def gen_false_alarms(radar: dict, scan_wall_t: float, coverage_bbox: dict) -> list[dict]:
    """Emit Poisson-distributed clutter measurements around the radar."""
    n = rng.poisson(FA_RATE_PER_SCAN)
    alarms = []
    for _ in range(n):
        # random position within radar range
        angle  = rng.uniform(0, 2 * math.pi)
        dist_m = rng.uniform(5_000, radar["max_range"] * 0.8)
        cx = radar["x"] + dist_m * math.cos(angle)
        cy = radar["y"] + dist_m * math.sin(angle)
        cz = rng.uniform(0, 1000)  # low altitude clutter
        meas_t = scan_wall_t + rng.uniform(0, radar["scan_period"] * 0.95)

        # 70% PSR, 30% SSR false alarm
        if rng.random() < 0.70:
            alarms.append({
                "t":          round(meas_t, 4),
                "radar_id":   radar["id"],
                "meas_type":  "PSR",
                "x":          round(cx, 2),
                "y":          round(cy, 2),
                "z":          round(cz, 2),
                "vx":         round(float(rng.normal(0, 5)), 3),
                "vy":         round(float(rng.normal(0, 5)), 3),
                "amplitude":  round(float(rng.normal(25, 10)), 2),
                "track_id":   -1,
                "source_lat": None,
                "source_lon": None,
            })
        else:
            alarms.append({
                "t":          round(meas_t, 4),
                "radar_id":   radar["id"],
                "meas_type":  "SSR",
                "x":          round(cx, 2),
                "y":          round(cy, 2),
                "z":          round(cz, 2),
                "mode3a":     rand_squawk(),
                "mode_s":     rand_hex6(),
                "track_id":   -1,
                "source_lat": None,
                "source_lon": None,
            })
    return alarms


# ── main generation loop ──────────────────────────────────────────────────────

def generate(
    input_path: Path,
    output_path: Path,
    radar_sites: List[Dict],
    region: str = "uae",
    max_duration: float = None,
    dataset_id: str = None,
) -> None:
    # ── phase offsets so each radar starts its scan at a different angle ──────────
    radar_phase_offsets = {r["id"]: rng.uniform(0, r["scan_period"]) for r in radar_sites}

    print(f"Loading CAT-62 data from {input_path} …")
    records = load_cat62(input_path)
    print(f"  {len(records):,} category-62 records loaded.")

    if not records:
        print("ERROR: no usable CAT-62 records found.")
        sys.exit(1)

    # Region integrity: refuse to tag as sweden if traffic is clearly UAE (and vice versa)
    lats = [float(r["lat"]) for r in records if r.get("lat") is not None]
    if lats:
        med_lat = float(np.median(lats))
        if region == "sweden" and med_lat < 50.0:
            print(
                f"ERROR: region=sweden but CAT-62 median lat={med_lat:.2f} looks like UAE/Gulf. "
                f"Use a Sweden CAT-62 source (e.g. cat_62_sweden_mini.txt)."
            )
            sys.exit(2)
        if region == "uae" and med_lat > 50.0:
            print(
                f"ERROR: region=uae but CAT-62 median lat={med_lat:.2f} looks like Scandinavia. "
                f"Use --region sweden."
            )
            sys.exit(2)
        print(f"  Region check OK: median lat={med_lat:.2f} for region={region}")

    epochs = build_track_epochs(records)
    time_keys = sorted(epochs.keys())
    print(f"  {len(time_keys):,} unique time epochs, "
          f"{sum(len(v) for v in epochs.values()):,} track-epoch pairs.")

    if not time_keys:
        print("ERROR: no usable CAT-62 records found.")
        sys.exit(1)

    # Normalise: wall-clock starts at 0.0
    t0_cat62 = time_keys[0]
    max_cat62_t = time_keys[-1] - t0_cat62
    if max_duration is not None:
        print(f"  Capping wall-clock duration to {max_duration:.1f}s (source span {max_cat62_t:.1f}s)")

    # Coverage bounding box (used for false-alarm placement)
    all_x = [r["x"] for recs in epochs.values() for r in recs]
    all_y = [r["y"] for recs in epochs.values() for r in recs]
    bbox = {
        "xmin": min(all_x), "xmax": max(all_x),
        "ymin": min(all_y), "ymax": max(all_y),
    }
    print(f"  Coverage bbox  x=[{bbox['xmin']:.0f}, {bbox['xmax']:.0f}]  "
          f"y=[{bbox['ymin']:.0f}, {bbox['ymax']:.0f}] metres")

    # Pre-compute altitude per (time, track_number) once
    alt_cache: dict = {}

    def get_alt(track_rec: dict):
        """Return (altitude_m, source) where source is 'cat62' or 'estimated'."""
        key = (track_rec["time"], str(track_rec["track_number"]))
        if key not in alt_cache:
            if track_rec.get("z") is not None:
                try:
                    z_val = float(track_rec["z"])
                    if z_val > 0:
                        alt_cache[key] = (z_val, "cat62")
                        return alt_cache[key]
                except (TypeError, ValueError):
                    pass
            speed = math.hypot(float(track_rec.get("vx", 0) or 0), float(track_rec.get("vy", 0) or 0))
            base = estimate_altitude_m(speed)
            alt_cache[key] = (base + float(rng.normal(0, ALT_NOISE_STD_M)), "estimated")
        return alt_cache[key]

    # Build scan schedule for each radar (next_scan_time indexed by radar_id)
    next_scan: dict[int, float] = {}
    last_fa_scan: dict[int, float] = {}
    for r in radar_sites:
        next_scan[r["id"]] = radar_phase_offsets[r["id"]]
        last_fa_scan[r["id"]] = -1e18

    ds_id = dataset_id or f"stream_{region}_v1"
    all_measurements: list[dict] = []
    total_meas = 0

    for cat62_t in time_keys:
        wall_t = cat62_t - t0_cat62
        if max_duration is not None and wall_t > max_duration:
            break
        track_list = epochs[cat62_t]

        for radar in radar_sites:
            rid = radar["id"]

            while next_scan[rid] < wall_t:
                next_scan[rid] += radar["scan_period"]

            scan_t = next_scan[rid]
            if max_duration is not None and scan_t > max_duration + radar["scan_period"]:
                continue

            for trk in track_list:
                tx0 = float(trk["x"])
                ty0 = float(trk["y"])
                gvx = float(trk.get("vx", 0) or 0)
                gvy = float(trk.get("vy", 0) or 0)
                gvz = float(trk.get("vz", 0) or 0)

                r_range = slant_range(radar, tx0, ty0)
                if r_range > radar["max_range"]:
                    continue

                tz, z_src = get_alt(trk)

                bearing_frac = rng.uniform(0.0, 1.0)
                meas_t = scan_t + bearing_frac * radar["scan_period"] * 0.95
                if max_duration is not None and meas_t > max_duration:
                    continue

                # Motion compensation to measurement time
                dt_sim = meas_t - wall_t
                tx = tx0 + gvx * dt_sim
                ty = ty0 + gvy * dt_sim
                tz_t = tz + gvz * dt_sim

                try:
                    tid = int(trk["track_number"])
                except (TypeError, ValueError):
                    tid = int(float(trk["track_number"]))

                # PSR
                if rng.random() < radar["psr_prob"]:
                    nx = float(rng.normal(0, POS_NOISE_STD_M))
                    ny = float(rng.normal(0, POS_NOISE_STD_M))
                    nz = float(rng.normal(0, ALT_NOISE_STD_M if z_src == "estimated" else ALT_NOISE_STD_M * 0.5))
                    nvx = float(rng.normal(0, VEL_NOISE_STD))
                    nvy = float(rng.normal(0, VEL_NOISE_STD))
                    amp = max(5.0, float(rng.normal(AMP_MEAN_DBZ, AMP_STD_DBZ)))
                    all_measurements.append({
                        "t": round(meas_t, 4),
                        "sensor_id": rid,
                        "radar_id": rid,
                        "meas_type": "PSR",
                        "type": "PSR",
                        "x": round(tx + nx, 2),
                        "y": round(ty + ny, 2),
                        "z": round(max(0.0, tz_t + nz), 2),
                        "vx": round(gvx + nvx, 3),
                        "vy": round(gvy + nvy, 3),
                        "gt_x": round(tx, 2),
                        "gt_y": round(ty, 2),
                        "gt_z": round(max(0.0, tz_t), 2),
                        "gt_vx": round(gvx, 3),
                        "gt_vy": round(gvy, 3),
                        "gt_vz": round(gvz, 3),
                        "amplitude": round(amp, 2),
                        "track_id": tid,
                        "is_clutter": False,
                        "source_lat": trk.get("lat"),
                        "source_lon": trk.get("lon"),
                        "region": region,
                        "dataset_id": ds_id,
                        "schema_version": 1,
                        "gt_z_source": z_src,
                    })
                    total_meas += 1

                # SSR
                if rng.random() < radar["ssr_prob"]:
                    nx2 = float(rng.normal(0, POS_NOISE_STD_M * 0.5))
                    ny2 = float(rng.normal(0, POS_NOISE_STD_M * 0.5))
                    nz2 = float(rng.normal(0, ALT_NOISE_STD_M * 0.3))
                    ssr_t = meas_t + rng.uniform(0.001, 0.020)
                    mode_s, ms_src = get_mode_s(trk["track_number"], trk)
                    m3a = trk.get("mode3a")
                    all_measurements.append({
                        "t": round(ssr_t, 4),
                        "sensor_id": rid,
                        "radar_id": rid,
                        "meas_type": "SSR",
                        "type": "SSR",
                        "x": round(tx + nx2, 2),
                        "y": round(ty + ny2, 2),
                        "z": round(max(0.0, tz_t + nz2), 2),
                        "gt_x": round(tx, 2),
                        "gt_y": round(ty, 2),
                        "gt_z": round(max(0.0, tz_t), 2),
                        "gt_vx": round(gvx, 3),
                        "gt_vy": round(gvy, 3),
                        "gt_vz": round(gvz, 3),
                        "mode_3a": str(m3a) if m3a is not None else None,
                        "mode3a": str(m3a) if m3a is not None else None,
                        "mode_s": mode_s,
                        "mode_s_source": ms_src,
                        "track_id": tid,
                        "is_clutter": False,
                        "source_lat": trk.get("lat"),
                        "source_lon": trk.get("lon"),
                        "region": region,
                        "dataset_id": ds_id,
                        "schema_version": 1,
                        "gt_z_source": z_src,
                    })
                    total_meas += 1

            # False alarms: once per radar scan, not once per CAT-62 epoch
            if scan_t - last_fa_scan[rid] >= radar["scan_period"] * 0.5:
                fa = gen_false_alarms(radar, scan_t, bbox)
                for a in fa:
                    a["sensor_id"] = rid
                    a["is_clutter"] = True
                    a["track_id"] = -1
                    a["region"] = region
                    a["dataset_id"] = ds_id
                    a["schema_version"] = 1
                    a["type"] = a.get("meas_type")
                    if a.get("mode3a") is not None:
                        a["mode_3a"] = a["mode3a"]
                all_measurements.extend(fa)
                total_meas += len(fa)
                last_fa_scan[rid] = scan_t

    print(f"  Generated {total_meas:,} total measurements before sorting.")
    all_measurements.sort(key=lambda m: m["t"])

    print(f"Writing to {output_path} …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for meas in all_measurements:
            fh.write(json.dumps(meas, ensure_ascii=False) + "\n")

    file_size_mb = output_path.stat().st_size / 1_048_576
    t_end = all_measurements[-1]["t"] if all_measurements else 0.0
    print(f"Done.  {len(all_measurements):,} measurements written "
          f"({file_size_mb:.1f} MB).")
    print(f"Time span: t=0.000 → t={t_end:.3f} s  "
          f"(wall clock ≈ {t_end/60:.1f} min)")

    psr_count = sum(1 for m in all_measurements if m.get("meas_type") == "PSR")
    ssr_count = sum(1 for m in all_measurements if m.get("meas_type") == "SSR")
    real_count = sum(1 for m in all_measurements if m.get("track_id", -1) != -1)
    fa_count = sum(1 for m in all_measurements if m.get("track_id", -1) == -1)
    print(f"\nSummary:")
    print(f"  PSR measurements : {psr_count:>10,}")
    print(f"  SSR measurements : {ssr_count:>10,}")
    print(f"  True-target meas : {real_count:>10,}")
    print(f"  False alarms     : {fa_count:>10,}")
    unique_tracks = len({m["track_id"] for m in all_measurements if m.get("track_id", -1) != -1})
    print(f"  Unique track IDs : {unique_tracks:>10,}")


if __name__ == "__main__":
    args = get_args()

    CONFIG, ACTIVE_REGION = get_config(args.region)
    RADAR_SITES = CONFIG["radar_sites"]
    print(f"Simulation Region: {ACTIVE_REGION.upper()} "
          f"(Origin: {CONFIG['origin_lat']}, {CONFIG['origin_lon']})")

    generate(
        Path(args.input),
        Path(args.output),
        RADAR_SITES,
        region=ACTIVE_REGION,
        max_duration=args.max_duration,
        dataset_id=args.dataset_id,
    )

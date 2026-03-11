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
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
INPUT_FILE = DATA_DIR / "cat_62_data.txt"
OUTPUT_FILE = DATA_DIR / "stream_radar_001.jsonl"

# ── reproducibility ─────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
random.seed(RNG_SEED)

# ── radar site definitions (lon, lat approximate) ───────────────────────────
#  5 notional radar sites placed to give overlapping coverage of UAE airspace.
#  Positions expressed in the same Cartesian frame as the CAT-62 x/y data
#  (metres east/north from a reference point near Abu Dhabi centre).
RADAR_SITES = [
    # id, x_site, y_site,  max_range_m,  psr_rate_hz, ssr_rate_hz, scan_period_s
    {"id": 0, "x": 0.0,      "y": 0.0,      "max_range": 450_000, "scan_period": 7.0,  "psr_prob": 0.92, "ssr_prob": 0.88},
    {"id": 1, "x": 150_000,  "y": 100_000,  "max_range": 400_000, "scan_period": 5.5,  "psr_prob": 0.90, "ssr_prob": 0.85},
    {"id": 2, "x": -200_000, "y": 200_000,  "max_range": 420_000, "scan_period": 8.0,  "psr_prob": 0.88, "ssr_prob": 0.83},
    {"id": 3, "x": 300_000,  "y": -50_000,  "max_range": 380_000, "scan_period": 6.5,  "psr_prob": 0.87, "ssr_prob": 0.80},
    {"id": 4, "x": -100_000, "y": -150_000, "max_range": 410_000, "scan_period": 9.0,  "psr_prob": 0.85, "ssr_prob": 0.82},
]

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
    required = {"track_number", "time", "lat", "lon", "x", "y", "vx", "vy", "mode3a"}
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


def build_track_epochs(records: list[dict]) -> dict:
    """
    Group records by (time, track_number) → unique aircraft epoch.
    Returns dict[time_float] → list of track dicts for that epoch.
    The CAT-62 feed repeats each track across multiple SAC/SIC entries;
    we deduplicate by track_number within the same time step.
    """
    by_time: dict[float, dict[int, dict]] = defaultdict(dict)
    for rec in records:
        t = rec["time"]
        tn = int(rec["track_number"])
        if tn not in by_time[t]:
            by_time[t][tn] = rec
    return {t: list(tracks.values()) for t, tracks in sorted(by_time.items())}


# ── assign Mode-S addresses once per track_number ────────────────────────────
_modes_map: dict[int, str] = {}

def get_mode_s(track_number: int) -> str:
    if track_number not in _modes_map:
        _modes_map[track_number] = rand_hex6()
    return _modes_map[track_number]


# ── phase offsets so each radar starts its scan at a different angle ──────────
RADAR_PHASE_OFFSETS = {r["id"]: rng.uniform(0, r["scan_period"]) for r in RADAR_SITES}


def next_scan_time(radar: dict, current_wall_t: float) -> float:
    """Return the wall-clock time of the next scan for this radar."""
    period = radar["scan_period"]
    phase  = RADAR_PHASE_OFFSETS[radar["id"]]
    # How many full periods since zero?
    elapsed = current_wall_t + phase
    return current_wall_t + (period - (elapsed % period))


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

def generate(input_path: Path, output_path: Path) -> None:
    print(f"Loading CAT-62 data from {input_path} …")
    records = load_cat62(input_path)
    print(f"  {len(records):,} category-62 records loaded.")

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
    alt_cache: dict[tuple, float] = {}

    def get_alt(track_rec: dict) -> float:
        key = (track_rec["time"], int(track_rec["track_number"]))
        if key not in alt_cache:
            speed = math.hypot(track_rec["vx"], track_rec["vy"])
            base  = estimate_altitude_m(speed)
            alt_cache[key] = base + float(rng.normal(0, ALT_NOISE_STD_M))
        return alt_cache[key]

    # Build scan schedule for each radar (next_scan_time indexed by radar_id)
    next_scan: dict[int, float] = {}
    for r in RADAR_SITES:
        next_scan[r["id"]] = RADAR_PHASE_OFFSETS[r["id"]]  # first scan

    # We'll collect all measurements and sort by t at the end
    all_measurements: list[dict] = []
    total_meas = 0

    # Iterate over cat62 epochs in time order
    # Wall-clock time of each CAT-62 epoch
    for cat62_t in time_keys:
        wall_t = cat62_t - t0_cat62   # seconds from recording start
        track_list = epochs[cat62_t]

        for radar in RADAR_SITES:
            rid = radar["id"]

            # Advance scan schedule past the current wall_t
            while next_scan[rid] < wall_t:
                next_scan[rid] += radar["scan_period"]

            scan_t = next_scan[rid]  # The scan timestamp this radar "sees" these AC

            for trk in track_list:
                tx = trk["x"]
                ty = trk["y"]

                r_range = slant_range(radar, tx, ty)
                if r_range > radar["max_range"]:
                    continue  # out of range for this radar

                tz = get_alt(trk)

                # Timing: radar sweeps past the target within the scan period
                # Stagger the exact detection time within the current scan
                bearing_frac = rng.uniform(0.0, 1.0)
                meas_t = scan_t + bearing_frac * radar["scan_period"] * 0.95

                # Add Gaussian position noise
                nx = float(rng.normal(0, POS_NOISE_STD_M))
                ny = float(rng.normal(0, POS_NOISE_STD_M))
                nz = float(rng.normal(0, ALT_NOISE_STD_M))

                # PSR measurement
                if rng.random() < radar["psr_prob"]:
                    nvx = float(rng.normal(0, VEL_NOISE_STD))
                    nvy = float(rng.normal(0, VEL_NOISE_STD))
                    amp  = float(rng.normal(AMP_MEAN_DBZ, AMP_STD_DBZ))
                    amp  = max(5.0, amp)  # floor

                    meas = {
                        "t":          round(meas_t, 4),
                        "radar_id":   rid,
                        "meas_type":  "PSR",
                        "x":          round(tx + nx, 2),
                        "y":          round(ty + ny, 2),
                        "z":          round(max(0.0, tz + nz), 2),
                        "vx":         round(trk["vx"] + nvx, 3),
                        "vy":         round(trk["vy"] + nvy, 3),
                        "amplitude":  round(amp, 2),
                        "track_id":   int(trk["track_number"]),
                        "source_lat": trk["lat"],
                        "source_lon": trk["lon"],
                    }
                    all_measurements.append(meas)
                    total_meas += 1

                # SSR measurement (slightly different timing — transponder reply
                # arrives a few milliseconds after the PSR sweep)
                if rng.random() < radar["ssr_prob"]:
                    nx2 = float(rng.normal(0, POS_NOISE_STD_M * 0.5))  # SSR more accurate
                    ny2 = float(rng.normal(0, POS_NOISE_STD_M * 0.5))
                    nz2 = float(rng.normal(0, ALT_NOISE_STD_M * 0.3))
                    ssr_t = meas_t + rng.uniform(0.001, 0.020)  # ~ms later

                    meas = {
                        "t":          round(ssr_t, 4),
                        "radar_id":   rid,
                        "meas_type":  "SSR",
                        "x":          round(tx + nx2, 2),
                        "y":          round(ty + ny2, 2),
                        "z":          round(max(0.0, tz + nz2), 2),
                        "mode3a":     trk["mode3a"],
                        "mode_s":     get_mode_s(int(trk["track_number"])),
                        "track_id":   int(trk["track_number"]),
                        "source_lat": trk["lat"],
                        "source_lon": trk["lon"],
                    }
                    all_measurements.append(meas)
                    total_meas += 1

            # Inject false alarms for this radar at this scan time
            fa = gen_false_alarms(radar, scan_t, bbox)
            all_measurements.extend(fa)
            total_meas += len(fa)

    print(f"  Generated {total_meas:,} total measurements before sorting.")

    # Sort chronologically
    all_measurements.sort(key=lambda m: m["t"])

    # Write output
    print(f"Writing to {output_path} …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for meas in all_measurements:
            fh.write(json.dumps(meas, ensure_ascii=False) + "\n")

    file_size_mb = output_path.stat().st_size / 1_048_576
    print(f"Done.  {len(all_measurements):,} measurements written "
          f"({file_size_mb:.1f} MB).")
    print(f"Time span: t=0.000 → t={all_measurements[-1]['t']:.3f} s  "
          f"(wall clock ≈ {all_measurements[-1]['t']/60:.1f} min)")

    # Print summary statistics
    psr_count = sum(1 for m in all_measurements if m["meas_type"] == "PSR")
    ssr_count = sum(1 for m in all_measurements if m["meas_type"] == "SSR")
    real_count = sum(1 for m in all_measurements if m["track_id"] != -1)
    fa_count   = sum(1 for m in all_measurements if m["track_id"] == -1)
    print(f"\nSummary:")
    print(f"  PSR measurements : {psr_count:>10,}")
    print(f"  SSR measurements : {ssr_count:>10,}")
    print(f"  True-target meas : {real_count:>10,}")
    print(f"  False alarms     : {fa_count:>10,}")
    unique_tracks = len({m["track_id"] for m in all_measurements if m["track_id"] != -1})
    print(f"  Unique track IDs : {unique_tracks:>10,}")
    print(f"\nSample records (first 3 PSR, first 3 SSR):")
    shown = {"PSR": 0, "SSR": 0}
    for m in all_measurements:
        mt = m["meas_type"]
        if shown[mt] < 3:
            print(f"  {json.dumps(m)}")
            shown[mt] += 1
        if shown["PSR"] >= 3 and shown["SSR"] >= 3:
            break


if __name__ == "__main__":
    generate(INPUT_FILE, OUTPUT_FILE)

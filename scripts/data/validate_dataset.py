"""
QA gates for canonical stream / batch datasets.

Usage:
  uv run python scripts/data/validate_dataset.py data/canonical/stream_uae_2min.jsonl
  uv run python scripts/data/validate_dataset.py data/canonical --report artifacts/data_qa_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]


def load_rows(path: Path, max_rows: int = 50000) -> Tuple[str, List[dict]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= max_rows:
                break
    if rows and isinstance(rows[0].get("measurements"), list):
        return "batch", rows
    return "stream", rows


def flatten_stream(fmt: str, rows: List[dict]) -> List[dict]:
    if fmt == "stream":
        return rows
    out = []
    for fr in rows:
        for m in fr.get("measurements") or []:
            if isinstance(m, dict):
                out.append(m)
    return out


def validate_stream(path: Path, rows: List[dict]) -> Dict[str, Any]:
    issues = []
    warnings = []
    n = len(rows)
    if n == 0:
        return {"path": str(path), "ok": False, "issues": ["empty file"], "warnings": []}

    # Required fields
    required = ["t", "x", "y", "z", "track_id"]
    for k in required:
        miss = sum(1 for r in rows if k not in r or r[k] is None)
        if miss / n > 0.01:
            issues.append(f"missing required field '{k}' on {miss}/{n} rows")

    # Sensor cardinality
    sensors = set()
    for r in rows:
        sid = r.get("sensor_id", r.get("radar_id"))
        if sid is not None:
            sensors.add(int(sid))
    if len(sensors) < 2:
        issues.append(f"sensor cardinality {len(sensors)} < 2 (expected multi-radar)")

    # Time sorted
    ts = [float(r["t"]) for r in rows if r.get("t") is not None]
    if ts != sorted(ts):
        # allow equal times
        if any(ts[i] > ts[i + 1] for i in range(len(ts) - 1)):
            issues.append("timestamps not non-decreasing")

    # Float track ids
    float_tid = sum(
        1 for r in rows
        if isinstance(r.get("track_id"), float) and not float(r["track_id"]).is_integer()
    )
    if float_tid:
        issues.append(f"{float_tid} non-integer float track_ids")

    # Clutter ratio
    clutter = sum(1 for r in rows if int(r.get("track_id", -1)) == -1 or r.get("is_clutter"))
    cr = clutter / n
    if cr < 0.01 or cr > 0.5:
        warnings.append(f"clutter ratio {cr:.3f} outside preferred [0.01, 0.50]")

    # Region vs lat
    region = next((r.get("region") for r in rows if r.get("region")), None)
    lats = [float(r["source_lat"]) for r in rows if r.get("source_lat") is not None]
    if region and lats:
        med = sorted(lats)[len(lats) // 2]
        if region == "sweden" and med < 50:
            issues.append(f"region=sweden but median lat={med:.2f} (misnamed geography)")
        if region == "uae" and med > 50:
            issues.append(f"region=uae but median lat={med:.2f}")

    # GT independence: gt should not always equal noisy x
    both = [
        r for r in rows
        if r.get("gt_x") is not None and r.get("x") is not None and int(r.get("track_id", -1)) != -1
    ]
    if both:
        equal = sum(1 for r in both if abs(float(r["gt_x"]) - float(r["x"])) < 1e-6)
        if equal / len(both) > 0.95:
            warnings.append(
                f"gt_x equals x on {equal}/{len(both)} targets — GT may not be independent noise-free truth"
            )
        else:
            # good
            pass
    else:
        warnings.append("no gt_x present on targets")

    # schema version
    with_schema = sum(1 for r in rows if r.get("schema_version") is not None)
    if with_schema < n * 0.5:
        warnings.append("schema_version missing on majority of rows (legacy file)")

    ok = len(issues) == 0
    return {
        "path": str(path).replace("\\", "/"),
        "ok": ok,
        "n_rows": n,
        "n_sensors": len(sensors),
        "clutter_ratio": round(cr, 4),
        "region_tag": region,
        "median_lat": sorted(lats)[len(lats) // 2] if lats else None,
        "issues": issues,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="File or directory of jsonl")
    ap.add_argument("--report", type=str, default=None)
    args = ap.parse_args()

    target = Path(args.path)
    files = []
    if target.is_dir():
        files = sorted(target.rglob("*.jsonl"))
        # skip episodes sub-tree bulk if validating top-level only
        files = [f for f in files if "episodes" not in f.parts or f.parent.name in ("train", "val", "test")]
        # Prefer top-level canonical only when dir is canonical
        top = sorted(target.glob("*.jsonl"))
        if top:
            files = top
    else:
        files = [target]

    results = []
    for f in files:
        fmt, rows = load_rows(f)
        flat = flatten_stream(fmt, rows)
        if fmt == "batch":
            # light batch checks
            res = {
                "path": str(f).replace("\\", "/"),
                "ok": True,
                "n_rows": len(rows),
                "format": "batch",
                "issues": [],
                "warnings": [],
            }
            if not rows:
                res["ok"] = False
                res["issues"].append("empty")
            results.append(res)
        else:
            results.append(validate_stream(f, flat))

    all_ok = all(r["ok"] for r in results)
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"[{status}] {r['path']}")
        for i in r.get("issues") or []:
            print(f"    ISSUE: {i}")
        for w in r.get("warnings") or []:
            print(f"    WARN:  {w}")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Data QA Report",
            "",
            f"Overall: **{'PASS' if all_ok else 'FAIL'}**",
            "",
            "| File | OK | Sensors | Clutter | Issues |",
            "|------|----|---------|---------|--------|",
        ]
        for r in results:
            issues = "; ".join(r.get("issues") or []) or "—"
            lines.append(
                f"| `{r['path']}` | {r['ok']} | {r.get('n_sensors', '—')} | "
                f"{r.get('clutter_ratio', '—')} | {issues} |"
            )
        lines.append("")
        lines.append("## Details")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(results, indent=2))
        lines.append("```")
        rep.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nWrote {rep}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

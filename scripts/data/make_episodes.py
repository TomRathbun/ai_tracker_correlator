"""
Slice a stream JSONL into fixed-length episodes with train/val/test manifests.

Usage:
  uv run python scripts/data/make_episodes.py \
    --input data/canonical/stream_uae_2min.jsonl \
    --out-dir data/canonical/episodes/uae_2min \
    --episode-s 60 --train 0.7 --val 0.15 --test 0.15
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_stream(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "t" in obj:
                rows.append(obj)
    rows.sort(key=lambda r: r["t"])
    return rows


def slice_episodes(rows: List[Dict], episode_s: float, min_meas: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    t0 = rows[0]["t"]
    t1 = rows[-1]["t"]
    episodes = []
    start = t0
    idx = 0
    ep_i = 0
    while start < t1:
        end = start + episode_s
        chunk = []
        while idx < len(rows) and rows[idx]["t"] < start:
            idx += 1
        j = idx
        while j < len(rows) and rows[j]["t"] < end:
            chunk.append(rows[j])
            j += 1
        if len(chunk) >= min_meas:
            tids = sorted({
                int(m["track_id"]) for m in chunk
                if m.get("track_id", -1) not in (-1, None)
            })
            episodes.append({
                "episode_id": f"ep_{ep_i:04d}",
                "t0": start,
                "t1": end,
                "n_meas": len(chunk),
                "track_ids": tids,
                "n_tracks": len(tids),
                "records": chunk,
            })
            ep_i += 1
        start = end
        idx = j
    return episodes


def assign_splits(episodes: List[Dict], train: float, val: float, seed: int):
    rng = random.Random(seed)
    order = list(range(len(episodes)))
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * train)
    n_val = max(1 if n >= 3 else 0, int(n * val))
    n_test = max(1 if n >= 2 else 0, n - n_train - n_val)
    # Rebalance if overflow
    while n_train + n_val + n_test > n and n_train > 1:
        n_train -= 1
    while n_train + n_val + n_test > n and n_val > 0:
        n_val -= 1
    splits = {}
    for k, i in enumerate(order):
        if k < n_train:
            splits[i] = "train"
        elif k < n_train + n_val:
            splits[i] = "val"
        else:
            splits[i] = "test"
    if n >= 3 and "test" not in splits.values():
        splits[order[-1]] = "test"
    if n >= 3 and "val" not in splits.values():
        # steal one from train
        for i in order:
            if splits.get(i) == "train":
                splits[i] = "val"
                break
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episode-s", type=float, default=60.0)
    p.add_argument("--min-meas", type=int, default=20)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6

    rows = load_stream(Path(args.input))
    print(f"Loaded {len(rows)} measurements from {args.input}")
    episodes = slice_episodes(rows, args.episode_s, args.min_meas)
    print(f"Created {len(episodes)} episodes of {args.episode_s}s")

    splits = assign_splits(episodes, args.train, args.val, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(args.input),
        "episode_s": args.episode_s,
        "n_episodes": len(episodes),
        "splits": {"train": [], "val": [], "test": []},
        "episodes": [],
    }

    for i, ep in enumerate(episodes):
        split = splits.get(i, "train")
        rel = f"{split}/{ep['episode_id']}.jsonl"
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in ep["records"]:
                f.write(json.dumps(r) + "\n")
        meta = {
            "episode_id": ep["episode_id"],
            "split": split,
            "path": rel.replace("\\", "/"),
            "t0": ep["t0"],
            "t1": ep["t1"],
            "n_meas": ep["n_meas"],
            "n_tracks": ep["n_tracks"],
            "track_ids": ep["track_ids"],
        }
        manifest["episodes"].append(meta)
        manifest["splits"][split].append(rel.replace("\\", "/"))

    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {man_path}")
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(manifest['splits'][s])} episodes")


if __name__ == "__main__":
    main()

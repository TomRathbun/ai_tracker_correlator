# Data QA Report

Overall: **PASS**

| File | OK | Sensors | Clutter | Issues |
|------|----|---------|---------|--------|
| `data/canonical/sim_batch_hetero.jsonl` | True | — | — | — |
| `data/canonical/stream_sweden_15min.jsonl` | True | 5 | 0.0295 | — |
| `data/canonical/stream_uae_2min.jsonl` | True | 5 | 0.0042 | — |

## Details

```json
[
  {
    "path": "data/canonical/sim_batch_hetero.jsonl",
    "ok": true,
    "n_rows": 300,
    "format": "batch",
    "issues": [],
    "warnings": []
  },
  {
    "path": "data/canonical/stream_sweden_15min.jsonl",
    "ok": true,
    "n_rows": 28379,
    "n_sensors": 5,
    "clutter_ratio": 0.0295,
    "region_tag": "sweden",
    "median_lat": 58.419901728630066,
    "issues": [],
    "warnings": []
  },
  {
    "path": "data/canonical/stream_uae_2min.jsonl",
    "ok": true,
    "n_rows": 42027,
    "n_sensors": 5,
    "clutter_ratio": 0.0042,
    "region_tag": "uae",
    "median_lat": 25.08081614971161,
    "issues": [],
    "warnings": [
      "clutter ratio 0.004 outside preferred [0.01, 0.50]"
    ]
  }
]
```
# Training Data Difficulty Report

Heuristic score (0–100): concurrency, nearest-neighbor closeness, multi-sensor overlap.

| File | Dur (min) | Tracks | Conc med/max | NN p50 (m) | Close &lt;5km | Multi-sensor | Score | Band |
|------|----------:|-------:|-------------:|-----------:|-------------:|-------------:|------:|------|
| `stream_sweden_15min.jsonl` | 10.02 | 54 | 20/31 | 32869 | 39 | 0.24 | 41.5 | medium |
| `stream_sweden_30min_holdout.jsonl` | 29.77 | 163 | 19/32 | 32398 | 121 | 0.24 | 41.3 | medium |
| `stream_sweden_30min_train.jsonl` | 30.24 | 162 | 19/32 | 32242 | 130 | 0.24 | 41.6 | medium |
| `stream_sweden_60min_test.jsonl` | 60.53 | 325 | 19/32 | 32206 | 256 | 0.24 | 41.5 | medium |
| `stream_uae_2min.jsonl` | 1.96 | 256 | 135/164 | 13651 | 449 | 0.36 | 62.7 | hard |

## Details

```json
[
  {
    "path": "data/canonical/stream_sweden_15min.jsonl",
    "n_meas": 28379,
    "duration_s": 601.4105999999999,
    "duration_min": 10.02,
    "n_tracks": 54,
    "clutter_ratio": 0.0295,
    "n_sensors": 5,
    "types": {
      "PSR": 14734,
      "SSR": 13645
    },
    "concurrent_median": 20.0,
    "concurrent_max": 31,
    "alt_m_p10": 926.5920000000002,
    "alt_m_p50": 8823.96,
    "alt_m_p90": 11582.4,
    "speed_mps_p50": 211.0584101131516,
    "track_duration_s_p50": 596.7949000000001,
    "nn_dist_m_p10": 11090.867323208558,
    "nn_dist_m_p50": 32869.163430781155,
    "nn_samples": 2219,
    "close_pairs_lt_5km": 39,
    "close_pairs_diff_alt": 32,
    "multi_sensor_frac": 0.2426,
    "difficulty_score_0_100": 41.5,
    "difficulty_band": "medium"
  },
  {
    "path": "data/canonical/stream_sweden_30min_holdout.jsonl",
    "n_meas": 80770,
    "duration_s": 1786.0465,
    "duration_min": 29.77,
    "n_tracks": 163,
    "clutter_ratio": 0.0319,
    "n_sensors": 5,
    "types": {
      "PSR": 42087,
      "SSR": 38683
    },
    "concurrent_median": 19.0,
    "concurrent_max": 32,
    "alt_m_p10": 853.44,
    "alt_m_p50": 9136.38,
    "alt_m_p90": 11582.4,
    "speed_mps_p50": 211.6992756154022,
    "track_duration_s_p50": 568.4684,
    "nn_dist_m_p10": 10814.984775305973,
    "nn_dist_m_p50": 32398.362270359383,
    "nn_samples": 6470,
    "close_pairs_lt_5km": 121,
    "close_pairs_diff_alt": 100,
    "multi_sensor_frac": 0.2367,
    "difficulty_score_0_100": 41.3,
    "difficulty_band": "medium"
  },
  {
    "path": "data/canonical/stream_sweden_30min_train.jsonl",
    "n_meas": 83637,
    "duration_s": 1814.2845,
    "duration_min": 30.24,
    "n_tracks": 162,
    "clutter_ratio": 0.0311,
    "n_sensors": 5,
    "types": {
      "PSR": 43356,
      "SSR": 40281
    },
    "concurrent_median": 19.0,
    "concurrent_max": 32,
    "alt_m_p10": 853.44,
    "alt_m_p50": 9022.08,
    "alt_m_p90": 11582.4,
    "speed_mps_p50": 211.06717340983488,
    "track_duration_s_p50": 596.32005,
    "nn_dist_m_p10": 10809.102644792823,
    "nn_dist_m_p50": 32241.79191940539,
    "nn_samples": 6634,
    "close_pairs_lt_5km": 130,
    "close_pairs_diff_alt": 107,
    "multi_sensor_frac": 0.2404,
    "difficulty_score_0_100": 41.6,
    "difficulty_band": "medium"
  },
  {
    "path": "data/canonical/stream_sweden_60min_test.jsonl",
    "n_meas": 165979,
    "duration_s": 3631.6254,
    "duration_min": 60.53,
    "n_tracks": 325,
    "clutter_ratio": 0.0315,
    "n_sensors": 5,
    "types": {
      "PSR": 86249,
      "SSR": 79730
    },
    "concurrent_median": 19.0,
    "concurrent_max": 32,
    "alt_m_p10": 853.44,
    "alt_m_p50": 9083.04,
    "alt_m_p90": 11582.4,
    "speed_mps_p50": 212.11846417237018,
    "track_duration_s_p50": 594.8179,
    "nn_dist_m_p10": 10800.523069241191,
    "nn_dist_m_p50": 32205.75751776846,
    "nn_samples": 13225,
    "close_pairs_lt_5km": 256,
    "close_pairs_diff_alt": 214,
    "multi_sensor_frac": 0.2381,
    "difficulty_score_0_100": 41.5,
    "difficulty_band": "medium"
  },
  {
    "path": "data/canonical/stream_uae_2min.jsonl",
    "n_meas": 42027,
    "duration_s": 117.47279999999999,
    "duration_min": 1.96,
    "n_tracks": 256,
    "clutter_ratio": 0.0042,
    "n_sensors": 5,
    "types": {
      "PSR": 21638,
      "SSR": 20389
    },
    "concurrent_median": 135.0,
    "concurrent_max": 164,
    "alt_m_p10": 1551.245,
    "alt_m_p50": 5819.424999999999,
    "alt_m_p90": 10572.845000000001,
    "speed_mps_p50": 147.1930389689765,
    "track_duration_s_p50": 112.35585,
    "nn_dist_m_p10": 3195.7936054914326,
    "nn_dist_m_p50": 13651.301747782138,
    "nn_samples": 2436,
    "close_pairs_lt_5km": 449,
    "close_pairs_diff_alt": 344,
    "multi_sensor_frac": 0.3569,
    "difficulty_score_0_100": 62.7,
    "difficulty_band": "hard"
  }
]
```

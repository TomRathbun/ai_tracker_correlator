# Canonical Training Data Schema (v1)

## Stream record (one JSON object per line)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `t` | float | yes | Seconds from stream start |
| `sensor_id` | int | yes | Radar site id (also written as `radar_id`) |
| `meas_type` | str | yes | `PSR` or `SSR` (also written as `type`) |
| `x,y,z` | float | yes | Cartesian metres |
| `vx,vy,vz` | float \| null | no | Present for PSR when available; omit for SSR |
| `amplitude` | float \| null | no | PSR only |
| `mode_3a` | str \| null | no | Also `mode3a` for legacy |
| `mode_s` | str \| null | no | |
| `track_id` | int | yes | `-1` = clutter |
| `is_clutter` | bool | yes | |
| `gt_x,gt_y,gt_z` | float | targets | Noise-free truth at `t` |
| `gt_vx,gt_vy,gt_vz` | float | targets | Truth velocity |
| `source_lat,source_lon` | float \| null | no | Original geo |
| `region` | str | yes | `uae` \| `sweden` \| `synthetic` |
| `dataset_id` | str | yes | e.g. `stream_uae_v1` |
| `schema_version` | int | yes | `1` |

## Batch frame (one JSON object per line)

```json
{
  "timestamp": 0.0,
  "measurements": [ /* same fields as stream, t may equal timestamp */ ],
  "gt_tracks": [{ "id": 0, "t": 0.0, "x": 0, "y": 0, "z": 0, "vx": 0, "vy": 0, "vz": 0 }],
  "schema_version": 1
}
```

## Rules

1. Prefer `null` / omitted over fake zeros for missing velocity or identity.
2. Consumers should call `src.data_schema.normalize_measurement` at load time.
3. Train only on `data/canonical/*` going forward; legacy files stay in `data/` for reference.
4. Files named Sweden must have Scandinavian geography (validated by `validate_dataset.py`).

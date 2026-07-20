# Training Data Generation and Validation

**Capstone technical chapter — AI Tracker Correlator**  
**Version:** 1.0 · **Date:** 2026-07-10

---

## 1. Purpose and claims

This chapter defines how multi-radar **training and evaluation streams** are built from ASTERIX CAT-062 traffic, what is real versus synthetic, how labels relate to model inputs, and how difficult the scenarios are. It underpins hybrid tracker results (pairwise association MLPs + asynchronous Kalman).

**Primary claims**

1. CAT-062 provides a **real traffic scenario**; multi-radar plots are produced by a **lightweight observation model**, not a full radar physics simulation.
2. Learned MLPs never take **`track_id` as a feature**; it is used only as a **supervision / metrics label**.
3. Scenario **duration alone is insufficient**; concurrent traffic and nearest-neighbor geometry matter more for association difficulty.
4. Recommended corpus: **dense multi-target 30 min train** + **geometry-augmented 30 min holdout** (from tiled mini scenario), plus short dense UAE stream for stress density.

---

## 2. Data sources

| Source | Geography | Role | Notes |
|--------|-----------|------|--------|
| `data/cat_62_sweden_mini.txt` | Sweden | **Dense multi-target core** | ~10 min, ~55 tracks, up to ~8 concurrent |
| `data/cat_62_sweden_subset.txt` | Sweden | Long sparse extract | Many hours, but mostly **1–2 concurrent** tracks — *not* used alone for multi-target training |
| `data/cat_62_data.txt` | UAE / Gulf | High-density short span | ~2 min, hundreds of tracks — hard density, different region |
| `data/canonical/sim_batch_hetero.jsonl` | Synthetic | Pairwise / clutter batch train | 300 frames, 20 tracks, controlled PSR/SSR mix |

**Important finding:** Naïvely cutting the first 30–60 minutes of `cat_62_sweden_subset.txt` yields almost **single-track** streams (easy, not representative of multi-radar fusion stress). Longer is not automatically harder.

---

## 3. Observation model (CAT-062 → multi-radar measurements)

Implemented in `src/generate_streaming_training_data.py`.

### 3.1 Conceptual pipeline

```
CAT-062 track states (time, x, y, z?, vx, vy, identity…)
        │
        ▼
Synthetic radar sites (range, scan period, Pd_psr, Pd_ssr)
        │
        ▼
Per epoch × per radar:
  range gate → Bernoulli detection → time stagger
  CV project to measurement time → Gaussian noise
  PSR: velocity + amplitude | SSR: Mode-3A / Mode-S
        │
        ▼
Poisson false alarms (clutter, track_id = -1)
        │
        ▼
Sort by t → JSONL stream (schema v1)
```

### 3.2 Algorithms (explicit)

| Step | Model |
|------|--------|
| Visibility | Planar range to radar site \(R < R_{\max}\) |
| Detection | Independent Bernoulli \(P_d\) for PSR and SSR |
| Measurement time | Radar phase + \(U(0, 0.95\,T_{\mathrm{scan}})\) within scan |
| Motion to \(t_{\mathrm{meas}}\) | Constant velocity: \(\mathbf{p}(t)=\mathbf{p}_0+\mathbf{v}\,\Delta t\) |
| Position noise | i.i.d. Gaussian, \(\sigma_{\mathrm{pos}} \approx 150\,\mathrm{m}\) (SSR smaller) |
| Velocity noise (PSR) | i.i.d. Gaussian, \(\sigma_v \approx 3\,\mathrm{m/s}\) |
| Amplitude (PSR) | Gaussian around fixed mean |
| Altitude | Prefer CAT-062 \(z\); else speed→cruise heuristic + noise |
| Mode-S | Prefer source field; else **synthetic stable hex per track** |
| Clutter | Poisson FA near radar; not terrain/weather-correlated |

### 3.3 What this is *not*

- Not polar CAT-048 with beam pattern / SNR vs range  
- Not the external **DIS + multi-radar model** pipeline (higher fidelity, out of repo)  
- Not registered multi-radar bias / multipath  

**Use interpretation:** suitable for **association / fusion ML feasibility** and hybrid track-life studies; not for claiming training on raw operational multi-radar plots.

### 3.4 Dense 30 / 60 minute construction

Because long Sweden extracts are temporally sparse in concurrency:

1. Take dense mini (~10 min, multi-target).  
2. **Tile** \(N\) copies (`scripts/data/build_dense_cat62.py`) with:
   - time shift \((T_{\mathrm{tile}}+\mathrm{gap})\)
   - spatial offset per tile  
   - remapped `track_number`  
3. Run the observation model to emit:
   - `stream_sweden_30min_train.jsonl` (3 tiles, ~30 min, ~162 tracks)  
   - `stream_sweden_60min_test.jsonl` (6 tiles, ~60 min, ~325 tracks)  
4. **Holdout:** second half of the 60 min stream → `stream_sweden_30min_holdout.jsonl`  
   - **No track_id overlap** with the first-half train tiles  

**Honesty:** holdout reuses the same underlying traffic *pattern* with new IDs/offsets (geometry-augmented holdout), not a new operational day. Cross-region eval (UAE stream) remains available for distribution shift.

---

## 4. Canonical schema and train/eval contract

See `data/canonical/schema.md`.

**Rules**

- Dual field names for compatibility (`sensor_id`/`radar_id`, `meas_type`/`type`, `mode_3a`/`mode3a`).  
- Prefer **omitted / null** optional velocity over fake zeros for SSR.  
- Always write `gt_*` kinematics for targets when generating.  
- `schema_version`, `region`, `dataset_id` on stream records.  
- Consumers normalize via `src/data_schema.py` at load time.

**Recommended roles**

| File | Role |
|------|------|
| `sim_batch_hetero.jsonl` | Train pairwise + clutter MLPs |
| `stream_sweden_30min_train.jsonl` | Streaming train / episode train |
| `stream_sweden_30min_holdout.jsonl` | Primary hybrid **holdout eval** |
| `stream_sweden_60min_test.jsonl` | Full-hour stress / long-run eval |
| `stream_sweden_15min.jsonl` | Legacy dense ~10 min (untiled mini) |
| `stream_uae_2min.jsonl` | High-density difficulty / transfer check |
| `episodes/sweden_30min_*` | 90 s clips + manifests |

---

## 5. Label leakage analysis (MLPs)

### 5.1 Pairwise association classifiers

| | Content |
|--|---------|
| **Features (PSR–PSR)** | \(\Delta p\), velocity cosine, \(\|\Delta v\|\), angular seps, \(\Delta\) amplitude |
| **Features (SSR–ANY)** | \(\Delta p\), azimuth sep, **Mode-3A match**, **Mode-S match** |
| **Label** | \(1\) iff `track_id_i == track_id_j` and both \(\neq -1\) |

**`track_id` is not an input feature.**  
Mode-3A / Mode-S matches are intentional SSR cues (operationally available). Synthetic Mode-S is **stable per track**, so the SSR MLP can learn identity consistency — realistic for SSR, weaker as a pure kinematics stress test.

### 5.2 Clutter classifier

Unary features: amplitude, velocity, normalized position, type bit — **not** `track_id`. Label: clutter if `track_id == -1` (or `is_clutter`).

### 5.3 Hybrid inference path

At eval time, association uses only features above + Kalman. `track_id` on measurements is used by **metrics** (`TrackingMetrics`) against interpolated GT, not by the hybrid updater as a feature.

### 5.4 Recommended ablations (future)

- SSR identity features ablated (kinematics-only association)  
- Mode-S dropout noise  
- Hard negatives: forced near-miss pairs at 1–5 km  

---

## 6. Difficulty analysis

Tool: `scripts/data/difficulty_report.py` → `artifacts/data_difficulty_report.md`.

**Heuristic score (0–100):** concurrency + nearest-neighbor closeness + multi-sensor fraction.

Illustrative bands (regenerate report for latest numbers):

| Stream | Dur | Tracks | Conc med | Band (typical) |
|--------|-----|--------|----------|----------------|
| UAE 2 min | ~2 m | high | very high | **hard** |
| Sweden 30 min train (tiled) | ~30 m | ~160 | ~19 | **medium** |
| Sweden 30 min holdout | ~30 m | ~160 | ~19 | **medium** |
| Sweden 15 min (mini) | ~10 m | ~54 | ~20 | **medium** |
| Naïve long subset cut | 30–60 m | 1–2 | 1 | **easy** (rejected) |

**Interpretation**

- Altitude mix exists (low to cruise).  
- **Close geometric crossings** remain limited (large median NN distance) — association is easier than a dense TMA.  
- High hybrid MOTA must be framed with this difficulty profile.

---

## 7. Validation procedures

| Layer | Tool | Checks |
|-------|------|--------|
| Schema / integrity | `scripts/data/validate_dataset.py` | fields, multi-sensor, region vs lat, clutter rate, GT presence |
| Inventory | `scripts/data/inventory_datasets.py` | misnamed geo, format |
| Visual | `scripts/data/plot_stream.py` | trails, clutter, multi-radar |
| Difficulty | `scripts/data/difficulty_report.py` | concurrency, NN, score |
| Downstream | `run_cli.py --mode hybrid` | MOTA / precision / recall / ID switches |

**Golden hybrid eval settings (ops-aligned with 1 / 3 / 10 s radars)**

```text
--mode hybrid
--max-age 10      # ~10 s coast @ 1 s frames ≈ one long-range scan
--min-hits 2
--data data/canonical/stream_sweden_30min_holdout.jsonl
```

Coasting must cover the **slowest sole-coverage sensor** (10 s long-range), not the 1 s radar average.

---

## 8. Experimental results (data-backed)

### 8.1 Early Sweden ~10 min stream (untiled mini)

| Config | MOTA | Precision | Recall | ID sw |
|--------|-----:|----------:|-------:|------:|
| Hybrid default max-age=2 | 0.56 | 0.997 | 0.56 | 0 |
| Hybrid max-age=10, min-hits=2 | **0.977** | 0.981 | 0.996 | 0 |
| Hybrid + no clutter filter | 0.975 | 0.980 | 0.996 | 0 |
| Kalman-only same life-cycle | −1.32 | 0.30 | 0.99 | 0 |

**Conclusion:** association ML + multi-radar fusion drives precision; track **coasting** dominates recall under mixed scan rates.

### 8.2 Holdout 30 min (tiled geometry-augmented)

| Config | MOTA | Precision | Recall | MOTP | ID sw |
|--------|-----:|----------:|-------:|-----:|------:|
| Hybrid max-age=10, min-hits=2 | **0.976** | 0.979 | 0.998 | 105 m | 0 |

- Run name: `hybrid_sweden_30min_holdout_coast10`  
- MLflow run: `7e848988b55a4ee7a21a259668fb5be9`  
- **No track_id overlap** with the 30 min train stream (second half of 6-tile packing).  
- Metrics match the ~10 min mini result → association **generalizes across tile offsets**.

---

## 9. Recommendations

1. **Train MLPs** on `sim_batch_hetero.jsonl` (volume + labels).  
2. **Evaluate hybrid** on `stream_sweden_30min_holdout.jsonl` with max-age=10, min-hits=2.  
3. **Do not** use sparse long subset cuts as “hard” multi-target data.  
4. **Report difficulty scores** alongside MOTA.  
5. **Future data paper / upgrade path:** DIS multi-radar exports → same schema; hard-crossing synthesis; time-based coasting; SSR identity ablation.  

---

## 10. Reproduction commands

```powershell
# Dense CAT-62 tiles
uv run python scripts/data/build_dense_cat62.py --input data/cat_62_sweden_mini.txt `
  --output data/canonical/cat62_sweden_dense_30min.txt --tiles 3

# Streams
uv run python -m src.generate_streaming_training_data --region sweden `
  --input data/canonical/cat62_sweden_dense_30min.txt `
  --output data/canonical/stream_sweden_30min_train.jsonl --max-duration 1900

# QA + difficulty
uv run python scripts/data/validate_dataset.py data/canonical/stream_sweden_30min_train.jsonl
uv run python scripts/data/difficulty_report.py data/canonical --out artifacts/data_difficulty_report.md
uv run python scripts/data/plot_stream.py data/canonical/stream_sweden_30min_train.jsonl --t0 0 --t1 180

# Hybrid holdout
uv run python run_cli.py --interactive --mode hybrid `
  --data data/canonical/stream_sweden_30min_holdout.jsonl `
  --max-age 10 --min-hits 2 --run-name hybrid_sweden_30min_holdout_coast10
```

---

## 11. Summary

| Topic | Stance |
|-------|--------|
| Is 15 min enough? | Enough for **MLP + hybrid feasibility**; not a full hard multi-scenario corpus. |
| 30 / 60 min? | Yes — built via **dense tiling**, not naïve long sparse cuts. |
| Measurement model? | Gaussian multi-radar plot synthesis from CAT-062. |
| track_id in MLPs? | **Labels only**, not features. |
| Standalone data paper? | This chapter is the right weight; deepen generator before a second paper. |

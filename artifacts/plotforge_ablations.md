# PlotForge domain adapt + V8 ablations

Holdout: PlotForge UAE stream **seed 91, 90 s** (20 258 hits, 24 radars, 47 truth tracks).
Train: PlotForge **seed 7, 180 s** (43 646 hits). Same Kalman / gates / track manager
(`min-hits=2`, `max-age=10`, match 7000 m, 1 s windows).

Parent commit: `8690d5b`.

Scripts: `train_plotforge.py`, `eval_plotforge.py`.
Weights: `checkpoints/plotforge/`. Raw numbers: `artifacts/plotforge_eval_latest.json`.

## Operational pick

**Hybrid MLP, cluster τ = 0.2, assign τ = 0.35, dustbin on.**

| | MOTA | MOTP | P | R | IDSW | FP |
|---|---:|---:|---:|---:|---:|---:|
| PlotForge MLP + A4b τ | **0.896** | 191 m | 0.941 | 0.999 | 185 | 279 |
| PlotForge MLP, default τ 0.5 / 0.0 | 0.865 | 174 m | 0.912 | 1.000 | 171 | 431 |
| Sweden MLP zero-shot on PlotForge | 0.774 | 235 m | 0.846 | 0.998 | 187 | 817 |
| Native Sweden hybrid (30 min, not PlotForge) | 0.976 | 105 m | 0.979 | 0.998 | 0 | — |

Domain gap to native Sweden is now **0.080 MOTA** (was 0.202 zero-shot). Remaining leak is ID switches and MOTP under 24-radar overlap + CMB splits, not the pair scorer.

```bash
uv run python eval_plotforge.py \
  --data path/to/plotforge_holdout.jsonl --assoc mlp \
  --clutter-model checkpoints/plotforge/clutter_classifier.pt \
  --psr-model checkpoints/plotforge/pairwise_psr_psr.pt \
  --ssr-model checkpoints/plotforge/pairwise_ssr_any.pt \
  --cluster-threshold 0.2 --assign-threshold 0.35 --dustbin \
  --min-hits 2 --max-age 10 --no-snapshots \
  --out artifacts/plotforge_eval/holdout.json
```

## Train val (seed 7, intra-file ID split)

| Head | F1 | P | R |
|---|---:|---:|---:|
| clutter | 1.000 | 1.000 | 1.000 |
| PSR–PSR MLP | 0.997 | 0.999 | 0.995 |
| SSR–ANY MLP | 0.976 | 0.965 | 0.987 |
| A3 dual+gated V8 (ep 1) | 0.140 | 0.996 | 0.075 |
| V8 PF-tuned shared head | 0.128 | 0.998 | 0.068 |
| A2 rel-only, attn off (ep 6) | 0.040 | 0.994 | 0.020 |

Pair-F1 on intra-file holdout IDs is **misleading** (P ≈ 0.99, R = 0.02–0.14). Ship on tracking MOTA.

## Ablations 1–4 (V8)

A1 encode only the gated clique. A2 score on 12-d `rel_ij` only, attention off.
A3 dual PSR/SSR score heads. A4 calibrate cluster/assign τ (+ Hungarian dustbin).

### Calibrated τ 0.2 / 0.35 (apples to apples)

| Run | MOTA | P | IDSW | FP | MOTP |
|---|---:|---:|---:|---:|---:|
| Hybrid MLP + A4b | **0.896** | 0.941 | 185 | 279 | 191 m |
| A2 rel-only + A4b | 0.878 | 0.930 | 201 | 338 | 236 m |
| A4b Sweden V8 | 0.706 | 0.823 | 339 | 965 | 319 m |
| A1 gated + A4b | 0.686 | 0.813 | 366 | 1028 | 329 m |
| A3 dual+gated + A4b | 0.680 | 0.798 | 289 | 1136 | 236 m |

### Default τ 0.5 / 0.0

| Run | MOTA | P | IDSW | FP |
|---|---:|---:|---:|---:|
| Hybrid MLP | 0.865 | 0.912 | 171 | 431 |
| Ensemble MLP+V8 (PF V8) | 0.867 | 0.915 | 177 | 414 |
| A1 gated-encode | 0.418 | 0.683 | 512 | 2065 |
| V8 Sweden | 0.206 | 0.604 | 610 | 2907 |
| A2 rel-only | −0.127 | 0.513 | 796 | 4249 |
| V8 PF-tuned | −0.425 | 0.449 | 892 | 5475 |
| A3 dual+gated | −0.671 | 0.409 | 1008 | 6474 |

### A4 sweep on Sweden V8

| Variant | cluster / assign | dustbin | MOTA | P | IDSW | FP |
|---|---|---|---:|---:|---:|---:|
| A4b | 0.2 / 0.35 | on | **0.706** | 0.823 | 339 | 965 |
| A4c | 0.3 / 0.50 | on | 0.545 | 0.741 | 462 | 1568 |
| A4a | 0.3 / 0.35 | on | 0.541 | 0.738 | 462 | 1589 |
| A4d | 0.3 / 0.35 | off | 0.541 | 0.738 | 462 | 1589 |

Dustbin is a no-op at these τ (A4a == A4d).

## Read

- **A4 is the dominant lever.** Sweden V8 0.206 → 0.706; A2 −0.127 → 0.878; MLP 0.865 → 0.896. Same architecture, lower cluster bar / higher assign bar.
- **A1** gated-encode helps at default τ (0.206 → 0.418) and is slightly worse than A4b alone once τ is calibrated (0.686 vs 0.706).
- **A2** is the best V8 (0.878) once τ is calibrated, still 0.017 MOTA and 45 m MOTP behind MLP at the same τ. It is a pairwise MLP on 12-d `rel_ij`.
- **A3** dual heads: 1-epoch best then early-stop. Worst at default τ; no gain over Sweden V8 + A4b.
- Ensemble (0.5 MLP + 0.5 V8) rides the MLP (0.863–0.867). Not a gain.
- 61 % of PlotForge hits have `sensor_id > 8`, so Sweden V8 `max_sensors=8` aliases radars 9–24. Irrelevant for A2 (no `sensor_emb`) and for MLP.

## Suggested 5–8 (not run)

| # | Change | Verdict |
|---|---|---|
| 5 | `max_sensors=32` | Only if keeping V8+attn. No-op for MLP / A2. |
| 6 | Temperature-scale logits | Dual of A4. Skip. |
| 7 | Hard negatives | Would push already-conservative pair scores down. Skip for cluster. |
| 8 | Freeze encoder, then unfreeze attn | A2 *is* freeze-encoder taken all the way. Only worth as a residual on A2. |

Do not spend a train on 5–8 as written. If V8 continues: freeze the A2 rel-head and add gated-clique attention as a residual.

## Checkpoints

| File | What |
|---|---|
| `checkpoints/plotforge/clutter_classifier.pt` | PlotForge clutter MLP |
| `checkpoints/plotforge/pairwise_psr_psr.pt` | PlotForge PSR–PSR pairwise |
| `checkpoints/plotforge/pairwise_ssr_any.pt` | PlotForge SSR–ANY pairwise |
| `checkpoints/plotforge/model_v8_assoc_best.pt` | V8 fine-tune on seed 7 (shared head; pair-F1 0.128) |
| `checkpoints/plotforge/ablate_a2_rel_best.pt` | A2 rel-only, attn off |
| `checkpoints/plotforge/ablate_a3_dual_best.pt` | A3 dual heads + gated encode |
| `checkpoints/model_v8_assoc_best.pt` | Sweden V8 (unchanged) |

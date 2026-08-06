# V7 Design: Transformer Tracker-Correlator

## Goal

Replace graph message-passing (GAT) with a **pure Transformer** stack for multi-sensor track–measurement association and state update, while keeping the same streaming window training contract as V3–V6.

## Why Transformer

| Issue in GNN path | Transformer approach |
|-------------------|----------------------|
| Hand-built sparse edges / distance gates | Soft attention over full measurement set (or gated radius mask) |
| Symmetry between track and meas nodes | Explicit **encoder (meas)** / **decoder (tracks as queries)** |
| Edge feature engineering | Relative geometry injected as attention bias or token features |
| torch_geometric dependency | PyTorch `nn.TransformerEncoder` / `MultiheadAttention` only |

## Architecture

```
Measurements ──► Meas Encoder (MLP + sensor emb) ──► Transformer Encoder
                                                              │
Active Tracks ──► Track Encoder (MLP + memory)  ──► Decoder cross-attn ──► Heads
                                                              │
                                              ◄── GRU temporal memory (per track)
```

1. **Token features (8-D)**: `[x, y, z, vx, vy, vz, amp, dt_to_window]` + sensor embedding + role embedding (meas vs track).
2. **Measurement self-attention**: contextualizes co-located PSR/SSR plots (soft spatial clustering).
3. **Track→Meas cross-attention**: DETR-style matchmaking; tracks query measurements.
4. **Optional radius mask**: attention blocked beyond `max_assoc_m` for efficiency / realism.
5. **Heads**:
   - state residual Δs ∈ R⁶
   - existence / initiation logits
   - clutter logit (measurement tokens)
6. **Temporal**: GRUCell on track tokens between windows (same pattern as V6).

## Training

- Stream windows (default 2 s), Hungarian match on position for regression targets.
- Loss: Smooth-L1 on matched states + focal BCE existence + FP penalty + optional attention entropy.
- Data: any JSONL loadable by `stream_utils.load_stream_and_truth`.

## Interfaces

- Module: `src/model_v7_transformer.py` (`TransformerTrackerV7`)
- Train: `src/train_streaming_v7.py`
- Checkpoint: `checkpoints/model_v7_transformer.pt`
- Factory key: `v7` / `transformer`

## Non-goals (v1 of this design)

- Full DETR bipartite matching loss (use Hungarian on state only, like V6).
- Learned continuous-time KF replacement (state residual only).
- Production hybrid handoff (research track parallel to Hybrid).

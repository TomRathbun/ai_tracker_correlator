### Hybrid Tracker: AI/ML Feasibility for Unified Radar Fusion in Air Traffic Systems

#### Core Concept: Single AI/ML Tracker for Multi-Sensor Inputs
In standard setups, each radar (PSR for raw position/Doppler, SSR for transponder IDs) runs its own tracker to filter plots and build local tracks, then a correlator fuses them using rules like Mahalanobis gating or probabilistic matching to resolve duplicates. Our hybrid tracker flips this: One ML-driven system processes all decoded plots together, learning to associate and fuse across sensors for a single correlated output. It's "hybrid" because it combines classical elements (like Kalman for reliable updates) with AI (classifiers and GNNs) to handle real-world noise—PSR clutter, SSR intermittency, or datalink delays—more adaptively.

This research tests if that's feasible: Can AI/ML simplify the pipeline while hitting metrics like MOTA >0.9 on sim data (and potentially real Asterix feeds), with zero ID switches in heterogeneous environments?

#### Pipeline Breakdown: From Plots to Correlated Tracks
The tracker processes frames sequentially, assuming inputs are time-aligned Asterix-decoded Cartesian plots.

1. **Clutter Rejection (Unary Classifier)**: PSR often floods with false returns from multipath or weather. We use an MLP on per-plot features (amplitude, Doppler velocity, range-normalized position, sensor type) to score P(clutter). Threshold at 0.5 to filter—trained on sim examples where low-amp/erratic-velocity plots are labeled junk. This gates early, reducing load on downstream association (e.g., FP/frame near 0 in tests).

2. **Pairwise Association (Dual Classifiers)**: For data association, we split by sensor: PSR-PSR pairs use kinematic MLPs (features like position dist, velocity cosine sim, angular sep) to score P(same target). SSR-involved pairs prioritize Mode 3A/S matching, falling back to position if codes mismatch. Thresholds (0.35 for PSR, 0.5 for SSR) create a sparse similarity matrix—key for fusing across datalinks without explicit rules.

3. **Graph Construction**: Turn scores into a graph: Nodes are plots, edges are high-P links with attrs like prob and dist. Use connected components to cluster duplicates (e.g., same plane seen by multiple radars). Fuse clusters into meta-plots (weighted avg position/velocity, preserving SSR IDs for correlation).

4. **State Estimation (GNN + Hybrid Update)**: A GNN (GAT layers for attention over edges, GRU for recurrence) predicts state deltas from the graph, blending new meta-plots with prior tracks. For feasibility, we hybridize: GNN handles nonlinear association/fusion, Kalman corrects with tuned noise covs. This outputs updated tracks, with logits for existence confidence.

5. **Track Management**: M/N rules (min 5 hits to confirm, max 5 misses to coast) promote/demote based on GNN logits and associations. Outputs only confirmed tracks—ensuring correlated, de-duplicated results (zero ID switches in sim).

#### Feasibility Insights from Tests
On sim data mimicking PSR/SSR mixes (with noise/clutter), we hit MOTA 0.925, recall 92.6%, precision 99.9%—far better than SORT baselines. The AI learns to prioritize SSR IDs for stable correlation, while GNN adapts to PSR Doppler for accurate velocity fusion. Real-world next: Fine-tune on Asterix traces to handle biases like transponder dropouts.

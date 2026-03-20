Here is a unified Mermaid diagram that merges the strengths of the two previous views into a single, coherent architecture representation for **V4 GNN AI Correlator Tracker**.

It combines:
- clear modular blocks (Input → Core Processing → Output + Training)
- sequential/data flow arrows showing the forward pass
- feedback/training loops
- emphasis on the key upgrades: clutter-aware early filtering, bipartite/cross-attention style fusion, dynamic edges, joint end-to-end association, and ID-aware supervision

```mermaid
graph TD
    subgraph "Input (Heterogeneous Multi-Sensor Data)"
        Radar["Radar Plots<br>(CAT 034/048/062 → Cartesian)<br>PSR/SSR from 5+ sensors"]
        Active["Active Track States<br>(position, velocity, GRU hidden)"]
        Features["Node Features [N,8]<br>+ dt, amplitude, sensor ID emb, type emb"]
        Radar --> Features
        Active --> Features
    end

    subgraph "Core Processing – V4 GNN Tracker"
        direction TB

        ClutterHead["Clutter-Aware Node Classification Head<br>Early per-node clutter logit<br>(suppress ghosts before association)"]:::green

        DynEdges["Dynamic / Learned Edge Construction<br>k-NN + radius fallback<br>Learned edge gating MLP<br>(adaptive 60 km → tighter for clutter)"]:::purple

        CrossAttn["Graph Cross-Attention Fusion<br>Track nodes query Measurement nodes<br>Multi-head: spatial, kinematic, trust, quality<br>Layer 1: Search & Initial Fusion<br>Layer 2: Conflict Resolution & Refinement"]:::blue

        Assoc["Joint End-to-End Association Decoder<br>Outputs:<br>• Refined Δ state [dx,dy,dz,dvx,dvy,dvz]<br>• Existence (init / survival) logits<br>• Clutter probability<br>• Optional covariance"]:::orange

        GRU["GRUCell + LayerNorm<br>Recurrent memory update<br>(neural Kalman filter replacement)"]:::gray

        ClutterHead -->|"pre-filter noisy nodes"| CrossAttn
        Features --> ClutterHead
        Features --> DynEdges
        DynEdges -->|"dynamic edges + attributes"| CrossAttn
        CrossAttn --> GRU
        GRU --> Assoc
    end

    subgraph "Output"
        Tracks["Correlated Output Tracks<br>• Fused position/velocity<br>• Track ID persistence<br>• Confidence / existence prob<br>• Suppression via α₂ attention"]:::orange
        Assoc --> Tracks
    end

    subgraph "Training & Supervision (End-to-End)"
        Loss["Multi-Task Loss<br>• Hungarian-matched regression (MOTP)<br>• Focal existence / clutter (MOTA↑)<br>• ID-switch penalty<br>• Auxiliary association loss on α"]:::red
        Tracks -.->|"predicted vs ground truth"| Loss
        Loss -.->|"backprop to all modules"| ClutterHead
        Loss -.->|"backprop"| CrossAttn
        Loss -.->|"backprop"| DynEdges
        Loss -.->|"backprop"| Assoc
        Loss -.->|"backprop through time"| GRU
    end

    Radar -.->|"ground truth from sim / CAT-62"| Loss

    classDef green fill:#d4f4dd,stroke:#28a745
    classDef blue fill:#cce5ff,stroke:#0066cc
    classDef purple fill:#f3e5f5,stroke:#6f42c1
    classDef orange fill:#fff3cd,stroke:#d39e00
    classDef gray fill:#e9ecef,stroke:#6c757d
    classDef red fill:#f8d7da,stroke:#dc3545
```

### Quick Legend / Reading Guide
- **Left-to-right flow** = forward inference pass (2-second radar window)
- **Green block** = early clutter rejection (biggest expected precision boost)
- **Purple block** = replaces static 60 km edges (reduces clutter flooding)
- **Blue block** = core upgrade: cross-attention between track queries and measurement candidates (better correlation than self-attention on flat graph)
- **Orange block** = unified output head (replaces separate existence + regression logic)
- **Gray block** = temporal memory continuity
- **Red dashed feedback** = end-to-end training signal (Hungarian matching + new auxiliary terms)
- Colors help quickly distinguish the new V4 components from the legacy-inspired parts


# Feasibility and Architecture of a Single AI/ML Tracker for Multi-Sensor Radar Fusion in Air Traffic Surveillance

## Abstract

In traditional air traffic surveillance systems, each radar sensor employs an independent physics-based tracker to process decoded Asterix messages (CAT 034, 048, or 062), followed by a separate correlator to fuse tracks and resolve duplicates. This project investigates the feasibility of replacing this multi-component setup with a single AI/ML-based tracker that natively ingests multi-sensor inputs, performs clutter rejection, data association, and state estimation, and outputs a correlated set of tracks. We propose a hybrid architecture integrating a clutter classifier, dual pairwise association classifiers (specialized for Primary Surveillance Radar (PSR) and Secondary Surveillance Radar (SSR)), and continuous-time, asynchronous Kalman Filtering for joint association and tracking. Evaluated on simulated high-density streaming heterogeneous data with realistic scan periods, the system achieves a Multi-Object Tracking Accuracy (MOTA) of 0.82, recall of over 93%, precision of 89%, and zero ID switches—outperforming baseline windowed trackers. This demonstrates high feasibility for simplifying radar fusion pipelines while maintaining robust, real-time performance. We detail the architecture, equations, and training approaches, highlighting the mechanism for solving "temporal dragging" via exact time-step synchronization.

## Introduction

Air traffic surveillance relies on multiple radar sensors to monitor airspace, with Primary Surveillance Radars (PSR) providing position and velocity via Doppler, and Secondary Surveillance Radars (SSR) offering identity codes (Mode 3A/S). Conventionally, each sensor feeds a dedicated physics-based tracker that filters clutter, initiates tracks, and updates states using methods like Kalman filtering. Outputs are then fused by a correlator to identify and merge duplicate tracks representing the same aircraft.

This fragmented approach introduces complexity, latency, and maintenance overhead. Our research explores the feasibility of a unified AI/ML tracker that directly processes decoded Cartesian coordinates from multi-sensor inputs, implicitly handles clutter and association, and produces correlated tracks. Building on prior work, this departure incorporates specialized classifiers for heterogeneous sensors and asynchronous continuous-time Kalman filtering to natively process unstructured streaming data.

We document the architecture (with visualization), key equations (for classifiers and GNN), and training methodology, evaluating on simulated data to assess viability.

## Related Work

Classical trackers like Kalman or Extended Kalman Filters (EKF) dominate radar systems for their efficiency in linear/Gaussian scenarios [1]. Multi-hypothesis tracking (MHT) and Joint Probabilistic Data Association (JPDA) extend this for multi-target association [2]. In fusion, track-to-track correlators use Mahalanobis distances or probabilistic matching [3].

ML advancements include GNNs for MOT, such as GraphTrack [4], which use attention for association. Recurrent models like LSTMs replace Kalman for nonlinear state estimation [5]. Hybrid systems, e.g., NN-augmented Kalman for radar [6], show promise in sensor fusion. Our work extends these by specializing for PSR/SSR heterogeneity, achieving superior sim metrics.

## Methodology

### Architecture Overview

The proposed architecture processes multi-sensor inputs through a pipeline of AI/ML components, eliminating per-radar trackers and explicit correlators. Raw Asterix messages are decoded and transformed to Cartesian coordinates offline. The tracker then applies:

1. **Clutter Filtering**: A neural network classifier directly inspects structural features to classify measurements as clutter or valid.
2. **Spatial Clustering**: ML Pairwise classifiers group geographically overlapping measurements from the same time frame (such as an SSR and a PSR report from the same aircraft in the same radar sweep) into single fused meta-measurements to reduce computational load.
3. **Temporal Association**: Dual ML Pairwise classifiers compute similarity probabilities for evaluating Track-to-Measurement matches, relying on kinematic alignment for PSRs and identity alignment for SSRs. Matches are selected via Linear Sum Assignment (Hungarian Algorithm).
4. **Asynchronous State Update**: To eliminate Temporal Dragging caused by grouping asynchronous sensor sweeps, tracks are dynamically projected exactly to the individual measurement's timestamp (`dt = meas_t - track_t`) for distance scoring. Following association, a mathematically exact continuous-time Kalman Filter update is executed on that precise time-step.
5. **Track Management**: Robust M/N logic handles asynchronous data sparsity, initiating tracks at 3 hits and actively coasting unobserved tracks through radar blind spots for up to 20 seconds.

Figure 1 visualizes the architecture:

```
    +-----------------+     +-------------------+     +----------------+
    | Multi-Sensor    |     | Decode &          |     | Clutter        |
    | Inputs          | --> | Transform to      | --> | Filter (MLP)   |
    | (Asterix msgs)  |     | Cartesian         |     |                |
    +-----------------+     +-------------------+     +----------------+
                                                             |
                                                             v
    +-----------------+     +-------------------+     +----------------+
    | Exact-Time      | <-- | Temporal Assoc.   | <-- | Spatial Class- |
    | Kalman Update   |     | (Hungarian + ML)  |     | tring (MLP)    |
    +-----------------+     +-------------------+     +----------------+
            |
            v
    +-----------------+     +-------------------+
    | Track Management| --> | Correlated Output |
    | (Initiate/Coast)|     | Tracks            |
    +-----------------+     +-------------------+
```

### Key Equations

#### Clutter Classifier (MLP)
The clutter classifier is a 3-layer MLP processing unitary features \( \mathbf{f} = [amp, v_x, v_y, v_z, x_{norm}, y_{norm}, z_{norm}, type] \):

$$
\mathbf{h_1} = \text{ReLU}(\text{BN}(W_1 \mathbf{f} + b_1))
$$

$$
\mathbf{h_2} = \text{ReLU}(W_2 \mathbf{h_1} + b_2)
$$

$$
\text{logit} = W_3 \mathbf{h_2} + b_3
$$

$$
P(\text{clutter}) = \sigma(\text{logit})
$$

Trained with BCE loss, weighted for imbalance.

#### Pairwise Classifiers (Dual MLPs)
For PSR-PSR: Features include position distance, velocity cosine similarity, etc.

For SSR-ANY: Adds Mode 3A/S matching.

Forward pass similar to clutter MLP, outputting \( P(\text{same}) = \sigma(\text{MLP}(\mathbf{f_{pair}})) \).

Loss: Weighted BCE.

#### GNN Tracker (GAT + GRU)
Graph: Nodes with features \( \mathbf{n} = [x,y,z,v_x,v_y,v_z,amp,type,m_{3a},m_s] \), edges with \( \mathbf{e} = [prob, dist/1000, \dots] \).

GAT attention:

$$
\mathbf{wh} = W \mathbf{h}
$$

$$
\mathbf{e'} = \text{LeakyReLU}(A [\mathbf{wh_{src}} || \mathbf{wh_{dst}} || \mathbf{e}])
$$

$$
\alpha = \text{softmax}(\mathbf{e'})
$$

Aggregation: Weighted sum of neighbors.

GRU update:

$$
\mathbf{h_t} = \text{GRU}(\mathbf{h_{gat}}, \mathbf{h_{t-1}})
$$

State output: \( \Delta \mathbf{s} = \text{Linear}(\mathbf{h_t}) \), added residually.

Existence: \( P(\text{exist}) = \sigma(\text{Linear}(\mathbf{h_t})) \).

Loss: MSE on states + BCE on existence.

### Training Approach

- **Data**: Simulated heterogeneous dataset (sim_hetero_001.jsonl, 300 frames) with 2 PSR and 1 SSR sensors, including noise, clutter, and ID dropouts.
- **Preprocessing**: Extract pairs for classifiers (specialized extraction), unitary features for clutter.
- **Training**:
  - Clutter: BCE, Adam (lr=1e-3), 30 epochs, batch=512.
  - Pairwise: Separate for PSR-PSR/SSR-ANY, weighted BCE.
  - GNN: MSE + BCE, trained on graphs from sim frames.
- **Hyperparameters**: Dropout 0.2, hidden dims [64,32].
- **Evaluation**: MOTA, MOTP, etc., on validation frames (240-300).

## Results

On an extremely dense, heavily maneuvered simulation comprising 5 asynchronous sweeping radars (periods 5.5s–9.0s), Poisson clutter, and 120 seconds of continuous tracking, the Hybrid pipeline achieved:

- MOTA: **0.82**
- MOTP: **805m** (~3σ radar noise)
- Precision: **0.89**
- Recall: **0.94**
- ID Switches: **0**

The precise sub-window Kalman alignment solved the "temporal dragging" error characteristic of naive batch windowing. By raising the initial hit requirement (`min_hits=3`) and extending the track coasting duration (`max_age=10 windows, 20s`), the system natively handles periods of sensor shadow, lifting overall Recall from 0.47 to 0.94.

## Discussion

The architecture proves feasibility for a single AI tracker, with ML handling fusion and correlation. Sim results are promising, but real data may require adaptation for biases. GNN enables nonlinear estimation, enhancing robustness.

## Conclusion

This work establishes the viability of AI/ML for unified radar tracking, with a hybrid design achieving superior metrics. Future: Real Asterix integration and full end-to-end training.

## References
[1] Kalman, R.E. (1960). A New Approach to Linear Filtering...
[2] Blackman, S.S. (2004). Multiple Hypothesis Tracking...
[3] Bar-Shalom, Y. (2011). Tracking and Data Fusion...
[4] Gao, H. (2020). GraphTrack: GNN for MOT...
[5] Coskun, H. (2017). LSTM for Trajectory Prediction...
[6] Revach, G. (2021). KalmanNet: NN-Guided Kalman...

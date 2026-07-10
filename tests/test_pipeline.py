"""
Basic tests for the modular Pipeline and state updaters.

These exercise the common track management, kalman mode (no external model deps),
config wiring, and basic initiation/coasting/promotion behavior using synthetic data.
GNN/hybrid modes are lightly exercised (they will warn on missing checkpoints but should not crash).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.pipeline import Pipeline
from src.config_schemas import PipelineConfig
from src.updater import FallbackUpdater


def make_synthetic_measurements(t: float, n: int = 2, spread: float = 50.0, base_x: float = 1000.0):
    """Create a small cluster of measurements around a moving target."""
    meas = []
    for i in range(n):
        m = {
            't': t,
            'x': base_x + (i * spread) + np.random.normal(0, 5),
            'y': 2000.0 + np.random.normal(0, 5),
            'z': 500.0 + np.random.normal(0, 2),
            'vx': 80.0 + np.random.normal(0, 1),
            'vy': 10.0,
            'vz': 0.0,
            'sensor_id': i % 2,  # mix PSR/SSR-ish
            'amplitude': 60.0,
        }
        meas.append(m)
    return meas


def test_kalman_pipeline_basic():
    """Kalman mode should initiate, promote, and coast tracks using the unified pipeline."""
    cfg = PipelineConfig()
    cfg.state_updater.type = "kalman"
    cfg.track_manager.min_hits = 3
    cfg.track_manager.max_age = 4
    cfg.state_updater.track_cap = 10

    pipe = Pipeline(cfg)

    # Frame 1: first sightings -> tentative
    meas1 = make_synthetic_measurements(0.0)
    confirmed = pipe.process_frame(meas1, t=0.0)
    assert len(confirmed) == 0, "Should not confirm on first hit"
    assert len(pipe.tracks) >= 1

    # Frame 2 + 3: more hits -> should promote after min_hits
    for t in [1.0, 2.0]:
        meas = make_synthetic_measurements(t, n=len(meas1))
        confirmed = pipe.process_frame(meas, t=t)
    assert len(confirmed) >= 1, "Should have promoted at least one track after min_hits"

    # Record a confirmed track id
    track_ids_before = {t.get('track_id') for t in confirmed if 'track_id' in t}

    # Now miss for several frames -> ages increase, eventually drop below max_age
    for t in [3.0, 4.0, 5.0, 6.0, 7.0]:
        confirmed = pipe.process_frame([], t=t)  # no measurements

    # After enough misses, the previously confirmed should be gone (aged out)
    still_alive = [tt for tt in pipe.tracks if tt.get('track_id') in track_ids_before]
    assert len(still_alive) == 0, "Tracks should have coasted out after exceeding max_age"

    # Reset should clear state
    pipe.reset()
    assert len(pipe.tracks) == 0
    assert pipe.last_t is None

    print("test_kalman_pipeline_basic PASSED")


def test_config_overrides_applied():
    """CLI-style overrides on PipelineConfig should be respected by the pipeline."""
    cfg = PipelineConfig()
    cfg.state_updater.type = "kalman"
    cfg.track_manager.min_hits = 5   # higher than default
    cfg.track_manager.max_age = 2    # very short coasting

    pipe = Pipeline(cfg)
    assert pipe.config.track_manager.min_hits == 5
    assert pipe.config.track_manager.max_age == 2

    # Run a couple frames; with high min_hits we should stay tentative longer
    for t in [0.0, 1.0, 2.0]:
        meas = make_synthetic_measurements(t)
        conf = pipe.process_frame(meas, t=t)
        assert len(conf) == 0, "With min_hits=5 we should not confirm after only 3 hits"

    print("test_config_overrides_applied PASSED")


def test_gnn_and_hybrid_no_crash_on_missing_models():
    """GNN and hybrid modes should initialize the Pipeline without exploding when checkpoints are absent.
    They will log warnings and fall back gracefully (model=None or classifiers missing).
    """
    for mode in ["gnn", "hybrid"]:
        cfg = PipelineConfig()
        cfg.state_updater.type = mode
        # Point to non-existent checkpoint so we exercise the warning paths
        cfg.state_updater.gnn_model_path = Path("checkpoints/does_not_exist.pt")

        pipe = Pipeline(cfg)  # should not raise
        meas = make_synthetic_measurements(0.0, n=1)
        out = pipe.process_frame(meas, t=0.0)
        # May return [] because no real model, but must not crash
        assert isinstance(out, list)

    print("test_gnn_and_hybrid_no_crash_on_missing_models PASSED")


def test_fallback_updater_direct():
    """Directly exercise FallbackUpdater (kalman) with dt and frame_t for signature coverage."""
    cfg = PipelineConfig()
    upd = FallbackUpdater(cfg)

    tracks = []
    meas = make_synthetic_measurements(10.0, n=1)

    # predict must accept dt
    tracks = upd.predict(tracks, dt=0.5)

    # update must accept frame_t (and dt)
    updated = upd.update(meas, tracks, dt=0.5, frame_t=10.5)
    assert isinstance(updated, list)

    print("test_fallback_updater_direct PASSED")


if __name__ == "__main__":
    test_kalman_pipeline_basic()
    test_config_overrides_applied()
    test_gnn_and_hybrid_no_crash_on_missing_models()
    test_fallback_updater_direct()
    print("\nAll pipeline tests passed!")

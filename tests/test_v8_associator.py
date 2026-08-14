"""Unit tests for V8 association transformer (no checkpoint required)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.config_schemas import PipelineConfig, PairwiseConfig
from src.model_v8_associator import (
    ASSOC_GATE_M,
    AssociationTransformerV8,
    mode_3a_index,
    project_track_to_time,
    relative_feature_vec,
)


def _meas(x, y, z=1000.0, tid=-1, **extra):
    m = {"t": 10.0, "x": x, "y": y, "z": z, "vx": 100.0, "vy": 0.0, "vz": 0.0, "amplitude": 50.0, "track_id": tid, "sensor_id": 0, "meas_type": "PSR"}
    m.update(extra)
    return m


def test_mode_3a_index_stable():
    assert mode_3a_index(None) == 0
    assert mode_3a_index("") == 0
    a = mode_3a_index("1200")
    b = mode_3a_index("1200")
    assert a == b
    assert 0 <= a < 4096


def test_project_track_to_time():
    tr = {"x": 0.0, "y": 0.0, "z": 0.0, "vx": 100.0, "vy": 0.0, "vz": 0.0, "kf_t": 0.0}
    out = project_track_to_time(tr, 2.0)
    assert abs(out["x"] - 200.0) < 1e-6
    assert abs(out["_dt"] - 2.0) < 1e-6


def test_relative_identity_flags():
    a = _meas(0, 0, tid=1, mode_3a="1200", mode_s="ABC")
    b = _meas(100, 0, tid=1, mode_3a="1200", mode_s="ABC")
    rel = relative_feature_vec(a, b)
    assert rel.shape == (12,)
    assert rel[9] == 1.0  # mode 3A match
    assert rel[10] == 1.0  # mode S match
    c = _meas(100, 0, tid=2, mode_3a="2200")
    rel2 = relative_feature_vec(a, c)
    assert rel2[9] == -1.0
    stream = {"x": 0.0, "y": 0.0, "z": 0.0, "mode3a": "1200"}
    rel_alias = relative_feature_vec(a, stream)
    assert rel_alias[9] == 1.0
    # missing vel must not produce a fake cosine match
    d = {"x": 0.0, "y": 0.0, "z": 0.0}
    e = {"x": 10.0, "y": 0.0, "z": 0.0}
    rel_miss = relative_feature_vec(d, e)
    assert rel_miss[5] == 0.0
    print("test_relative_identity_flags PASSED")


def test_forward_score_pairs_and_assignment():
    model = AssociationTransformerV8(hidden_dim=32, num_heads=4, num_layers=1)
    model.eval()
    items = [_meas(0, 0, tid=1), _meas(150, 20, tid=1), _meas(8000, 0, tid=2)]
    idx = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    with torch.no_grad():
        logits = model.score_pairs(items, items, idx)
    assert logits.shape == (2,)
    # clustering shorthand still accepted
    with torch.no_grad():
        logits2 = model.score_pairs(items, idx)
    assert logits2.shape == (2,)
    assert torch.isfinite(logits).all()

    tracks = [{"x": 0.0, "y": 0.0, "z": 1000.0, "vx": 80.0, "vy": 0.0, "vz": 0.0, "kf_t": 9.0, "age": 0, "hits": 3, "mode_3a": "1200"}]
    metas = [_meas(80, 0, tid=1, t=10.0, mode_3a="1200"), _meas(20000, 0, tid=2, t=10.0)]
    with torch.no_grad():
        S, dust = model.score_assignment(tracks, metas)
    assert S.shape == (1, 2)
    assert dust.shape == (1,)
    assert torch.isfinite(S).all() and torch.isfinite(dust).all()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 10_000
    print(f"test_forward_score_pairs_and_assignment PASSED params={n_params}")


def test_empty_sets():
    model = AssociationTransformerV8(hidden_dim=32, num_heads=4, num_layers=1)
    model.eval()
    with torch.no_grad():
        empty_pairs = model.score_pairs([_meas(0, 0)], torch.zeros((0, 2), dtype=torch.long))
        S, dust = model.score_assignment([], [_meas(0, 0)])
        S2, dust2 = model.score_assignment([{"x": 0.0, "y": 0.0, "z": 0.0, "kf_t": 0.0}], [])
    assert empty_pairs.numel() == 0
    assert S.shape[0] == 0
    assert dust.numel() == 0
    assert S2.shape == (1, 0)
    assert dust2.shape == (1,)


def test_config_backend_default_mlp():
    cfg = PipelineConfig()
    assert cfg.pairwise.backend == "mlp"
    assert cfg.pairwise.use_dustbin is False
    pw = PairwiseConfig(backend="transformer", use_dustbin=True)
    assert pw.backend == "transformer"
    assert pw.use_dustbin is True
    print("test_config_backend_default_mlp PASSED")


def test_assoc_gate_constant():
    assert ASSOC_GATE_M == 8000.0


if __name__ == "__main__":
    test_mode_3a_index_stable()
    test_project_track_to_time()
    test_relative_identity_flags()
    test_forward_score_pairs_and_assignment()
    test_empty_sets()
    test_config_backend_default_mlp()
    test_assoc_gate_constant()
    print("\nAll V8 associator tests passed!")

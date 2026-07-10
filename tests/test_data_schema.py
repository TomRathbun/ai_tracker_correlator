"""Unit tests for canonical data schema adapters."""

from src.data_schema import (
    get_meas_type,
    get_mode_3a,
    get_sensor_id,
    is_batch_frame,
    is_stream_hit,
    normalize_batch_frame,
    normalize_measurement,
    normalize_measurement_dict,
)


def test_normalize_stream_hit():
    raw = {
        "t": 1.5,
        "radar_id": 3,
        "meas_type": "SSR",
        "x": 100.0,
        "y": 200.0,
        "z": 3000.0,
        "mode3a": "6027",
        "mode_s": "ABC123",
        "track_id": 42.0,
        "gt_x": 101.0,
        "gt_y": 201.0,
        "source_lat": 59.6,
        "source_lon": 17.9,
    }
    m = normalize_measurement(raw)
    assert m.t == 1.5
    assert m.sensor_id == 3
    assert m.meas_type == "SSR"
    assert m.vx is None  # not present
    assert m.mode_3a == "6027"
    assert m.track_id == 42
    assert m.gt_x == 101.0

    d = m.to_legacy_dict()
    assert d["sensor_id"] == 3
    assert d["radar_id"] == 3
    assert d["type"] == "SSR"
    assert d["meas_type"] == "SSR"
    assert d["mode_3a"] == "6027"
    assert d["mode3a"] == "6027"
    assert "vx" not in d  # no fake zero


def test_normalize_batch_psr():
    raw = {
        "sensor_id": 0,
        "timestamp": 3.0,
        "type": "PSR",
        "x": 10.0,
        "y": 20.0,
        "z": 5000.0,
        "vx": 100.0,
        "vy": 50.0,
        "amplitude": 55.0,
        "track_id": 1,
    }
    m = normalize_measurement(raw)
    assert m.sensor_id == 0
    assert m.meas_type == "PSR"
    assert m.t == 3.0
    assert m.vx == 100.0
    assert m.amplitude == 55.0
    assert m.is_clutter is False


def test_clutter_flag():
    m = normalize_measurement({"t": 0, "x": 0, "y": 0, "z": 0, "track_id": -1})
    assert m.is_clutter is True
    assert m.track_id == -1


def test_batch_frame():
    frame = {
        "timestamp": 0.0,
        "measurements": [
            {"sensor_id": 0, "type": "PSR", "x": 1, "y": 2, "z": 3, "vx": 1, "vy": 2, "track_id": 0},
            {"sensor_id": 1, "type": "SSR", "x": 1, "y": 2, "z": 3, "mode_3a": 1000, "track_id": 0},
        ],
        "gt_tracks": [{"id": 0, "x": 1, "y": 2, "z": 3, "vx": 1, "vy": 2, "vz": 0, "t": 0.0}],
    }
    assert is_batch_frame(frame)
    norm = normalize_batch_frame(frame)
    assert len(norm["measurements"]) == 2
    assert norm["measurements"][0]["meas_type"] == "PSR"
    assert norm["measurements"][1]["mode_3a"] == "1000"
    assert get_sensor_id(norm["measurements"][1]) == 1


def test_helpers():
    d = normalize_measurement_dict(
        {"radar_id": 2, "meas_type": "ssr", "t": 1, "x": 0, "y": 0, "z": 0, "mode3a": "1200"}
    )
    assert get_sensor_id(d) == 2
    assert get_meas_type(d) == "SSR"
    assert get_mode_3a(d) == "1200"
    assert is_stream_hit(d)

"""Status payload provenance (A-4).

Self-reported pick state and verified success are separate fields: the phase
machine can only claim ``picked``; verification requires an explicit external
source and must never appear for a phase in which nothing was picked.
"""

import pytest
from tomato_harvest_sim.harvest_sim_node import PICKED_SOURCE, build_status_payload


def _payload(**overrides):
    base = {
        "phase": "LIFT",
        "target": [0.1, 0.2, 0.3],
        "eef": [0.4, 0.5, 0.6],
        "frame_index": 4,
        "frames_total": 20,
        "elapsed_s": 1.5,
        "trajectory_duration_s": 2.0,
        "dunk_every": 8,
        "joints": {"lift": 0.3},
        "error": None,
        "target_source": "recognition_artifact",
    }
    base.update(overrides)
    return build_status_payload(**base)


def test_status_payload_labels_self_reported_pick_state():
    payload = _payload()
    assert payload["picked"] is True
    assert payload["picked_verified"] is False
    assert payload["picked_source"] == PICKED_SOURCE
    assert payload["verified_source"] is None


def test_status_payload_carries_target_provenance():
    assert _payload()["target_source"] == "recognition_artifact"
    assert _payload(target_source=None)["target_source"] == "unknown"


def test_verified_pick_requires_external_source_and_picked_phase():
    with pytest.raises(ValueError):
        _payload(picked_verified=True)
    with pytest.raises(ValueError):
        _payload(phase="APPROACH", picked_verified=True, verified_source="vision")
    verified = _payload(picked_verified=True, verified_source="vision")
    assert verified["picked_verified"] is True
    assert verified["verified_source"] == "vision"

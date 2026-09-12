from tomato_harvest_sim.viewer_node import build_state, make_scene_payload

SCENE = {
    "scene_id": "scene-x",
    "fruits": [
        {"position": [0.1, 0.2, 1.0], "ripeness": 2, "cluster": 1},
        {"position": [0.2, 0.2, 1.1], "ripeness": 2, "cluster": 1},
    ],
    "stems": [[0.0, 0.0, 0.9]] * 1200,
    "target": {"position": [0.35, 0.52, 1.27]},
    "trajectory": [
        {"time_from_start_s": 0.0, "joints": {"lift": 0.0}, "eef": [0.3, 0.4, 1.2]},
        {"time_from_start_s": 1.0, "joints": {"lift": 0.1}, "eef": [0.36, 0.53, 1.27]},
    ],
    "eef_path": [[0.3, 0.4, 1.2], [0.36, 0.53, 1.27]],
}

STATUS = {
    "phase": "APPROACH",
    "picked": False,
    "frame_index": 3,
    "eef": [0.31, 0.41, 1.21],
    "target": [0.35, 0.52, 1.27],
}


def test_build_state_contains_webcheck_fields():
    state = build_state(STATUS, SCENE)
    for key in ("phase", "picked", "frame_index", "eef", "target", "points_count"):
        assert key in state, key
    assert state["phase"] == "APPROACH"
    assert state["picked"] is False
    assert state["frame_index"] == 3
    assert state["points_count"] >= 1200




def test_make_scene_payload_is_projectable():
    payload = make_scene_payload(SCENE, max_points=500)
    assert payload["scene_id"] == "scene-x"
    assert len(payload["stems"]) == 500
    assert payload["fruits"][0]["position"] == [0.1, 0.2, 1.0]
    assert payload["eef_path"]

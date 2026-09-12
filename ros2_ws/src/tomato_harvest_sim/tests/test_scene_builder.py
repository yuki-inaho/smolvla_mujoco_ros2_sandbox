import json
from pathlib import Path

import numpy as np
import pytest
from paths import BUNDLE_DIR
from tomato_harvest_sim.scene_builder import (
    SCENE_ID,
    build_scene,
    locate_bundle,
    write_scene,
)


def _require_bundle() -> None:
    if not BUNDLE_DIR.exists():
        pytest.fail(f"required dataset bundle is missing: {BUNDLE_DIR}", pytrace=False)


def test_locate_bundle_returns_existing_paths():
    _require_bundle()
    paths = locate_bundle(str(BUNDLE_DIR), SCENE_ID)
    for key in ("feasibility", "trajectory", "perception", "tf", "pick_trigger", "joint_context"):
        assert Path(paths[key]).exists(), f"{key} path missing: {paths[key]}"


def test_locate_bundle_rejects_unknown_scene():
    _require_bundle()
    with pytest.raises(FileNotFoundError):
        locate_bundle(str(BUNDLE_DIR), "does-not-exist")


def test_build_scene_contains_required_keys_and_nonempty_cloud():
    _require_bundle()
    scene = build_scene(str(BUNDLE_DIR), SCENE_ID, max_frames=5)

    assert scene["scene_id"] == SCENE_ID
    assert len(scene["fruits"]) == 50
    assert len(scene["stems"]) == 3440
    for fruit in scene["fruits"]:
        assert np.all(np.isfinite(fruit["position"]))
        assert "ripeness" in fruit and "cluster" in fruit
    for point in scene["stems"][:10]:
        assert np.all(np.isfinite(point))


def test_build_scene_target_is_finite_and_world_frame():
    _require_bundle()
    scene = build_scene(str(BUNDLE_DIR), SCENE_ID, max_frames=3)
    target = scene["target"]
    np.testing.assert_allclose(target["position"], [0.3534742649644613, 0.5227577420258906, 1.275849107842993], atol=1e-9)


def test_build_scene_trajectory_and_eef_path_align():
    _require_bundle()
    scene = build_scene(str(BUNDLE_DIR), SCENE_ID, max_frames=3)
    trajectory = scene["trajectory"]
    assert len(trajectory) > 10
    assert len(trajectory) == len(scene["eef_path"])
    first = trajectory[0]
    assert first["time_from_start_s"] == 0.0
    assert set(first["joints"]) == {"lift", "shoulder_yaw", "elbow_yaw", "wrist_yaw"}
    assert len(first["eef"]) == 3
    np.testing.assert_allclose(scene["eef_path"][-1], [0.36347501053405695, 0.5377548170398302, 1.2758476170457242], atol=1e-6)


def test_build_scene_frames_resolve_to_real_files():
    _require_bundle()
    scene = build_scene(str(BUNDLE_DIR), SCENE_ID, max_frames=4)
    assert len(scene["frames"]) >= 1
    for frame in scene["frames"]:
        assert Path(frame["rgb"]).exists(), frame["rgb"]
        assert Path(frame["depth"]).exists(), frame["depth"]
        assert frame["rgb"].endswith("_camera_l_rgb.png")
        assert frame["depth"].endswith("_camera_l_depth.png")


def test_write_scene_creates_scene_json(tmp_path):
    _require_bundle()
    scene = build_scene(str(BUNDLE_DIR), SCENE_ID, max_frames=2)
    scene_path = write_scene(scene, str(tmp_path))
    assert scene_path == tmp_path / SCENE_ID / "scene.json"
    payload = json.loads(scene_path.read_text())
    assert payload["scene_id"] == SCENE_ID
    assert payload["frames"]

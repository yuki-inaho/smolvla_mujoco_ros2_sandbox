import json
from pathlib import Path

import numpy as np
import pytest
from paths import SCENE_PATH

DEPTH_SCALE = 0.001


def _scene():
    if not SCENE_PATH.exists():
        pytest.fail(
            f"scene.json is missing; run `pixi run build-scene` first: {SCENE_PATH}",
            pytrace=False,
        )
    return json.loads(SCENE_PATH.read_text())


def test_scene_frames_and_camera_info_are_broadcastable():
    scene = _scene()
    frames = scene["frames"]
    assert frames, "scene.json contains no frames"
    color = scene["camera_info"]["color"]
    depth = scene["camera_info"]["depth"]
    assert [color["width"], color["height"]] == [800, 600]
    assert [depth["width"], depth["height"]] == [640, 480]
    assert len(color["K"]) == 9 and len(depth["K"]) == 9
    assert color["frame_id"] == "camera_rgb_optical_frame"
    assert depth["frame_id"] == "camera_depth_optical_frame"
    for frame in frames:
        assert Path(frame["rgb"]).exists(), frame["rgb"]
        assert Path(frame["depth"]).exists(), frame["depth"]
        assert frame["rgb"].endswith("_camera_l_rgb.png")
        assert frame["depth"].endswith("_camera_l_depth.png")
        assert isinstance(frame["index"], int)


def test_depth_image_is_16bit_expected_shape():
    import cv2

    frame = _scene()["frames"][0]
    depth = cv2.imread(frame["depth"], cv2.IMREAD_UNCHANGED)
    assert depth is not None
    assert depth.dtype == np.uint16
    assert depth.shape == (480, 640)


def test_depth_to_points_returns_enough_points():
    from tomato_harvest_sim.rgbd_broadcaster_node import depth_to_points, load_depth

    scene = _scene()
    frame = scene["frames"][0]
    depth_k = scene["camera_info"]["depth"]["K"]
    depth = load_depth(frame["depth"], DEPTH_SCALE)
    points = depth_to_points(depth, depth_k, stride=4)
    assert points.ndim == 2 and points.shape[1] == 3
    assert points.shape[0] > 1000
    assert np.all(np.isfinite(points))

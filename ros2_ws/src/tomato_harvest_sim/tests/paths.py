"""Shared dataset/model paths for the tomato_harvest_sim tests (DRY)."""

from pathlib import Path

BUNDLE_DIR = Path("/home/kasm-user/Desktop/data/2026-08-04_tmt4-02")
SCENE_ID = "7a7e56ff__camera_l__2026-08-04_08-19-43-646"
RUN_DIR = BUNDLE_DIR / "grasp" / "runs" / SCENE_ID
RECORDING_DIR = BUNDLE_DIR / "grasp" / "recordings" / SCENE_ID
SCENE_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "scenes" / SCENE_ID / "scene.json"
)
MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "tomato_scara.xml"
TRAJECTORY_PATH = RUN_DIR / "trajectory.parquet"
RGB_DIR = BUNDLE_DIR / "standard" / "rgb"
DEPTH_DIR = BUNDLE_DIR / "standard" / "depth"
SNAPSHOT_PATH = BUNDLE_DIR / "config_snapshots" / "robot_snapshot_tmt4-02.toml"

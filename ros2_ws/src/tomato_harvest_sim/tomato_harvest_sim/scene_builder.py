"""Build a simulation-ready ``scene.json`` from a grasp recording bundle.

The bundle layout follows ``Desktop/data/2026-08-04_tmt4-02``::

    grasp/runs/<scene_id>/{feasibility,trajectory}.parquet
    grasp/recordings/<scene_id>/scene_perception_snapshot.parquet
    grasp/recordings/<scene_id>/{tf_snapshot,pick_trigger,joint_context}.parquet
    standard/{rgb,depth}/<uuid>__<timestamp>_<camera>_{rgb,depth}.png

The point cloud is expressed in the world frame using the recorded
``world_from_camera_capture`` transform, and the harvest target / trajectory
come from the grasp run artifacts (the single source of truth).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tomato_harvest_sim.sim_common import (
    load_numeric_series,
    make_transform,
    transform_points,
)

SCENE_ID = "7a7e56ff__camera_l__2026-08-04_08-19-43-646"
SCENE_CAMERA = "camera_l"

TRAJECTORY_COLUMNS = [
    "time_from_start_s",
    "joint_lift_rad",
    "joint_shoulder_yaw_rad",
    "joint_elbow_yaw_rad",
    "joint_wrist_yaw_rad",
    "eef_x_m",
    "eef_y_m",
    "eef_z_m",
]

JOINT_COLUMN_TO_NAME = {
    "joint_lift_rad": "lift",
    "joint_shoulder_yaw_rad": "shoulder_yaw",
    "joint_elbow_yaw_rad": "elbow_yaw",
    "joint_wrist_yaw_rad": "wrist_yaw",
}


def locate_bundle(bundle_dir: str, scene_id: str) -> dict[str, str]:
    """Resolve the parquet files that make up a single grasp scene."""
    root = Path(bundle_dir)
    run_dir = root / "grasp" / "runs" / scene_id
    rec_dir = root / "grasp" / "recordings" / scene_id
    paths = {
        "feasibility": run_dir / "feasibility.parquet",
        "trajectory": run_dir / "trajectory.parquet",
        "perception": rec_dir / "scene_perception_snapshot.parquet",
        "tf": rec_dir / "tf_snapshot.parquet",
        "pick_trigger": rec_dir / "pick_trigger.parquet",
        "joint_context": rec_dir / "joint_context.parquet",
        "rgb_dir": root / "standard" / "rgb",
        "depth_dir": root / "standard" / "depth",
        "camera_params_dir": root / "standard" / "camera_parameters",
        "run_manifest": run_dir / "run_manifest.json",
    }
    for key in ("feasibility", "trajectory", "perception", "tf", "pick_trigger", "joint_context"):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"scene bundle component missing: {paths[key]}")
    return {key: str(value) for key, value in paths.items()}


def _read_rows(path: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    import duckdb

    connection = duckdb.connect()
    try:
        cursor = connection.execute("SELECT * FROM read_parquet(?)", [path])
        names = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
    finally:
        connection.close()
    return names, rows


def _row_dict(path: str) -> dict[str, Any]:
    names, rows = _read_rows(path)
    if not rows:
        raise ValueError(f"expected at least one row in {path}")
    return dict(zip(names, rows[0]))


def _reshape(shape: Sequence[int], data: Sequence[float]) -> np.ndarray:
    return np.asarray(data, dtype=np.float64).reshape(tuple(int(v) for v in shape))


def _capture_transform(tf_row: dict[str, Any]) -> np.ndarray:
    translation = [
        tf_row["world_from_camera_capture__translation__x"],
        tf_row["world_from_camera_capture__translation__y"],
        tf_row["world_from_camera_capture__translation__z"],
    ]
    quat = [
        tf_row["world_from_camera_capture__rotation__x"],
        tf_row["world_from_camera_capture__rotation__y"],
        tf_row["world_from_camera_capture__rotation__z"],
        tf_row["world_from_camera_capture__rotation__w"],
    ]
    return make_transform(translation, quat)


def _load_camera_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text())
    return {
        "width": payload.get("width"),
        "height": payload.get("height"),
        "K": payload.get("K"),
        "P": payload.get("P"),
        "frame_id": payload.get("header", {}).get("frame_id"),
    }


def _collect_frames(
    rgb_dir: Path,
    depth_dir: Path,
    camera: str,
    scene_timestamp: str,
    max_frames: int,
) -> list[dict[str, Any]]:
    pattern = re.compile(
        rf"__(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}-\d{{3}})_{camera}_rgb\.png$"
    )
    pairs: list[tuple[str, Path, Path]] = []
    for rgb_path in sorted(rgb_dir.glob(f"*_{camera}_rgb.png")):
        match = pattern.search(rgb_path.name)
        if not match:
            continue
        depth_path = depth_dir / rgb_path.name.replace("_rgb.png", "_depth.png")
        if not depth_path.exists():
            continue
        pairs.append((match.group("ts"), rgb_path, depth_path))
    if not pairs:
        raise FileNotFoundError(f"no {camera} rgb/depth frames found under {rgb_dir}")
    pairs.sort(key=lambda item: item[0])

    timestamps = [item[0] for item in pairs]
    start = timestamps.index(scene_timestamp) if scene_timestamp in timestamps else 0
    count = max(1, min(int(max_frames), len(pairs)))
    frames: list[dict[str, Any]] = []
    for offset in range(count):
        timestamp, rgb_path, depth_path = pairs[(start + offset) % len(pairs)]
        frames.append(
            {
                "index": (start + offset) % len(pairs),
                "timestamp": timestamp,
                "rgb": str(rgb_path),
                "depth": str(depth_path),
            }
        )
    return frames


def build_scene(bundle_dir: str, scene_id: str = SCENE_ID, max_frames: int = 20) -> dict[str, Any]:
    """Assemble a scene dictionary from the recorded grasp artifacts."""
    paths = locate_bundle(bundle_dir, scene_id)

    tf_row = _row_dict(paths["tf"])
    transform = _capture_transform(tf_row)

    perception = _row_dict(paths["perception"])
    fruit_points = _reshape(perception["fruit_points__shape"], perception["fruit_points__data"])
    fruits_world = transform_points(fruit_points, transform)
    ripeness = list(perception["fruit_ripeness_labels__data"])
    clusters = list(perception["fruit_cluster_labels__data"])

    stem_points = _reshape(perception["stem_points__shape"], perception["stem_points__data"])
    stems_world = transform_points(stem_points, transform)

    pick = _row_dict(paths["pick_trigger"])
    target = {
        "position": [float(pick["target__x"]), float(pick["target__y"]), float(pick["target__z"])],
        "trigger_id": pick["trigger_id"],
        "mode": pick["mode"],
        "source": pick["source"],
        "stamp_ns": int(pick["stamp_ns"]),
    }

    trajectory = _load_trajectory(paths["trajectory"])

    home_joints: list[float] = []
    manifest_path = Path(paths["run_manifest"])
    if manifest_path.exists():
        home_joints = list(json.loads(manifest_path.read_text()).get("home_joints_rad", []))

    if "__" in scene_id:
        parts = scene_id.split("__")
        scene_timestamp = parts[-1]
    else:
        scene_timestamp = ""

    frames = _collect_frames(
        Path(paths["rgb_dir"]),
        Path(paths["depth_dir"]),
        SCENE_CAMERA,
        scene_timestamp,
        max_frames,
    )

    camera_params = Path(paths["camera_params_dir"])
    return {
        "scene_id": scene_id,
        "camera": SCENE_CAMERA,
        "source": {key: value for key, value in paths.items()},
        "camera_info": {
            "color": _load_camera_info(camera_params / "rgb_camera_param.yaml"),
            "depth": _load_camera_info(camera_params / "depth_camera_param.yaml"),
        },
        "home_joints": home_joints,
        "frames": frames,
        "fruits": [
            {
                "position": fruits_world[index].tolist(),
                "ripeness": int(ripeness[index]),
                "cluster": int(clusters[index]),
            }
            for index in range(fruits_world.shape[0])
        ],
        "stems": stems_world.tolist(),
        "target": target,
        "trajectory": trajectory,
        "eef_path": [entry["eef"] for entry in trajectory],
    }


def _load_trajectory(path: str) -> list[dict[str, Any]]:
    series = load_numeric_series(path, TRAJECTORY_COLUMNS)
    count = series["time_from_start_s"].shape[0]
    trajectory: list[dict[str, Any]] = []
    for index in range(count):
        trajectory.append(
            {
                "time_from_start_s": float(series["time_from_start_s"][index]),
                "joints": {
                    name: float(series[column][index])
                    for column, name in JOINT_COLUMN_TO_NAME.items()
                },
                "eef": [
                    float(series["eef_x_m"][index]),
                    float(series["eef_y_m"][index]),
                    float(series["eef_z_m"][index]),
                ],
            }
        )
    return trajectory


def write_scene(scene: dict[str, Any], out_dir: str) -> Path:
    """Write ``scene.json`` under ``<out_dir>/<scene_id>/`` and return its path."""
    scene_dir = Path(out_dir) / scene["scene_id"]
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scene_dir / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2))
    return scene_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a tomato harvest scene.json")
    parser.add_argument(
        "--bundle-dir",
        default="/home/kasm-user/Desktop/data/2026-08-04_tmt4-02",
        help="extracted grasp bundle root",
    )
    parser.add_argument("--scene-id", default=SCENE_ID)
    parser.add_argument("--out", default="assets/scenes")
    parser.add_argument("--max-frames", type=int, default=20)
    args = parser.parse_args(argv)

    scene = build_scene(args.bundle_dir, args.scene_id, max_frames=args.max_frames)
    scene_path = write_scene(scene, args.out)
    print(
        f"wrote {scene_path} "
        f"(fruits={len(scene['fruits'])}, stems={len(scene['stems'])}, "
        f"trajectory={len(scene['trajectory'])}, frames={len(scene['frames'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render a real-time MuJoCo video of the tomato harvest simulation.

Uses the same motion contract as ``harvest_sim_node`` (recorded approach,
gripper close, reverse-playback lift, config-faithful dunk every N fruits), so
the clip shows the real joint targets, limits and dunk poses.

Scene elements drawn:
  - URDF-faithful SCARA arm (shoulder at 0.960 + lift, 0.285/0.345 m links,
    0.090x0.090x0.260 m hand, grasp point 0.070 m below the arm plane)
  - recognized fruits (TPE ripeness: 1=red / 0=green)
  - recognized stems (downsampled points)
  - RGB-D point cloud (depth PNG -> world frame via tf_snapshot)
  - pipe / gutter support (inferred from planning_params: height_pipe=1.30,
    target_y_max=0.58; explicit cylinder_array.parquet is empty for this scene)

Simulation time advances frame-by-frame at ``--fps``, so the output video
plays in real time (1 s of video = 1 s of robot motion).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2_ws" / "src" / "tomato_harvest_sim"
sys.path.insert(0, str(PKG))

from tomato_harvest_sim.harvest_sim_node import (
    JOINT_NAMES,
    build_cycle_schedule,
    motion_state_for_elapsed,
    schedule_duration,
)
from tomato_harvest_sim.rgbd_broadcaster_node import depth_to_points, load_depth
from tomato_harvest_sim.sim_common import make_transform, transform_points

DEFAULT_SCENE = (
    PKG / "assets" / "scenes" / "7a7e56ff__camera_l__2026-08-04_08-19-43-646" / "scene.json"
)
DEFAULT_MODEL = PKG / "assets" / "tomato_scara.xml"
DEFAULT_RECORDING = Path(
    "/home/kasm-user/Desktop/data/2026-08-04_tmt4-02/grasp/recordings/"
    "7a7e56ff__camera_l__2026-08-04_08-19-43-646"
)
DEFAULT_DEPTH = Path(
    "/home/kasm-user/Desktop/data/2026-08-04_tmt4-02/standard/depth/"
    "7a7e56ff-155e-46ae-9fc8-e4d1728af7db__2026-08-04_08-19-43-646_camera_l_depth.png"
)

COLOR_FRUIT_RED = "0.95 0.12 0.10 1"
COLOR_FRUIT_GREEN = "0.25 0.75 0.20 1"
COLOR_STEM = "0.10 0.55 0.15 1"
COLOR_CLOUD = "0.35 0.55 0.95 0.45"
COLOR_PIPE = "0.62 0.62 0.66 1"
COLOR_GUTTER = "0.45 0.42 0.38 0.35"


def depth_cloud_world(depth_path: Path, k, tf, scale: float, stride: int) -> np.ndarray:
    """Depth PNG -> camera 3D points -> world, reusing the broadcaster helpers."""
    depth = load_depth(str(depth_path), scale)
    points = depth_to_points(depth, k, stride)
    transform = make_transform(
        [tf["tx"], tf["ty"], tf["tz"]],
        [tf["qx"], tf["qy"], tf["qz"], tf["qw"]],
    )
    return transform_points(points, transform)


def _probe_video(path: Path) -> dict[str, float]:
    """Read duration / frame count / size, raising instead of guessing."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_frames,width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (rc={probe.returncode}): {probe.stderr.strip() or path}"
        )
    try:
        payload = json.loads(probe.stdout)
        duration = float((payload.get("format") or {}).get("duration"))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"ffprobe returned unusable metadata: {probe.stdout!r}") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise RuntimeError(f"ffprobe reported a non-positive duration: {duration}")
    stream = (payload.get("streams") or [{}])[0]
    return {
        "duration": duration,
        "frames": float(stream.get("nb_frames") or 0),
        "width": float(stream.get("width") or 0),
        "height": float(stream.get("height") or 0),
    }


def _verify_video(
    path: Path,
    expected_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Gate before publishing: full decode plus expected frame count/size."""
    info = _probe_video(path)
    if expected_frames is not None and int(info["frames"]) != int(expected_frames):
        raise RuntimeError(
            f"frame count {int(info['frames'])} != expected {expected_frames} ({path})"
        )
    if width is not None and int(info["width"]) != int(width):
        raise RuntimeError(f"width {int(info['width'])} != expected {width} ({path})")
    if height is not None and int(info["height"]) != int(height):
        raise RuntimeError(f"height {int(info['height'])} != expected {height} ({path})")
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if decode.returncode != 0:
        raise RuntimeError(
            f"decode failed (rc={decode.returncode}): {decode.stderr.strip()[-400:]}"
        )


@contextlib.contextmanager
def exclusive_output(
    out: Path,
    expected_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
):
    """Serialize writers on ``out`` and publish atomically after verification.

    Yields a unique temp path to render into. Concurrent renders fail fast via
    an exclusive lock; the temp file must decode fully with the expected frame
    count/size and is only then moved onto ``out``, so a reader never observes
    a partially written video. Stale temp files from killed writers are
    removed once the lock is held.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out.with_suffix(out.suffix + ".lock")
    with open(lock_path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"another render is already writing {out} (lock: {lock_path})"
            ) from exc
        for stale in sorted(out.parent.glob(f"{out.stem}.partial-*{out.suffix}")):
            stale.unlink(missing_ok=True)
        temp_path = out.with_name(f"{out.stem}.partial-{os.getpid()}{out.suffix}")
        try:
            yield temp_path
            _verify_video(temp_path, expected_frames, width, height)
            os.replace(temp_path, out)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_tf(recording: Path) -> dict:
    import duckdb

    path = recording / "tf_snapshot.parquet"
    row = duckdb.sql(
        f"""
        select world_from_camera_capture__translation__x,
               world_from_camera_capture__translation__y,
               world_from_camera_capture__translation__z,
               world_from_camera_capture__rotation__x,
               world_from_camera_capture__rotation__y,
               world_from_camera_capture__rotation__z,
               world_from_camera_capture__rotation__w
        from read_parquet('{path}')
        """
    ).fetchone()
    return dict(zip(["tx", "ty", "tz", "qx", "qy", "qz", "qw"], [float(v) for v in row]))


def load_pipe_contract(run_dir: Path) -> dict:
    """Pipe values actually used by the planner for this scene.

    ``pipe_dist_m`` / ``pipe_height_m`` come from the recognition artifact when
    available, otherwise from the config defaults (pipe_source records which).
    Conventions from pipe_recognizer_client.py: dist = abs(pipe_pos.y) in the
    world/arm frame, height = pipe_pos.z (pipe top).
    """
    import duckdb

    path = run_dir / "feasibility.parquet"
    row = duckdb.sql(
        f"""
        select pipe_source, pipe_dist_m, pipe_height_m,
               target_world_x_m, target_world_y_m, target_world_z_m,
               target_z_min_m, target_z_max_m, hand_gutter_collision
        from read_parquet('{path}')
        """
    ).fetchone()
    keys = [
        "pipe_source",
        "pipe_dist_m",
        "pipe_height_m",
        "target_x",
        "target_y",
        "target_z",
        "target_z_min",
        "target_z_max",
        "hand_gutter_collision",
    ]
    return dict(zip(keys, row))


def sphere(pos, radius: float, rgba: str) -> str:
    return (
        f'<geom type="sphere" size="{radius}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
        f'rgba="{rgba}" contype="0" conaffinity="0" group="1"/>'
    )


def build_scene_xml(scene: dict, cloud: np.ndarray | None, args, pipe: dict) -> str:
    xml = args.model.read_text()
    xml = xml.replace(
        'offwidth="640" offheight="480"',
        f'offwidth="{args.width}" offheight="{args.height}"',
    )
    inserts = []
    stems = np.asarray(scene["stems"], dtype=float)[:: args.stride_stems]
    for point in stems:
        inserts.append(sphere(point, args.stem_radius, COLOR_STEM))
    for fruit in scene["fruits"]:
        color = COLOR_FRUIT_RED if int(fruit["ripeness"]) == 1 else COLOR_FRUIT_GREEN
        inserts.append(sphere(fruit["position"], args.fruit_radius, color))
    if cloud is not None and len(cloud):
        for point in cloud[:: args.cloud_step]:
            inserts.append(sphere(point, args.cloud_radius, COLOR_CLOUD))

    # Pipe: axis at y = +/-dist_pipe (world/arm frame), top at pipe_height.
    side = 1.0 if float(pipe["target_y"]) >= 0.0 else -1.0
    pipe_y = side * float(pipe["pipe_dist_m"])
    pipe_top = float(pipe["pipe_height_m"])
    pipe_radius = 0.024  # greenhouse heating pipe OD ~48 mm
    pipe_z = pipe_top - pipe_radius
    inserts.append(
        f'<geom name="pipe" type="cylinder" fromto="-1.1 {pipe_y:.4f} {pipe_z:.4f} 1.1 {pipe_y:.4f} {pipe_z:.4f}" '
        f'size="{pipe_radius}" rgba="{COLOR_PIPE}" contype="0" conaffinity="0" group="1"/>'
    )
    # Crop trough: top = pipe_height - height_pipe_from_gutter (0.65 m, planning params).
    # Width/height are INFERRED (no drawing in repo); the top level is documented.
    trough_top = pipe_top - 0.65
    trough_y = pipe_y
    half_w, wall = 0.12, 0.012
    depth = 0.12
    inserts.append(
        f'<geom name="trough_bottom" type="box" size="1.1 {half_w} 0.012" pos="0 {trough_y:.4f} {trough_top - depth:.4f}" rgba="{COLOR_GUTTER}" contype="0" conaffinity="0" group="1"/>'
    )
    for sign in (+1.0, -1.0):
        inserts.append(
            f'<geom name="trough_wall_{"p" if sign > 0 else "n"}" type="box" size="1.1 {wall} {depth / 2:.4f}" '
            f'pos="0 {trough_y + sign * half_w:.4f} {trough_top - depth / 2:.4f}" rgba="{COLOR_GUTTER}" contype="0" conaffinity="0" group="1"/>'
        )
    return xml.replace("</worldbody>", "\n".join(inserts) + "\n</worldbody>")


def joint_map(mujoco, model):
    mapping = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint not found: {name}")
        mapping[name] = int(model.jnt_qposadr[jid])
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "video" / "harvest_realtime.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=30.0, help="video seconds (= sim seconds)")
    parser.add_argument("--dunk-every", type=int, default=3, help="demo frequency; real limit is 8")
    parser.add_argument("--phase", default=None, help="start phase offset in seconds")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stride-stems", type=int, default=4)
    parser.add_argument("--cloud-step", type=int, default=14)
    parser.add_argument("--cloud-stride", type=int, default=2)
    parser.add_argument("--stem-radius", type=float, default=0.005)
    parser.add_argument("--fruit-radius", type=float, default=0.028)
    parser.add_argument("--cloud-radius", type=float, default=0.004)
    parser.add_argument(
        "--pipe-run",
        type=Path,
        default=Path(
            "/home/kasm-user/Desktop/data/2026-08-04_tmt4-02/grasp/runs/"
            "7a7e56ff__camera_l__2026-08-04_08-19-43-646"
        ),
        help="run dir whose feasibility.parquet holds the used pipe values",
    )
    parser.add_argument("--camera-yaw", type=float, default=1.5708, help="camera_motor angle (left=+90deg)")
    parser.add_argument("--azimuth", type=float, default=125.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--lookat", type=float, nargs=3, default=[0.45, 0.40, 1.05])
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    import mujoco

    scene = json.loads(args.scene.read_text())
    pipe = load_pipe_contract(args.pipe_run)
    print(
        f"pipe contract: source={pipe['pipe_source']} dist={float(pipe['pipe_dist_m']):.4f} m "
        f"height={float(pipe['pipe_height_m']):.4f} m target_y={float(pipe['target_y']):.4f} "
        f"hand_gutter_collision={pipe['hand_gutter_collision']}",
        flush=True,
    )
    tf = load_tf(args.recording)
    cloud = None
    if args.cloud_step > 0:
        cloud = depth_cloud_world(
            args.depth, scene["camera_info"]["depth"]["K"], tf, 0.001, args.cloud_stride
        )
    xml = build_scene_xml(scene, cloud, args, pipe)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    qpos = joint_map(mujoco, model)
    camera_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "camera_yaw")
    camera_qpos = int(model.jnt_qposadr[camera_joint]) if camera_joint >= 0 else None
    target_mocap = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_fruit")])

    trajectory = scene["trajectory"]
    times = np.array([entry["time_from_start_s"] for entry in trajectory])
    joint_values = np.array(
        [
            [
                entry["joints"]["lift"],
                entry["joints"]["shoulder_yaw"],
                entry["joints"]["elbow_yaw"],
                entry["joints"]["wrist_yaw"],
            ]
            for entry in trajectory
        ]
    )
    trajectory_duration = float(times[-1])
    schedule = build_cycle_schedule(trajectory_duration, args.dunk_every, cycles=max(8, args.dunk_every))
    block = schedule_duration(schedule)
    target = np.array(scene["target"]["position"], dtype=float)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = args.lookat
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    renderer = mujoco.Renderer(model, args.height, args.width)

    def step_to(elapsed_total: float) -> str:
        state = motion_state_for_elapsed(elapsed_total, schedule, times, joint_values, trajectory_duration)
        data.qpos[qpos["lift"]] = state.joints[0]
        data.qpos[qpos["shoulder_yaw"]] = state.joints[1]
        data.qpos[qpos["elbow_yaw"]] = state.joints[2]
        data.qpos[qpos["wrist_yaw"]] = state.joints[3]
        data.qpos[qpos["gripper"]] = state.gripper
        if camera_qpos is not None:
            data.qpos[camera_qpos] = args.camera_yaw
        data.mocap_pos[target_mocap] = target
        mujoco.mj_forward(model, data)
        return state.phase

    offset = float(args.phase) if args.phase is not None else 0.0

    if args.preview is not None:
        phase = step_to((offset + 0.55 * block) % block)
        renderer.update_scene(data, camera=camera)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        cv2.imwrite(str(args.preview), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        print(f"preview written: {args.preview} (phase={phase})")
        return 0

    n_frames = int(args.duration * args.fps)
    with exclusive_output(args.out, n_frames, args.width, args.height) as temp_out:
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_out),
        ]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for index in range(n_frames):
            elapsed = offset + index / args.fps
            phase = step_to(elapsed % block)
            renderer.update_scene(data, camera=camera)
            proc.stdin.write(renderer.render().tobytes())
            if index % 60 == 0:
                print(f"frame {index}/{n_frames} ({index / args.fps:.1f}s, phase={phase})", flush=True)
        proc.stdin.close()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with rc={code}")
    print(f"video written: {args.out} ({n_frames} frames, {n_frames / args.fps:.1f}s, block={block:.2f}s, rc=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

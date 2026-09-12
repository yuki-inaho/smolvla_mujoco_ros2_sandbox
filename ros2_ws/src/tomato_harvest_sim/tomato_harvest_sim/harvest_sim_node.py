"""MuJoCo-based tomato harvest simulation node.

Motion follows the real machine contracts:

* approach  : replay of the recorded (planned) joint trajectory home->grasp
* close     : gripper semiopen 0.09 m -> closed 0.0057 m (single-jaw stroke 0.0843 m)
* lift      : reverse replay of the recorded trajectory (grasp->home, fruit held)
* unload    : every ``dunk_every`` fruits (real limit ``max_fruits_in_hand = 8``)
              a dunk sequence home->dunk_left_5 (arm_positions.toml) with
              open -> 2x shake (0.024 m @ 0.42 m/s) -> settle 0.3 s -> return.
              Other cycles just stow the fruit (<= 8 in hand).
* done      : short hold, then the next cycle.

MuJoCo/scene failures are surfaced as an error status and then raised: there
is no implicit mock fallback.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from tomato_harvest_sim.sim_common import sample_trajectory

JOINT_NAMES = ("lift", "shoulder_yaw", "elbow_yaw", "wrist_yaw", "gripper")

# Real machine constants (robot_snapshot_tmt4-02.toml / arm_positions.toml / arm_constants.py)
HOME_JOINTS = (0.30, math.radians(25.0), 0.0, math.radians(-25.0))
DUNK_JOINTS = (0.003, math.radians(34.0), math.radians(-35.0), math.radians(1.0))  # dunk_left_5
JOINT_VMAX = (0.5, 6.0, 6.0, 6.0)
JOINT_AMAX = (8.0, 40.0, 40.0, 40.0)
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.0843  # equivalent single-jaw stroke of the symmetric twin belts
# (surface gap 0.090 m semiopen -> 0.0057 m closed; per-belt travel 0.04215 m)
CLOSE_DURATION_S = 0.4
UNLOAD_HOLD_S = 0.3  # non-dunk cycle: fruit stays in hand (max_fruits_in_hand = 8)
DONE_HOLD_S = 0.2
DUNK_SHAKE_COUNT = 2  # down->up round trips
DUNK_SHAKE_AMPLITUDE = 0.024
DUNK_SHAKE_VELOCITY = 0.42
DUNK_SHAKE_ACCEL = 8.0
DUNK_SETTLE_S = 0.3
DUNK_RELEASE_S = 0.4
PICKED_SOURCE = "phase_state_machine"  # self-reported; never a verified label

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = (
    PACKAGE_ROOT
    / "assets"
    / "scenes"
    / "7a7e56ff__camera_l__2026-08-04_08-19-43-646"
    / "scene.json"
)
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "assets" / "tomato_scara.xml"


def is_picked(phase: str) -> bool:
    """True once the gripper has closed on the fruit."""
    return phase in ("LIFT", "UNLOAD", "DONE")


def trapezoid_time(delta: float, vmax: float, amax: float) -> float:
    """Minimum time for a trapezoidal (or triangular) profile of ``delta``."""
    distance = abs(delta)
    if distance == 0.0:
        return 0.0
    ramp_time = vmax / amax
    ramp_distance = 0.5 * amax * ramp_time * ramp_time
    if distance <= 2.0 * ramp_distance:
        return 2.0 * math.sqrt(distance / amax)
    return 2.0 * ramp_time + (distance - 2.0 * ramp_distance) / vmax


def _trapezoid_position(delta: float, t: float, duration: float, vmax: float, amax: float) -> float:
    """Signed position along a trapezoid fitted to ``duration``.

    ``duration`` is authoritative (synchronised multi-joint move). The peak
    velocity is solved from ``T = d/v + v/a`` (the smaller root), so the
    profile respects both ``vmax`` and ``amax`` for any ``duration`` returned
    by :func:`joint_move_duration` and stays continuous.
    """
    sign = 1.0 if delta >= 0 else -1.0
    distance = abs(delta)
    if distance == 0.0:
        return delta
    if duration <= 0.0:
        raise ValueError(
            f"infeasible duration {duration} for |delta|={distance:.6f}"
        )
    minimum = trapezoid_time(delta, vmax, amax)
    if duration + 1e-9 * (1.0 + minimum) < minimum:
        raise ValueError(
            f"infeasible duration {duration:.6f}s for |delta|={distance:.6f} "
            f"(minimum {minimum:.6f}s at vmax={vmax}, amax={amax})"
        )
    t = min(max(t, 0.0), duration)
    discriminant = (duration * amax) ** 2 - 4.0 * distance * amax
    peak = (duration * amax - math.sqrt(max(discriminant, 0.0))) / 2.0
    if peak <= 0.0:
        return delta
    peak = min(peak, vmax)
    ramp_time = peak / amax
    ramp_distance = 0.5 * amax * ramp_time * ramp_time
    if t < ramp_time:
        return sign * 0.5 * amax * t * t
    if t > duration - ramp_time:
        return sign * (distance - 0.5 * amax * (duration - t) ** 2)
    return sign * (ramp_distance + peak * (t - ramp_time))


def joint_move_duration(start: Sequence[float], target: Sequence[float]) -> float:
    """Synchronised minimum time to move all joints from start to target."""
    return max(
        trapezoid_time(target[i] - start[i], JOINT_VMAX[i], JOINT_AMAX[i])
        for i in range(4)
    )


def joint_position(
    start: Sequence[float],
    target: Sequence[float],
    t: float,
    duration: float | None = None,
) -> tuple[float, float, float, float]:
    """Interpolate a synchronised joint move with real velocity/accel limits."""
    if duration is None:
        duration = joint_move_duration(start, target)
    return tuple(
        start[i] + _trapezoid_position(target[i] - start[i], t, duration, JOINT_VMAX[i], JOINT_AMAX[i])
        for i in range(4)
    )


def dunk_shake_duration() -> float:
    """Duration of the configured lift shake (N round trips of the stroke)."""
    return DUNK_SHAKE_COUNT * 2.0 * trapezoid_time(
        DUNK_SHAKE_AMPLITUDE, DUNK_SHAKE_VELOCITY, DUNK_SHAKE_ACCEL
    )


def dunk_phase_duration() -> float:
    """Full dunk sequence: move -> open -> shake -> settle -> return."""
    move = joint_move_duration(HOME_JOINTS, DUNK_JOINTS)
    return 2.0 * move + DUNK_RELEASE_S + dunk_shake_duration() + DUNK_SETTLE_S


def _shake_offset(t: float) -> float:
    """Lift offset during the dunk shake; starts and ends at 0."""
    step = trapezoid_time(DUNK_SHAKE_AMPLITUDE, DUNK_SHAKE_VELOCITY, DUNK_SHAKE_ACCEL)
    if step <= 0.0:
        return 0.0
    steps = DUNK_SHAKE_COUNT * 2
    index = int(t // step)
    if index >= steps:
        return 0.0
    local = t - index * step
    position = _trapezoid_position(
        DUNK_SHAKE_AMPLITUDE, local, step, DUNK_SHAKE_VELOCITY, DUNK_SHAKE_ACCEL
    )
    return position if index % 2 == 0 else DUNK_SHAKE_AMPLITUDE - position


@dataclass(frozen=True)
class MotionState:
    phase: str
    joints: tuple[float, float, float, float]
    gripper: float


def build_cycle_schedule(trajectory_duration_s: float, dunk_every: int, cycles: int = 8) -> list[dict[str, Any]]:
    """One repeating block of cycles with per-cycle phase durations."""
    dunk_duration = dunk_phase_duration()
    schedule = []
    for index in range(max(cycles, 1)):
        is_dunk = dunk_every > 0 and (index + 1) % dunk_every == 0
        phases = (
            ("APPROACH", trajectory_duration_s, False),
            ("CLOSE", CLOSE_DURATION_S, False),
            ("LIFT", trajectory_duration_s, False),
            ("UNLOAD", dunk_duration if is_dunk else UNLOAD_HOLD_S, is_dunk),
            ("DONE", DONE_HOLD_S, False),
        )
        schedule.append(
            {
                "index": index,
                "is_dunk": is_dunk,
                "duration": sum(item[1] for item in phases),
                "phases": phases,
            }
        )
    return schedule


def schedule_duration(schedule: Sequence[dict[str, Any]]) -> float:
    return sum(cycle["duration"] for cycle in schedule)


def motion_state_for_elapsed(
    elapsed_total: float,
    schedule: Sequence[dict[str, Any]],
    trajectory_times: np.ndarray,
    trajectory_values: np.ndarray,
    trajectory_duration_s: float,
) -> MotionState:
    """Phase + joint/gripper targets at a (possibly wrapped) elapsed time."""
    block = schedule_duration(schedule)
    if block <= 0.0:
        raise ValueError("schedule must have a positive duration")
    t = elapsed_total % block
    cycle = schedule[-1]
    for candidate in schedule:
        if t < candidate["duration"]:
            cycle = candidate
            break
        t -= candidate["duration"]
    dunk_move = joint_move_duration(HOME_JOINTS, DUNK_JOINTS)

    for phase, duration, is_dunk in cycle["phases"]:
        if t >= duration:
            t -= duration
            continue
        if phase == "APPROACH":
            values = sample_trajectory(trajectory_times, trajectory_values, t)
            return MotionState("APPROACH", tuple(float(v) for v in values), GRIPPER_OPEN)
        if phase == "CLOSE":
            last = sample_trajectory(trajectory_times, trajectory_values, trajectory_duration_s)
            ratio = t / duration if duration > 0 else 1.0
            gripper = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * min(max(ratio, 0.0), 1.0)
            return MotionState("CLOSE", tuple(float(v) for v in last), gripper)
        if phase == "LIFT":
            playback = max(trajectory_duration_s - t, 0.0)
            values = sample_trajectory(trajectory_times, trajectory_values, playback)
            return MotionState("LIFT", tuple(float(v) for v in values), GRIPPER_CLOSED)
        if phase == "UNLOAD":
            if not is_dunk:
                return MotionState("UNLOAD", HOME_JOINTS, GRIPPER_CLOSED)
            if t < dunk_move:
                joints = joint_position(HOME_JOINTS, DUNK_JOINTS, t, dunk_move)
                return MotionState("UNLOAD", joints, GRIPPER_CLOSED)
            t -= dunk_move
            if t < DUNK_RELEASE_S:
                ratio = t / DUNK_RELEASE_S if DUNK_RELEASE_S > 0 else 1.0
                gripper = GRIPPER_CLOSED + (GRIPPER_OPEN - GRIPPER_CLOSED) * min(max(ratio, 0.0), 1.0)
                return MotionState("UNLOAD", DUNK_JOINTS, gripper)
            t -= DUNK_RELEASE_S
            shake = dunk_shake_duration()
            if t < shake:
                joints = list(DUNK_JOINTS)
                joints[0] = max(DUNK_JOINTS[0] + _shake_offset(t), 0.0)
                return MotionState("UNLOAD", tuple(joints), GRIPPER_OPEN)
            t -= shake
            if t < DUNK_SETTLE_S:
                return MotionState("UNLOAD", DUNK_JOINTS, GRIPPER_OPEN)
            t -= DUNK_SETTLE_S
            joints = joint_position(DUNK_JOINTS, HOME_JOINTS, t, dunk_move)
            return MotionState("UNLOAD", joints, GRIPPER_OPEN)
        # DONE: after a dunk the gripper is already open; otherwise the jaws
        # release the stem into the hand channel smoothly over the done hold.
        if cycle["is_dunk"]:
            return MotionState("DONE", HOME_JOINTS, GRIPPER_OPEN)
        ratio = t / duration if duration > 0 else 1.0
        gripper = GRIPPER_CLOSED + (GRIPPER_OPEN - GRIPPER_CLOSED) * min(max(ratio, 0.0), 1.0)
        return MotionState("DONE", HOME_JOINTS, gripper)


def load_scene(path: str) -> dict[str, Any]:
    """Load a generated ``scene.json``."""
    scene_path = Path(path)
    if not scene_path.exists():
        raise FileNotFoundError(f"scene file not found: {scene_path}")
    return json.loads(scene_path.read_text())


def build_status_payload(
    *,
    phase: str,
    target: Sequence[float],
    eef: Sequence[float],
    frame_index: int,
    frames_total: int,
    elapsed_s: float,
    trajectory_duration_s: float,
    dunk_every: int,
    joints: dict[str, float],
    error: str | None,
    target_source: str | None = None,
    picked_verified: bool = False,
    verified_source: str | None = None,
) -> dict[str, Any]:
    """Status payload with explicit provenance for self-reported fields.

    ``picked`` comes from the phase machine and is therefore only
    ``picked_source=phase_state_machine``. ``picked_verified`` may only be set
    by a caller that provides a ``verified_source``, and never for a phase in
    which nothing was picked, so the payload cannot silently claim success.
    """
    picked = is_picked(phase)
    if picked_verified and not picked:
        raise ValueError(
            f"picked_verified cannot be true in phase '{phase}' (nothing picked)"
        )
    if picked_verified and not verified_source:
        raise ValueError("picked_verified requires an explicit verified_source")
    return {
        "phase": phase,
        "target": list(target),
        "eef": list(eef),
        "picked": picked,
        "picked_source": PICKED_SOURCE,
        "picked_verified": bool(picked_verified),
        "verified_source": verified_source,
        "target_source": target_source or "unknown",
        "frame_index": frame_index,
        "frames_total": frames_total,
        "elapsed_s": elapsed_s,
        "trajectory_duration_s": trajectory_duration_s,
        "dunk_every": dunk_every,
        "joints": joints,
        "error": error,
    }


class HarvestSimNode(Node):
    def __init__(self) -> None:
        super().__init__("harvest_sim_node")

        self.declare_parameter("scene_path", str(DEFAULT_SCENE_PATH))
        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("time_scale", 1.0)
        self.declare_parameter("frames_per_sec", 2.0)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("status_topic", "/sim/status")
        self.declare_parameter("loop", True)
        self.declare_parameter("dunk_every", 8)

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.time_scale = float(self.get_parameter("time_scale").value)
        self.frames_per_sec = float(self.get_parameter("frames_per_sec").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.dunk_every = int(self.get_parameter("dunk_every").value)

        joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)

        self.joint_pub = self.create_publisher(JointState, joint_state_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)

        self.error: str | None = None
        self._load_scene_and_model()

        self._schedule = build_cycle_schedule(
            self.trajectory_duration_s, max(self.dunk_every, 0), cycles=max(8, max(self.dunk_every, 1))
        )
        self.block_duration_s = schedule_duration(self._schedule)
        self._start_time = time.monotonic()
        self._last_logged_phase = ""
        self.create_timer(1.0 / max(self.publish_hz, 1.0), self._tick)
        self.get_logger().info(
            f"harvest sim started: scene={self.scene['scene_id']} "
            f"trajectory={self.trajectory_duration_s:.3f}s frames={len(self.frames)} "
            f"dunk_every={self.dunk_every} block={self.block_duration_s:.2f}s"
        )

    def _load_scene_and_model(self) -> None:
        scene_path = str(self.get_parameter("scene_path").value)
        model_path = str(self.get_parameter("model_path").value)
        try:
            scene = load_scene(scene_path)
            import mujoco

            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)
        except Exception as exc:
            self._fail_fast(f"harvest sim init failed: {exc}")
            raise

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.scene = scene
        self.frames: list[dict[str, Any]] = list(scene.get("frames", []))
        self.trajectory = list(scene["trajectory"])
        self.trajectory_duration_s = float(self.trajectory[-1]["time_from_start_s"])
        self.times = np.array([entry["time_from_start_s"] for entry in self.trajectory])
        self.joint_values = np.array(
            [
                [
                    entry["joints"]["lift"],
                    entry["joints"]["shoulder_yaw"],
                    entry["joints"]["elbow_yaw"],
                    entry["joints"]["wrist_yaw"],
                ]
                for entry in self.trajectory
            ]
        )
        self.target = np.array(scene["target"]["position"], dtype=np.float64)
        self._target_source = str(scene["target"].get("source") or "unknown")

        self._joint_qpos: dict[str, int] = {}
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                self._fail_fast(f"joint '{name}' missing from model {model_path}")
            self._joint_qpos[name] = int(model.jnt_qposadr[joint_id])
        self._eef_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        if self._eef_site < 0:
            self._fail_fast("site 'eef_site' missing from model")
        target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_fruit")
        self._mocap_target = int(model.body_mocapid[target_body]) if target_body >= 0 else -1

    def _fail_fast(self, message: str) -> None:
        self.error = message
        payload = build_status_payload(
            phase="ERROR",
            target=getattr(self, "target", []),
            eef=[],
            frame_index=0,
            frames_total=len(getattr(self, "frames", [])),
            elapsed_s=0.0,
            trajectory_duration_s=float(getattr(self, "trajectory_duration_s", 0.0)),
            dunk_every=self.dunk_every,
            joints={},
            error=message,
            target_source=getattr(self, "_target_source", "unknown"),
        )
        status = String()
        status.data = json.dumps(payload)
        self.status_pub.publish(status)
        self.get_logger().error(message)

    def _set_qpos(self, name: str, value: float) -> None:
        self.data.qpos[self._joint_qpos[name]] = float(value)

    def _tick(self) -> None:
        raw_elapsed = (time.monotonic() - self._start_time) * self.time_scale
        elapsed = raw_elapsed % self.block_duration_s if self.loop else raw_elapsed
        state = motion_state_for_elapsed(
            elapsed, self._schedule, self.times, self.joint_values, self.trajectory_duration_s
        )
        if state.phase != self._last_logged_phase:
            self.get_logger().info(
                f"phase -> {state.phase} (t={elapsed:.3f}s picked={is_picked(state.phase)})"
            )
            self._last_logged_phase = state.phase

        self._set_qpos("lift", state.joints[0])
        self._set_qpos("shoulder_yaw", state.joints[1])
        self._set_qpos("elbow_yaw", state.joints[2])
        self._set_qpos("wrist_yaw", state.joints[3])
        self._set_qpos("gripper", state.gripper)

        if self._mocap_target >= 0:
            self.data.mocap_pos[self._mocap_target] = self.target
        self.mujoco.mj_forward(self.model, self.data)

        eef = [float(v) for v in self.data.site_xpos[self._eef_site]]
        self._publish_joint_state()
        self._publish_status(elapsed, state.phase, eef)

    def _frame_index(self, elapsed: float) -> int:
        if not self.frames:
            return 0
        return int(elapsed * self.frames_per_sec) % len(self.frames)

    def _publish_joint_state(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [float(self.data.qpos[self._joint_qpos[name]]) for name in JOINT_NAMES]
        message.velocity = [0.0] * len(JOINT_NAMES)
        message.effort = [0.0] * len(JOINT_NAMES)
        self.joint_pub.publish(message)

    def _publish_status(self, elapsed: float, phase: str, eef: Sequence[float]) -> None:
        joints = {
            name: float(self.data.qpos[self._joint_qpos[name]]) for name in JOINT_NAMES
        }
        payload = build_status_payload(
            phase=phase,
            target=self.target,
            eef=eef,
            frame_index=self._frame_index(elapsed),
            frames_total=len(self.frames),
            elapsed_s=elapsed,
            trajectory_duration_s=self.trajectory_duration_s,
            dunk_every=self.dunk_every,
            joints=joints,
            error=self.error,
            target_source=self._target_source,
        )
        message = String()
        message.data = json.dumps(payload)
        self.status_pub.publish(message)


def main(args: Sequence[str] | None = None) -> int:
    rclpy.init(args=list(args) if args is not None else None)
    try:
        node = HarvestSimNode()
    except Exception:  # noqa: BLE001
        rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

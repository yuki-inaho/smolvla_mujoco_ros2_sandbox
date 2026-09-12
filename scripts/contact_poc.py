#!/usr/bin/env python3
"""Feasibility probe for contact/grasp physics (A-8).

Kinematic replay cannot tell us whether the modelled hand can actually hold a
fruit. This script builds a contact-enabled variant of the model (the mocap
target becomes a dynamic free body with friction) and runs a short MuJoCo
simulation: settle at the recorded grasp pose, close the gripper, lift, and
measure whether the fruit stays between the jaws.

It prints one JSON object so the result can be recorded as evidence. A low
retention number is a useful boundary result, not a script failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2_ws" / "src" / "tomato_harvest_sim"
sys.path.insert(0, str(PKG))

DEFAULT_SCENE = (
    PKG / "assets" / "scenes" / "7a7e56ff__camera_l__2026-08-04_08-19-43-646" / "scene.json"
)
DEFAULT_MODEL = PKG / "assets" / "tomato_scara.xml"
JOINT_NAMES = ("lift", "shoulder_yaw", "elbow_yaw", "wrist_yaw")
ACTUATOR_FOR_JOINT = {
    "lift": "lift_motor",
    "shoulder_yaw": "shoulder_motor",
    "elbow_yaw": "elbow_motor",
    "wrist_yaw": "wrist_motor",
    "gripper": "gripper_motor",
}
# Fruit properties are INFERRED (no material data in the bundle); this probe
# reports how sensitive retention is to them rather than claiming a value.
FRUIT_FRICTION = "1.0 0.05 0.001"
FRUIT_SOLREF = "0.02 1"
FRUIT_SOLIMP = "0.9 0.95 0.001"


def make_contact_model_xml(model_path: Path, target: np.ndarray) -> str:
    """Turn the kinematic mocap target into a contact-enabled free body."""
    xml = model_path.read_text()
    replacement = (
        f'<body name="target_fruit" pos="{target[0]:.6f} {target[1]:.6f} {target[2]:.6f}">'
        '<freejoint name="target_free"/>'
        '<geom name="target_geom" type="sphere" size="0.03" material="mat_fruit" '
        f'contype="1" conaffinity="1" friction="{FRUIT_FRICTION}" '
        f'solref="{FRUIT_SOLREF}" solimp="{FRUIT_SOLIMP}"/>'
        "</body>"
    )
    pattern = re.compile(r'<body name="target_fruit".*?</body>', re.DOTALL)
    if not pattern.search(xml):
        raise SystemExit("target_fruit body not found in model")
    return pattern.sub(replacement, xml, count=1)


def _id(mujoco, model, kind, name: str) -> int:
    identifier = mujoco.mj_name2id(model, kind, name)
    if identifier < 0:
        raise SystemExit(f"'{name}' missing from model")
    return int(identifier)


def run(
    scene_path: Path, model_path: Path, close_seconds: float, lift_seconds: float
) -> dict:
    import mujoco

    scene = json.loads(scene_path.read_text())
    target = np.array(scene["target"]["position"], dtype=np.float64)
    grasp = scene["trajectory"][-1]["joints"]

    model = mujoco.MjModel.from_xml_string(make_contact_model_xml(model_path, target))
    data = mujoco.MjData(model)

    qpos = {
        name: int(model.jnt_qposadr[_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in (*JOINT_NAMES, "gripper")
    }
    for name in JOINT_NAMES:
        data.qpos[qpos[name]] = float(grasp[name])
    data.qpos[qpos["gripper"]] = 0.0

    ctrl = {
        name: _id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR_FOR_JOINT[name])
        for name in (*JOINT_NAMES, "gripper")
    }
    for name in JOINT_NAMES:
        data.ctrl[ctrl[name]] = float(grasp[name])
    data.ctrl[ctrl["gripper"]] = float(model.actuator_ctrlrange[ctrl["gripper"]][1])

    site_id = _id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    fruit_body = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "target_fruit")
    lift_target = min(
        float(grasp["lift"]) + 0.25,
        float(model.actuator_ctrlrange[ctrl["lift"]][1]),
    )

    def snapshot() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(data.xpos[fruit_body], dtype=np.float64),
            np.array(data.site_xpos[site_id], dtype=np.float64),
        )

    # Settle, then close the jaws.
    for _ in range(round(close_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    fruit_start, eef_start = snapshot()
    offset_start = float(np.linalg.norm(fruit_start - eef_start))

    # Lift with the jaws closed; the fruit must ride along.
    data.ctrl[ctrl["lift"]] = lift_target
    max_penetration = 0.0
    for _ in range(round(lift_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        penetration = 0.03 - float(np.linalg.norm(data.xpos[fruit_body] - data.site_xpos[site_id]))
        max_penetration = max(max_penetration, penetration)
    fruit_end, eef_end = snapshot()
    offset_end = float(np.linalg.norm(fruit_end - eef_end))

    return {
        "scene_id": scene.get("scene_id"),
        "fruit_friction": FRUIT_FRICTION,
        "fruit_solref": FRUIT_SOLREF,
        "fruit_solimp": FRUIT_SOLIMP,
        "offset_start_m": round(offset_start, 5),
        "offset_end_m": round(offset_end, 5),
        "slip_m": round(offset_end - offset_start, 5),
        "fruit_z_start_m": round(float(fruit_start[2]), 5),
        "fruit_z_end_m": round(float(fruit_end[2]), 5),
        "lift_command_m": round(lift_target, 5),
        "max_penetration_m": round(max_penetration, 5),
        "retained": bool(offset_end - offset_start < 0.02 and fruit_end[2] > fruit_start[2] + 0.05),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--close-seconds", type=float, default=0.6)
    parser.add_argument("--lift-seconds", type=float, default=1.5)
    args = parser.parse_args()

    result = run(args.scene, args.model, args.close_seconds, args.lift_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

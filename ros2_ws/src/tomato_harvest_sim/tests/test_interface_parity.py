"""Interface parity contract (A-2).

Published ``/joint_states`` names, the real machine arm joints and the MuJoCo
model joint set must agree. Sim-only degrees of freedom (e.g. the removed
``rail`` abstraction) must never leak into any of these sets.
"""

import pytest
import tomllib
from paths import MODEL_PATH, SNAPSHOT_PATH
from tomato_harvest_sim.harvest_sim_node import JOINT_NAMES, JOINT_VMAX

REAL_ARM_JOINTS = {"lift", "shoulder_yaw", "elbow_yaw", "wrist_yaw", "gripper"}
DOCUMENTED_NON_ARM_JOINTS = {"camera_yaw"}  # camera motor (S6), separate subsystem
ACTUATORS = {
    "lift_motor",
    "shoulder_motor",
    "elbow_motor",
    "wrist_motor",
    "gripper_motor",
}
SNAPSHOT_TO_SIM = {
    "lift": 0,
    "shoulder_yaw": 1,
    "elbow_yaw": 2,
    "wrist_yaw": 3,
}


def _model():
    mujoco = pytest.importorskip("mujoco")
    return mujoco, mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def test_published_joint_names_match_real_machine_arm():
    assert set(JOINT_NAMES) == REAL_ARM_JOINTS
    assert "rail" not in JOINT_NAMES


def test_model_joint_set_equals_published_plus_documented_extras():
    mujoco, model = _model()
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    assert names == REAL_ARM_JOINTS | DOCUMENTED_NON_ARM_JOINTS


def test_model_actuator_set_matches_arm_joints_only():
    mujoco, model = _model()
    actuators = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    }
    assert actuators == ACTUATORS


def test_snapshot_joint_limits_bind_the_published_arm_joints():
    """The real robot snapshot must back the published limits (A-2/M4).

    The snapshot uses ``j_arm_*`` names for the four arm joints (the gripper is
    documented separately), so the mapping is asserted explicitly instead of
    assuming name equality.
    """
    snapshot = tomllib.loads(SNAPSHOT_PATH.read_text())
    limits = snapshot["joint_limits"]
    names = {key.removeprefix("j_arm_") for key in limits}
    assert names == REAL_ARM_JOINTS - {"gripper"}

    mujoco, model = _model()
    for sim_name, index in SNAPSHOT_TO_SIM.items():
        entry = limits[f"j_arm_{sim_name}"]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
        low, high = model.jnt_range[joint_id]
        if sim_name == "lift":
            # spec §7-7: the model uses 0 while the snapshot records -0.0005.
            assert abs(low - entry["min_position"]) <= 5e-4
        else:
            assert low == pytest.approx(entry["min_position"], abs=1e-9)
        assert high == pytest.approx(entry["max_position"], abs=1e-9)
        assert entry["max_velocity"] == pytest.approx(JOINT_VMAX[index])
        assert entry["max_acceleration"] > 0.0

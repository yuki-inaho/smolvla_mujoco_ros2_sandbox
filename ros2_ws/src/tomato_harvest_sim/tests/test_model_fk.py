import math
from pathlib import Path

import numpy as np
import pytest
from paths import MODEL_PATH, TRAJECTORY_PATH
from tomato_harvest_sim.sim_common import load_numeric_series

XML_PATH = MODEL_PATH

TRAJECTORY_COLUMNS = [
    "joint_lift_rad",
    "joint_shoulder_yaw_rad",
    "joint_elbow_yaw_rad",
    "joint_wrist_yaw_rad",
    "eef_x_m",
    "eef_y_m",
    "eef_z_m",
]

FK_TOLERANCE_M = 0.01

EXPECTED_JOINT_RANGES = {
    "lift": (0.0, 0.9725),
    "shoulder_yaw": (-2.36, 2.36),
    "elbow_yaw": (-3.05, 3.05),
    "wrist_yaw": (-6.1, 6.1),
    "gripper": (0.0, 0.0843),
}


def _load_trajectory():
    if not Path(TRAJECTORY_PATH).exists():
        pytest.fail(f"recorded trajectory missing: {TRAJECTORY_PATH}", pytrace=False)
    return load_numeric_series(TRAJECTORY_PATH, TRAJECTORY_COLUMNS)


def _mujoco():
    mujoco = pytest.importorskip("mujoco")
    if not XML_PATH.exists():
        pytest.fail(f"MuJoCo model missing: {XML_PATH}", pytrace=False)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    return mujoco, model, mujoco.MjData(model)


def _set_pose(mujoco, model, data, lift, shoulder, elbow, wrist):
    values = {
        "lift": lift,
        "shoulder_yaw": shoulder,
        "elbow_yaw": elbow,
        "wrist_yaw": wrist,
        "gripper": 0.0,
    }
    for name, value in values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"joint {name} not found in model"
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def _eef_position(mujoco, model, data):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    assert site_id >= 0, "eef_site not found in model"
    return np.array(data.site_xpos[site_id], dtype=np.float64)


def test_model_declares_scara_joints_and_actuators():
    mujoco, model, _ = _mujoco()
    for joint in ("lift", "shoulder_yaw", "elbow_yaw", "wrist_yaw", "gripper"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0
    for actuator in (
        "lift_motor",
        "shoulder_motor",
        "elbow_motor",
        "wrist_motor",
        "gripper_motor",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) >= 0


def test_joint_limits_match_robot_snapshot():
    mujoco, model, _ = _mujoco()
    for name, (low, high) in EXPECTED_JOINT_RANGES.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actual = tuple(float(v) for v in model.jnt_range[joint_id])
        assert actual == pytest.approx((low, high), abs=1e-9), f"{name} limits {actual}"


def test_grasp_point_matches_urdf_geometry():
    """Zero pose: shoulder at (0.195, 0, 0.960); grasp at x=0.345, z=0.890.

    x = 0.195 + 0.285 - 0.345 + 0.210 (upperarm/forearm/grasp offsets)
    z = 0.960 - 0.070 (wrist -> eef_footprint vertical drop)
    """
    mujoco, model, data = _mujoco()
    _set_pose(mujoco, model, data, 0.0, 0.0, 0.0, 0.0)
    position = _eef_position(mujoco, model, data)
    assert position == pytest.approx([0.345, 0.0, 0.890], abs=1e-6)


def test_hand_geometry_declared():
    mujoco, model, _ = _mujoco()
    for geom in ("hand_body", "finger_fixed", "finger_moving"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom) >= 0
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hand_body")
    half = model.geom_size[hand_id]
    assert float(half[0]) == pytest.approx(0.13, abs=1e-9)  # 0.26 m long
    assert float(half[1]) == pytest.approx(0.045, abs=1e-9)  # 0.09 m wide
    assert float(half[2]) == pytest.approx(0.045, abs=1e-9)  # 0.09 m high


def test_eef_endpoint_matches_recorded_within_tolerance():
    mujoco, model, data = _mujoco()
    trajectory = _load_trajectory()
    index = -1
    _set_pose(
        mujoco,
        model,
        data,
        trajectory["joint_lift_rad"][index],
        trajectory["joint_shoulder_yaw_rad"][index],
        trajectory["joint_elbow_yaw_rad"][index],
        trajectory["joint_wrist_yaw_rad"][index],
    )
    expected = np.array(
        [
            trajectory["eef_x_m"][index],
            trajectory["eef_y_m"][index],
            trajectory["eef_z_m"][index],
        ]
    )
    error = float(np.linalg.norm(_eef_position(mujoco, model, data) - expected))
    assert error < FK_TOLERANCE_M, f"endpoint FK error {error:.4f} m exceeds {FK_TOLERANCE_M} m"


def test_eef_along_trajectory_stays_within_tolerance():
    mujoco, model, data = _mujoco()
    trajectory = _load_trajectory()
    count = trajectory["eef_x_m"].shape[0]
    errors = []
    for index in range(0, count, 50):
        _set_pose(
            mujoco,
            model,
            data,
            trajectory["joint_lift_rad"][index],
            trajectory["joint_shoulder_yaw_rad"][index],
            trajectory["joint_elbow_yaw_rad"][index],
            trajectory["joint_wrist_yaw_rad"][index],
        )
        expected = np.array(
            [
                trajectory["eef_x_m"][index],
                trajectory["eef_y_m"][index],
                trajectory["eef_z_m"][index],
            ]
        )
        errors.append(float(np.linalg.norm(_eef_position(mujoco, model, data) - expected)))
    assert max(errors) < FK_TOLERANCE_M, f"swept FK max error {max(errors):.4f} m exceeds {FK_TOLERANCE_M} m"


def test_recorded_trajectories_start_at_home():
    """The motion model assumes every recorded approach starts at HOME_JOINTS."""
    import duckdb
    from paths import RUN_DIR
    from tomato_harvest_sim.harvest_sim_node import HOME_JOINTS

    runs_root = RUN_DIR.parent
    row = duckdb.sql(
        f"""
        with firsts as (
            select any_value(joint_lift_rad) as lift,
                   any_value(joint_shoulder_yaw_rad) as shoulder,
                   any_value(joint_elbow_yaw_rad) as elbow,
                   any_value(joint_wrist_yaw_rad) as wrist
            from (
                select filename, joint_lift_rad, joint_shoulder_yaw_rad,
                       joint_elbow_yaw_rad, joint_wrist_yaw_rad,
                       row_number() over (partition by filename order by time_from_start_s) as rn
                from read_parquet('{runs_root}/*/trajectory.parquet', filename=true)
            )
            where rn = 1
            group by filename
        )
        select count(*) filter (
                   where abs(lift - {HOME_JOINTS[0]}) > 1e-6
                      or abs(shoulder - {HOME_JOINTS[1]}) > 1e-6
                      or abs(elbow - {HOME_JOINTS[2]}) > 1e-6
                      or abs(wrist - {HOME_JOINTS[3]}) > 1e-6
               ),
               count(*)
        from firsts
        """
    ).fetchone()
    assert row[1] >= 100
    assert row[0] == 0, f"{row[0]} trajectories do not start at HOME_JOINTS"


def test_motion_helpers_use_real_limits():
    from tomato_harvest_sim.harvest_sim_node import (
        DUNK_JOINTS,
        HOME_JOINTS,
        dunk_phase_duration,
        joint_move_duration,
    )

    assert HOME_JOINTS == pytest.approx(
        (0.30, math.radians(25.0), 0.0, math.radians(-25.0)), abs=1e-9
    )
    assert DUNK_JOINTS == pytest.approx(
        (0.003, math.radians(34.0), math.radians(-35.0), math.radians(1.0)), abs=1e-9
    )
    move = joint_move_duration(HOME_JOINTS, DUNK_JOINTS)
    assert 0.5 < move < 0.8, move  # lift 0.297 m at 0.5 m/s dominates
    assert dunk_phase_duration() > 2.0


def test_vehicle_wheels_use_documented_radii():
    """vehicle_driver.py: WHEEL_RADIUS_RAIL = 0.050 m, ground mode = 0.125 m."""
    mujoco, model, _ = _mujoco()
    for wheel in ("wheel_front_l", "wheel_front_r", "wheel_rear_l", "wheel_rear_r"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wheel)
        assert geom_id >= 0, wheel
        assert float(model.geom_size[geom_id][0]) == pytest.approx(0.05, abs=1e-9)
    for geom in ("chassis", "rail_beam_l", "rail_beam_r", "caster_front", "caster_rear"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom) >= 0, geom


def test_camera_mount_reproduces_documented_chain_with_calibration():
    """config/tmt4-02/camera_tf_broadcaster.py + camera_tf_calibration.toml:
    arm_lift_stage -> arm_shoulder_footprint [0,0,0] rpy [0,-90,0];
    camera_base = [0.500, 0, 0.191 + cal.z(0.02)]; camera_motor +-90 deg."""
    mujoco, model, _ = _mujoco()

    def body_position(name):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id >= 0, name
        return np.array(model.body_pos[body_id], dtype=np.float64)

    assert body_position("lift_column") == pytest.approx([0.0, 0.0, 0.960], abs=1e-9)
    assert body_position("shoulder") == pytest.approx([0.195, 0.0, 0.0], abs=1e-9)
    assert body_position("shoulder_footprint") == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert body_position("camera_mount") == pytest.approx([0.500, 0.0, 0.211], abs=1e-9)
    footprint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shoulder_footprint")
    w, x, y, z = (float(v) for v in model.body_quat[footprint_id])
    # body_quat is stored as float32, so allow 1e-5 tolerance.
    assert (w, x, y, z) == pytest.approx(
        (math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0), abs=1e-5
    )
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "camera_yaw")
    assert joint_id >= 0
    assert tuple(float(v) for v in model.jnt_range[joint_id]) == pytest.approx(
        (-1.5708, 1.5708), abs=1e-6
    )


def test_hand_belts_declared():
    """belt_l / belt_r (EPOS4 ID9/10, DJI M2006) are the hand side belts."""
    mujoco, model, _ = _mujoco()
    for belt in ("belt_l", "belt_r"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, belt) >= 0


def _finger_surface_gap(mujoco, model, data) -> float:
    fixed_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_fixed")
    moving_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_moving")
    fixed_y = float(data.geom_xpos[fixed_id][1])
    moving_y = float(data.geom_xpos[moving_id][1])
    fixed_half = float(model.geom_size[fixed_id][1])
    moving_half = float(model.geom_size[moving_id][1])
    return (fixed_y - fixed_half) - (moving_y + moving_half)


def test_gripper_surface_gap_matches_documented_widths():
    """planning_params: grip_width semiopen 0.090 m -> closed 0.0057 m
    (belt surface distance, collision_range.toml). Joint stroke 0.0843 m."""
    mujoco, model, data = _mujoco()
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    qpos_adr = model.jnt_qposadr[joint_id]
    for stroke, half in ((0.0, 0.045), (0.0843, 0.00285)):
        data.qpos[qpos_adr] = stroke
        mujoco.mj_forward(model, data)
        gap = _finger_surface_gap(mujoco, model, data)
        assert gap == pytest.approx(2.0 * half, abs=1e-6), (stroke, gap)

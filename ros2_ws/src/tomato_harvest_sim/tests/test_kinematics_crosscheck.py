"""Kinematic cross-check: MuJoCo analytic Jacobian vs finite differences.

An independent numerical method must agree with the analytic Jacobian the
model provides, for both translation and rotation, across every movable
joint. Disagreement would mean the model geometry and the reported site
motion are inconsistent (a silent way for FK claims to drift).
"""

import numpy as np
import pytest
from paths import MODEL_PATH

POSE = {
    "lift": 0.4,
    "shoulder_yaw": 0.6,
    "elbow_yaw": -0.7,
    "wrist_yaw": 0.9,
    "gripper": 0.02,
    "camera_yaw": 0.3,
}
EPS = 1e-6
TOLERANCE = 1e-5


def _model():
    mujoco = pytest.importorskip("mujoco")
    return mujoco, mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def test_site_jacobian_matches_finite_differences():
    mujoco, model = _model()
    data = mujoco.MjData(model)
    qpos_addresses = {}
    dof_addresses = {}
    for name, value in POSE.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, name
        qpos_addresses[name] = int(model.jnt_qposadr[joint_id])
        dof_addresses[name] = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_addresses[name]] = value
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    rotation = np.array(data.site_xmat[site_id]).reshape(3, 3)

    for name, qpos_address in qpos_addresses.items():
        dof = dof_addresses[name]
        data.qpos[qpos_address] += EPS
        mujoco.mj_forward(model, data)
        plus_position = np.array(data.site_xpos[site_id])
        plus_rotation = np.array(data.site_xmat[site_id]).reshape(3, 3)
        data.qpos[qpos_address] -= 2.0 * EPS
        mujoco.mj_forward(model, data)
        minus_position = np.array(data.site_xpos[site_id])
        minus_rotation = np.array(data.site_xmat[site_id]).reshape(3, 3)
        data.qpos[qpos_address] += EPS
        mujoco.mj_forward(model, data)

        numeric_position = (plus_position - minus_position) / (2.0 * EPS)
        delta_rotation = (plus_rotation - minus_rotation) / (2.0 * EPS)
        skew = delta_rotation @ rotation.T
        numeric_rotation = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        assert np.allclose(jacp[:, dof], numeric_position, atol=TOLERANCE), name
        assert np.allclose(jacr[:, dof], numeric_rotation, atol=TOLERANCE), name

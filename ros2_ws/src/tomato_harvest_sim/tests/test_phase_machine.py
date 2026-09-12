"""Phase-machine tests against the real motion API (``motion_state_for_elapsed``)."""

import json

import pytest
from paths import SCENE_PATH
from tomato_harvest_sim.harvest_sim_node import (
    CLOSE_DURATION_S,
    DONE_HOLD_S,
    DUNK_JOINTS,
    DUNK_RELEASE_S,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_JOINTS,
    UNLOAD_HOLD_S,
    build_cycle_schedule,
    dunk_phase_duration,
    dunk_shake_duration,
    joint_move_duration,
    load_scene,
    motion_state_for_elapsed,
    schedule_duration,
)

TRAJECTORY_DURATION = 2.0
# Recorded trajectories always start at HOME_JOINTS (verified for the dataset).
TRAJECTORY_TIMES = [0.0, TRAJECTORY_DURATION]
TRAJECTORY_VALUES = [
    list(HOME_JOINTS),
    [0.20, 0.10, 0.30, 0.20],
]


def _state(elapsed: float, dunk_every: int):
    schedule = build_cycle_schedule(TRAJECTORY_DURATION, dunk_every, cycles=dunk_every)
    return motion_state_for_elapsed(
        elapsed,
        schedule,
        TRAJECTORY_TIMES,
        TRAJECTORY_VALUES,
        TRAJECTORY_DURATION,
    )


def test_phase_sequence_and_timing_without_dunk():
    cycle = TRAJECTORY_DURATION + CLOSE_DURATION_S + TRAJECTORY_DURATION + UNLOAD_HOLD_S + DONE_HOLD_S
    assert _state(0.0, dunk_every=0).phase == "APPROACH"
    assert _state(TRAJECTORY_DURATION + CLOSE_DURATION_S / 2, 0).phase == "CLOSE"
    assert _state(TRAJECTORY_DURATION + CLOSE_DURATION_S + 1.0, 0).phase == "LIFT"
    assert _state(TRAJECTORY_DURATION + CLOSE_DURATION_S + TRAJECTORY_DURATION + 0.1, 0).phase == "UNLOAD"
    assert _state(cycle - 0.05, 0).phase == "DONE"
    assert _state(cycle + 0.05, 0).phase == "APPROACH"


def test_approach_forward_and_lift_reverse_playback():
    approach = _state(1.0, dunk_every=0)
    assert approach.joints[0] == pytest.approx(0.25, abs=1e-6)
    lift_time = TRAJECTORY_DURATION + CLOSE_DURATION_S + 1.0
    retract = _state(lift_time, dunk_every=0)
    assert retract.phase == "LIFT"
    assert retract.joints[0] == pytest.approx(0.25, abs=1e-6)  # mirror of approach


def test_gripper_closes_then_releases_smoothly():
    close_mid = _state(TRAJECTORY_DURATION + CLOSE_DURATION_S / 2, dunk_every=0)
    assert close_mid.gripper == pytest.approx((GRIPPER_OPEN + GRIPPER_CLOSED) / 2, abs=1e-6)
    after_close = _state(TRAJECTORY_DURATION + CLOSE_DURATION_S + 0.1, dunk_every=0)
    assert after_close.gripper == GRIPPER_CLOSED
    cycle = TRAJECTORY_DURATION + CLOSE_DURATION_S + TRAJECTORY_DURATION + UNLOAD_HOLD_S + DONE_HOLD_S
    done_start = _state(cycle - DONE_HOLD_S + 1e-9, dunk_every=0)
    assert done_start.phase == "DONE"
    assert done_start.gripper == pytest.approx(GRIPPER_CLOSED, abs=1e-6)
    done_end = _state(cycle - 1e-9, dunk_every=0)
    assert done_end.gripper == pytest.approx(GRIPPER_OPEN, abs=1e-6)


def test_dunk_cycle_moves_to_dunk_pose_and_releases():
    dunk_every = 1  # every cycle dunks
    schedule = build_cycle_schedule(TRAJECTORY_DURATION, dunk_every, cycles=1)
    approach_end = TRAJECTORY_DURATION
    close_end = approach_end + CLOSE_DURATION_S
    lift_end = close_end + TRAJECTORY_DURATION
    unload_start = lift_end
    dunk_move = joint_move_duration(HOME_JOINTS, DUNK_JOINTS)
    assert schedule[0]["duration"] == pytest.approx(
        lift_end + dunk_phase_duration() + DONE_HOLD_S, abs=1e-9
    )
    # after the dunk move the release starts: gripper opens continuously
    releasing = _state(unload_start + dunk_move + DUNK_RELEASE_S / 2, dunk_every)
    assert releasing.phase == "UNLOAD"
    assert releasing.joints == pytest.approx(DUNK_JOINTS, abs=1e-9)
    assert 0.0 < releasing.gripper < GRIPPER_CLOSED
    # once released (settle window), the arm sits at the dunk pose with gripper open
    settle_start = unload_start + dunk_move + DUNK_RELEASE_S + dunk_shake_duration()
    at_dunk = _state(settle_start + 0.05, dunk_every)
    assert at_dunk.phase == "UNLOAD"
    assert at_dunk.joints == pytest.approx(DUNK_JOINTS, abs=1e-9)
    assert at_dunk.gripper == GRIPPER_OPEN
    # DONE of a dunk cycle: home pose, gripper open (fruit released)
    done = _state(schedule_duration(schedule) - 0.01, dunk_every)
    assert done.phase == "DONE"
    assert done.joints == pytest.approx(HOME_JOINTS, abs=1e-9)
    assert done.gripper == GRIPPER_OPEN


def test_schedule_duration_matches_phase_sum():
    schedule = build_cycle_schedule(TRAJECTORY_DURATION, dunk_every=2, cycles=2)
    assert schedule_duration(schedule) == pytest.approx(
        2 * (2 * TRAJECTORY_DURATION + CLOSE_DURATION_S + UNLOAD_HOLD_S + DONE_HOLD_S)
        - UNLOAD_HOLD_S
        + dunk_phase_duration(),
        abs=1e-9,
    )


def test_load_scene_reads_generated_scene_json():
    if not SCENE_PATH.exists():
        pytest.fail(f"generated scene.json missing: {SCENE_PATH}", pytrace=False)
    scene = load_scene(str(SCENE_PATH))
    assert scene["scene_id"] == "7a7e56ff__camera_l__2026-08-04_08-19-43-646"
    assert scene["fruits"] and scene["trajectory"]
    assert json.dumps(scene)  # JSON-serialisable payload

def test_no_teleports_and_joint_velocity_limits():
    """Physics guard: a full block must stay continuous and within limits.

    Regression for the trapezoid-profile bug that made the arm jump with
    ~1000 rad/s during synced moves, and for the gripper snap at cycle
    boundaries.
    """
    from tomato_harvest_sim.harvest_sim_node import JOINT_VMAX

    dt = 1.0 / 120.0
    block = build_cycle_schedule(TRAJECTORY_DURATION, dunk_every=2, cycles=4)
    total = schedule_duration(block)
    previous = None
    max_speed = [0.0] * 5
    steps = int(total / dt) + 1
    for index in range(steps):
        state = _state(index * dt, dunk_every=2)
        values = list(state.joints) + [state.gripper]
        if previous is not None:
            for joint in range(5):
                speed = abs(values[joint] - previous[joint]) / dt
                max_speed[joint] = max(max_speed[joint], speed)
        previous = values
    for joint in range(4):
        assert max_speed[joint] <= JOINT_VMAX[joint] * 1.05, (joint, max_speed[joint])
    assert max_speed[4] <= 0.5, max_speed[4]  # gripper: no snap (m/s)

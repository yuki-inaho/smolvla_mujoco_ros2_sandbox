"""Continuity and profile contracts (A-1 / A-6 regressions).

A-1: every phase boundary of the cycle schedule must stay within the real
     joint velocity limits (no teleports, no setpoint steps).
A-6: a synchronised move given an infeasible duration must fail loudly
     instead of silently truncating; feasible profiles must respect the
     velocity/acceleration limits, stay monotonic and reach the target.
"""

import pytest
from tomato_harvest_sim.harvest_sim_node import (
    DUNK_JOINTS,
    HOME_JOINTS,
    JOINT_AMAX,
    JOINT_VMAX,
    build_cycle_schedule,
    joint_move_duration,
    joint_position,
    motion_state_for_elapsed,
    schedule_duration,
    trapezoid_time,
)

TRAJECTORY_DURATION = 2.0
TIMES = [0.0, TRAJECTORY_DURATION]
VALUES = [list(HOME_JOINTS), [0.20, 0.10, 0.30, 0.20]]
DT = 1.0 / 1000.0
GRIPPER_RATE_LIMIT = 0.5  # m/s, bounded by the physics audit (spec §12)


def _state(t: float, dunk_every: int = 2):
    schedule = build_cycle_schedule(
        TRAJECTORY_DURATION, dunk_every, cycles=max(4, dunk_every)
    )
    return (
        motion_state_for_elapsed(t, schedule, TIMES, VALUES, TRAJECTORY_DURATION),
        schedule,
    )


def test_phase_boundary_setpoints_stay_within_velocity_limits():
    _, schedule = _state(0.0)
    total = schedule_duration(schedule)
    previous = None
    for index in range(int(total / DT) + 1):
        current, _ = _state(index * DT)
        values = list(current.joints) + [current.gripper]
        if previous is not None:
            for joint in range(4):
                speed = abs(values[joint] - previous[joint]) / DT
                assert speed <= JOINT_VMAX[joint] * 1.05, (
                    index * DT,
                    joint,
                    speed,
                )
            assert abs(values[4] - previous[4]) / DT <= GRIPPER_RATE_LIMIT
        previous = values


def test_minimum_duration_profile_reaches_target_exactly():
    delta = 0.5
    duration = trapezoid_time(delta, 0.5, 8.0)
    values = joint_position([0.0] * 4, [delta] * 4, duration, duration)
    for value in values:
        assert value == pytest.approx(delta, abs=1e-9)


def test_infeasible_duration_is_rejected():
    delta = 0.5
    minimum = trapezoid_time(delta, 0.5, 8.0)
    with pytest.raises(ValueError):
        joint_position([0.0] * 4, [delta] * 4, 0.0, duration=minimum * 0.9)


@pytest.mark.parametrize("duration", [0.0, -1.0, 1e-12])
def test_nonpositive_duration_is_rejected(duration):
    with pytest.raises(ValueError):
        joint_position([0.0] * 4, [0.5] * 4, 0.0, duration=duration)


def _boundary_times(schedule):
    edges = []
    total = 0.0
    for cycle in schedule:
        for _, duration, _ in cycle["phases"]:
            total += duration
            edges.append(total)
    return edges


def test_boundaries_are_position_continuous_and_velocity_bounded():
    _, schedule = _state(0.0)
    total = schedule_duration(schedule)
    for boundary in _boundary_times(schedule) + [total]:
        before, _ = _state(boundary - 1e-6)
        after, _ = _state(boundary + 1e-6)
        for joint in range(4):
            delta = abs(after.joints[joint] - before.joints[joint])
            assert delta <= JOINT_VMAX[joint] * 2e-6 * 1.05, (boundary, joint)
        warm = _state(boundary - 5e-4)[0]
        middle = _state(boundary - 1e-9)[0]
        cool = _state(boundary + 5e-4)[0]
        for joint in range(4):
            incoming = abs(middle.joints[joint] - warm.joints[joint]) / 5e-4
            outgoing = abs(cool.joints[joint] - middle.joints[joint]) / 5e-4
            assert max(incoming, outgoing) <= JOINT_VMAX[joint] * 1.1, (
                boundary,
                joint,
                incoming,
                outgoing,
            )


def test_feasible_profiles_respect_limits_and_monotonicity():
    for delta in (0.01, 0.05, 0.2, 0.9):
        minimum = trapezoid_time(delta, JOINT_VMAX[0], JOINT_AMAX[0])
        for scale in (1.0, 1.5, 3.0):
            duration = max(minimum * scale, minimum)
            samples = 2001
            step = duration / (samples - 1)
            previous = None
            previous_speed = None
            for index in range(samples):
                t = index * step
                value = joint_position([0.0] * 4, [delta] * 4, t, duration)[0]
                if previous is not None:
                    assert value >= previous - 1e-12, (delta, scale, t)
                    speed = (value - previous) / step
                    assert speed <= JOINT_VMAX[0] * 1.02, (delta, scale, t, speed)
                    if previous_speed is not None:
                        accel = abs(speed - previous_speed) / step
                        assert accel <= JOINT_AMAX[0] * 1.10, (
                            delta,
                            scale,
                            t,
                            accel,
                        )
                    previous_speed = speed
                previous = value
            last = joint_position([0.0] * 4, [delta] * 4, duration, duration)[0]
            assert last == pytest.approx(delta, abs=1e-9)


def test_synchronised_move_duration_covers_every_joint():
    duration = joint_move_duration(HOME_JOINTS, DUNK_JOINTS)
    assert duration > 0.0
    values = joint_position(HOME_JOINTS, DUNK_JOINTS, duration, duration)
    for index, value in enumerate(values):
        assert value == pytest.approx(DUNK_JOINTS[index], abs=1e-9)

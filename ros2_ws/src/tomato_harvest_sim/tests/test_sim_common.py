import math

import numpy as np
import pytest
from tomato_harvest_sim.sim_common import (
    estimate_depth_scale,
    load_numeric_series,
    quat_to_matrix,
    sample_trajectory,
    transform_points,
)


def test_quat_identity_returns_identity_matrix():
    matrix = quat_to_matrix([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)


def test_quat_90deg_about_z_rotates_x_to_y():
    half = math.pi / 4.0
    matrix = quat_to_matrix([0.0, 0.0, math.sin(half), math.cos(half)])
    np.testing.assert_allclose(matrix @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(matrix @ np.array([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-9)


def test_quat_180deg_about_z_negates_x_and_y():
    matrix = quat_to_matrix([0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        matrix,
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-9,
    )


def test_quat_accepts_non_unit_norm_by_normalizing():
    matrix = quat_to_matrix([0.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(
        matrix,
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-9,
    )


def test_quat_rejects_wrong_length():
    with pytest.raises(ValueError):
        quat_to_matrix([0.0, 0.0, 1.0])


def test_transform_points_known_translation():
    tf = np.eye(4)
    tf[:3, 3] = [1.0, 2.0, 3.0]
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        transform_points(points, tf),
        [[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]],
        atol=1e-12,
    )


def test_transform_points_applies_rotation_then_translation():
    half = math.pi / 4.0
    tf = np.eye(4)
    tf[:3, :3] = quat_to_matrix([0.0, 0.0, math.sin(half), math.cos(half)])
    tf[:3, 3] = [1.0, 0.0, 0.0]
    points = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(transform_points(points, tf), [[1.0, 1.0, 0.0]], atol=1e-9)


def test_sample_trajectory_endpoints_and_midpoint():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    np.testing.assert_allclose(sample_trajectory(times, values, 0.0), [0.0, 10.0])
    np.testing.assert_allclose(sample_trajectory(times, values, 2.0), [2.0, 30.0])
    np.testing.assert_allclose(sample_trajectory(times, values, 0.5), [0.5, 15.0])


def test_sample_trajectory_clamps_outside_range():
    times = np.array([0.0, 1.0])
    values = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(sample_trajectory(times, values, -5.0), [1.0])
    np.testing.assert_allclose(sample_trajectory(times, values, 99.0), [2.0])


def test_sample_trajectory_returns_scalar_for_single_column():
    times = np.array([0.0, 1.0])
    values = np.array([[1.0], [2.0]])
    result = sample_trajectory(times, values, 0.5)
    assert np.ndim(result) == 0
    assert abs(float(result) - 1.5) < 1e-12


def test_estimate_depth_scale_millimeters():
    scale = estimate_depth_scale([1000, 1500, 2000], [1.0, 1.5, 2.0])
    assert abs(scale - 0.001) < 1e-12


def test_estimate_depth_scale_ignores_non_positive_raw():
    scale = estimate_depth_scale([0, 1000, 1500], [0.0, 1.0, 1.5])
    assert abs(scale - 0.001) < 1e-12


def test_estimate_depth_scale_rejects_empty_signal():
    with pytest.raises(ValueError):
        estimate_depth_scale([0, 0], [0.0, 0.0])


def test_load_numeric_series_roundtrip(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "series.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            "COPY (SELECT * FROM (VALUES (0.0, 1.5), (1.0, 2.5)) AS t(a, b)) "
            f"TO '{path}' (FORMAT PARQUET)"
        )
    finally:
        con.close()

    data = load_numeric_series(str(path), columns=["a", "b"])
    assert set(data) == {"a", "b"}
    np.testing.assert_allclose(data["a"], [0.0, 1.0])
    np.testing.assert_allclose(data["b"], [1.5, 2.5])

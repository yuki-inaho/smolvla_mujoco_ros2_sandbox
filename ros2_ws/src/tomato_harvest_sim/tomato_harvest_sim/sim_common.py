"""Numeric helpers shared by the tomato harvest scene builder and simulation.

All functions operate on plain numpy arrays so that they can be unit tested
without ROS or a running MuJoCo instance.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

QUAT_XYZW = "xyzw"


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Convert a quaternion ``[x, y, z, w]`` into a 3x3 rotation matrix.

    The quaternion is normalized internally so that slightly non-unit inputs
    (as returned by some perception pipelines) still yield a valid rotation.
    """
    array = np.asarray(quat, dtype=np.float64)
    if array.shape != (4,):
        raise ValueError(f"quaternion must have 4 elements, got shape {array.shape}")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = array / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_transform(translation: Sequence[float], quat: Sequence[float]) -> np.ndarray:
    """Build a homogeneous 4x4 transform from translation + quaternion xyzw."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_matrix(quat)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to an ``(N, 3)`` array of points."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {array.shape}")
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {matrix.shape}")
    rotated = array @ matrix[:3, :3].T
    return rotated + matrix[:3, 3]


def sample_trajectory(times: np.ndarray, values: np.ndarray, t: float):
    """Linearly sample ``values`` at time ``t``, clamping outside the range.

    Returns a numpy scalar when ``values`` has a single column, otherwise a
    1-D array with one entry per column.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if times.shape[0] != values.shape[0]:
        raise ValueError(
            f"times and values length mismatch: {times.shape[0]} vs {values.shape[0]}"
        )
    if times.shape[0] == 0:
        raise ValueError("trajectory must contain at least one sample")
    if times.shape[0] == 1:
        sampled = values[0]
    else:
        sampled = np.array(
            [np.interp(float(t), times, values[:, column]) for column in range(values.shape[1])],
            dtype=np.float64,
        )
    if sampled.shape[0] == 1:
        return float(sampled[0])
    return sampled


def estimate_depth_scale(
    raw_values: Iterable[float], metric_values: Iterable[float]
) -> float:
    """Estimate the ``metric = raw * scale`` factor from paired samples.

    Only pairs with positive raw and metric readings contribute. Returns the
    median ratio so that a few outliers cannot dominate.
    """
    raw = np.asarray(list(raw_values), dtype=np.float64).reshape(-1)
    metric = np.asarray(list(metric_values), dtype=np.float64).reshape(-1)
    if raw.shape != metric.shape:
        raise ValueError("raw_values and metric_values must have the same length")
    valid = (raw > 0.0) & (metric > 0.0)
    if not np.any(valid):
        raise ValueError("no positive raw/metric pairs to estimate depth scale")
    ratios = metric[valid] / raw[valid]
    return float(np.median(ratios))


def load_numeric_series(path: str, columns: Sequence[str]) -> dict[str, np.ndarray]:
    """Load selected numeric columns from a parquet file into numpy arrays."""
    import duckdb

    selected = ", ".join(f'"{name}"' for name in columns)
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            f"SELECT {selected} FROM read_parquet(?)", [str(path)]
        )
        rows = cursor.fetchall()
    finally:
        connection.close()

    result: dict[str, np.ndarray] = {name: np.array([], dtype=np.float64) for name in columns}
    if not rows:
        return result
    columns_data = list(zip(*rows))
    for name, series in zip(columns, columns_data):
        result[name] = np.asarray(series, dtype=np.float64)
    return result

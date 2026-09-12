import glob
from pathlib import Path

import numpy as np
import pytest
from paths import BUNDLE_DIR, SCENE_ID
from tomato_harvest_sim.sim_common import estimate_depth_scale

SCENE_TIMESTAMP = SCENE_ID.split("__")[-1]

DEPTH_K = np.array(
    [
        [413.48858642578125, 0.0, 324.76543438433146],
        [0.0, 445.7857360839844, 218.2393468216178],
        [0.0, 0.0, 1.0],
    ]
)


def _require_bundle() -> None:
    if not BUNDLE_DIR.exists():
        pytest.fail(f"required dataset bundle is missing: {BUNDLE_DIR}", pytrace=False)


def _depth_image_path() -> Path:
    matches = sorted(glob.glob(str(BUNDLE_DIR / "standard" / "depth" / f"*__{SCENE_TIMESTAMP}_camera_l_depth.png")))
    if not matches:
        pytest.fail(f"depth frame not found for {SCENE_TIMESTAMP}", pytrace=False)
    return Path(matches[0])


def _fruit_points() -> np.ndarray:
    duckdb = pytest.importorskip("duckdb")
    path = BUNDLE_DIR / "grasp" / "recordings" / SCENE_ID / "scene_perception_snapshot.parquet"
    connection = duckdb.connect()
    try:
        shape, data = connection.execute(
            "SELECT fruit_points__shape, fruit_points__data FROM read_parquet(?)", [str(path)]
        ).fetchone()
    finally:
        connection.close()
    return np.asarray(data, dtype=np.float64).reshape(tuple(shape))


def _paired_depth_samples():
    """Return raw depth (mm) and the matching metric camera-Z (m) per fruit."""
    cv2 = pytest.importorskip("cv2")
    depth = cv2.imread(str(_depth_image_path()), cv2.IMREAD_UNCHANGED)
    assert depth is not None and depth.dtype == np.uint16

    points = _fruit_points()
    u = (DEPTH_K[0, 0] * points[:, 0] + DEPTH_K[0, 2] * points[:, 2]) / points[:, 2]
    v = (DEPTH_K[1, 1] * points[:, 1] + DEPTH_K[1, 2] * points[:, 2]) / points[:, 2]

    raw_samples = []
    metric_samples = []
    for index in range(points.shape[0]):
        ui, vi = round(float(u[index])), round(float(v[index]))
        if not (0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]):
            continue
        patch = depth[max(0, vi - 1) : vi + 2, max(0, ui - 1) : ui + 2]
        positives = patch[patch > 0]
        if positives.size == 0:
            continue
        raw_samples.append(float(np.median(positives)))
        metric_samples.append(float(points[index, 2]))
    return np.asarray(raw_samples), np.asarray(metric_samples)


def test_depth_scale_estimates_millimeter_units():
    _require_bundle()
    raw, metric = _paired_depth_samples()
    assert raw.size >= 10, "expected enough fruit samples projecting onto valid depth"
    scale = estimate_depth_scale(raw, metric)
    # Raw depth is stored in millimetres; metric axis is metres.
    assert 0.00095 < scale < 0.00105, f"unexpected depth scale {scale}"


def test_reprojection_median_error_within_50mm():
    _require_bundle()
    raw, metric = _paired_depth_samples()
    scale = estimate_depth_scale(raw, metric)
    errors = np.abs(metric - raw * scale)
    # Occasional fruits project onto an occluding/background surface, which the
    # simulation tolerates. The scale itself must be consistent for the bulk of
    # the cloud, so we assert on the median error and the hit rate.
    assert float(np.median(errors)) < 0.05, f"median reprojection error {np.median(errors):.4f} m"
    assert float((errors < 0.05).mean()) >= 0.6


def test_depth_scale_rejects_all_zero_signal():
    with pytest.raises(ValueError):
        estimate_depth_scale([0, 0, 0], [0.0, 0.0, 0.0])

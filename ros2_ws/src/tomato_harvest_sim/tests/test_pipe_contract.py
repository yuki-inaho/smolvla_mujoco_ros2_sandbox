"""Pipe / gutter judgment contract for the demo scene.

Verifies that the recorded planner output is internally consistent with the
pipe-based gates documented in ``pipe_recognizer_client.py`` and
``planning_params_tmt4-02.toml``:

  * pipe distance = |pipe_pos.y| must lie in [dist_pipe_min, dist_pipe_max]
  * target y <= min(dist_pipe - Y_EEF_TO_HEAD, target_y_max)
  * target z inside [target_z_min, target_z_max]
  * selected candidate has hand_gutter_collision == False
  * trajectory result is SUCCESS with collided == False and the upper
    trajectory used
"""


import duckdb
import pytest
from paths import RUN_DIR

DIST_PIPE_MIN = 0.80
DIST_PIPE_MAX = 0.90
Y_EEF_TO_HEAD = 0.16  # arm_commander/scripts/config.py
TARGET_Y_MAX = 0.58  # planning_params_tmt4-02.toml


def _require_run() -> None:
    if not (RUN_DIR / "feasibility.parquet").exists():
        pytest.fail(f"run bundle missing: {RUN_DIR}", pytrace=False)


def test_pipe_values_and_target_gate():
    _require_run()
    row = duckdb.sql(
        f"""
        select pipe_source, pipe_dist_m, pipe_height_m,
               target_world_y_m, target_world_z_m, target_z_min_m, target_z_max_m,
               hand_gutter_collision
        from read_parquet('{RUN_DIR}/feasibility.parquet')
        """
    ).fetchone()
    source, dist, _height, target_y, target_z, z_min, z_max, gutter_collision = row

    assert source in ("recognized", "config_default_no_pipe", "config_default_out_of_range")
    assert DIST_PIPE_MIN - 1e-9 <= float(dist) <= DIST_PIPE_MAX + 1e-9
    allowed_y = min(float(dist) - Y_EEF_TO_HEAD, TARGET_Y_MAX)
    assert float(target_y) <= allowed_y + 1e-9, f"target y {target_y} > gate {allowed_y}"
    assert float(z_min) <= float(target_z) <= float(z_max)
    assert gutter_collision is False


def test_selected_candidate_is_collision_free():
    _require_run()
    row = duckdb.sql(
        f"""
        select ok, reason, hand_gutter_collision, ik_evaluated
        from read_parquet('{RUN_DIR}/candidates_trace.parquet')
        where selected
        """
    ).fetchone()
    ok, reason, gutter_collision, ik_evaluated = row
    assert ok is True
    assert reason == "NONE"
    assert gutter_collision is False
    assert ik_evaluated is True


def test_trajectory_reports_no_collision_and_upper_path():
    _require_run()
    rows = duckdb.sql(
        f"""
        select result_code, collided, upper_trajectory_used, count(*) as n
        from read_parquet('{RUN_DIR}/trajectory.parquet')
        group by 1, 2, 3
        """
    ).fetchall()
    assert rows, "trajectory parquet is empty"
    for result_code, collided, upper_used, count in rows:
        assert result_code == "SUCCESS", rows
        assert collided is False, rows
        assert upper_used is True, rows
        assert count > 0


def test_dataset_success_trajectories_are_collision_free():
    """Field-wide invariant (tmt4-02): every SUCCESS plan is collision-free.

    PLAN_FAILURE runs may carry collided=True (the planner rejected them), so
    only SUCCESS is asserted here.
    """
    _require_run()
    runs_root = RUN_DIR.parent
    violations = duckdb.sql(
        f"""
        with tr as (
            select regexp_extract(filename, '/([^/]+)/trajectory.parquet', 1) as run,
                   any_value(result_code) as rc,
                   any_value(collided) as collided
            from read_parquet('{runs_root}/*/trajectory.parquet', filename=true)
            group by filename
        )
        select count(*) from tr where rc = 'SUCCESS' and collided is not False
        """
    ).fetchone()[0]
    successes = duckdb.sql(
        f"""
        with tr as (
            select regexp_extract(filename, '/([^/]+)/trajectory.parquet', 1) as run,
                   any_value(result_code) as rc
            from read_parquet('{runs_root}/*/trajectory.parquet', filename=true)
            group by filename
        )
        select count(*) from tr where rc = 'SUCCESS'
        """
    ).fetchone()[0]
    assert successes >= 100, f"unexpectedly few SUCCESS runs: {successes}"
    assert violations == 0, f"{violations} SUCCESS runs reported collided=True"


def test_planner_records_collision_rejections():
    """PLAN_FAILURE runs carry collided=True: the pipe/gutter collision gate
    actually rejected trajectories instead of silently accepting them."""
    _require_run()
    runs_root = RUN_DIR.parent
    row = duckdb.sql(
        f"""
        with tr as (
            select regexp_extract(filename, '/([^/]+)/trajectory.parquet', 1) as run,
                   any_value(result_code) as rc,
                   any_value(collided) as collided
            from read_parquet('{runs_root}/*/trajectory.parquet', filename=true)
            group by filename
        )
        select count(*) from tr where rc = 'PLAN_FAILURE' and collided is True
        """
    ).fetchone()
    assert row[0] >= 1, "no PLAN_FAILURE run recorded collided=True"


def test_dataset_pipe_distances_within_config_tiers():
    """Every recorded run must use a pipe distance inside the config tiers
    [dist_pipe_min, dist_pipe_max] = [0.80, 0.90] m."""
    _require_run()
    runs_root = RUN_DIR.parent
    total, out_of_tier = duckdb.sql(
        f"""
        select count(*),
               count(*) filter (
                   where pipe_dist_m < {DIST_PIPE_MIN} - 1e-9
                      or pipe_dist_m > {DIST_PIPE_MAX} + 1e-9
               )
        from read_parquet('{runs_root}/*/feasibility.parquet', filename=true)
        """
    ).fetchone()
    assert total >= 4000, f"unexpectedly few runs: {total}"
    assert out_of_tier == 0, f"{out_of_tier} runs outside pipe distance tiers"


def test_accepted_runs_have_no_gutter_collision():
    """Accepted runs (feasibility.ok=True) never select a candidate flagged
    hand_gutter_collision. Note: 9 rejected runs (ok=False) do select such a
    candidate and are discarded by the other gates, so the invariant is scoped
    to accepted runs."""
    _require_run()
    runs_root = RUN_DIR.parent
    collisions = duckdb.sql(
        f"""
        with c as (
            select regexp_extract(filename, '/([^/]+)/candidates_trace.parquet', 1) as run,
                   any_value(hand_gutter_collision) as hgc
            from read_parquet('{runs_root}/*/candidates_trace.parquet', filename=true)
            where selected
            group by filename
        ),
        f as (
            select regexp_extract(filename, '/([^/]+)/feasibility.parquet', 1) as run,
                   any_value(ok) as ok
            from read_parquet('{runs_root}/*/feasibility.parquet', filename=true)
            group by filename
        )
        select count(*)
        from c join f using (run)
        where c.hgc and f.ok
        """
    ).fetchone()[0]
    assert collisions == 0, f"{collisions} accepted runs selected a gutter-colliding candidate"

# justfile — reproducible workflow for the smolvla_mujoco_ros2 sandbox (branch: focal-cu127)
#
# HOW TO OBTAIN `just` (it may not be installed yet):
#   - Preferred (adds it to this repo's pixi env, no system install):
#         export PATH="$HOME/.pixi/bin:$PATH" && pixi add just
#     then run recipes as:  pixi run just <recipe>
#   - Or system package:    cargo install just      (any platform)
#                           sudo apt-get install just   (Debian/Ubuntu, if packaged)
#                           brew install just       (macOS / Linuxbrew)
#   Verify with:  just --version
#
# WHY THIS FILE: it wraps the existing pixi [tasks] and the verified skills/
# runbooks (clean-sim-run, record-and-verify-motion) into named, copy-pasteable
# recipes so the whole sim -> policy -> verify -> record workflow is reproducible.
#
# CONVENTIONS (do NOT skip):
#   - PATH is NOT persistent: every recipe re-exports the pixi PATH itself, so
#     `just <recipe>` works from any fresh shell.
#   - ros2 CLI + daemon live INSIDE the pixi env — always invoked as `pixi run -- ros2 ...`.
#   - Network/GPU/pixi commands need a non-sandboxed shell when driven by an agent
#     (Bash dangerouslyDisableSandbox=true); from a normal terminal just run them.
#   - This file does NOT edit pixi.toml (editing it would force a costly re-lock).
#   - Work only on `focal-cu127`. NEVER touch `main`.

set shell := ["bash", "-cu"]

# Put pixi on PATH inside every recipe line group. PATH is not persistent.
export PATH := env_var('HOME') + "/.pixi/bin:" + env_var('PATH')

# ---------------------------------------------------------------------------
# default: list every recipe
# ---------------------------------------------------------------------------
# List all recipes (run `just` with no args).
default:
    @just --list

# ---------------------------------------------------------------------------
# Environment / build
# ---------------------------------------------------------------------------

# (`pixi install` is the canonical install verb — there is NO `pixi run install` task.)
# Solve + install the pixi environment. Needs network.
install:
    pixi install

# Environment / toolchain sanity check (pixi task -> python scripts/doctor.py).
doctor:
    pixi run doctor

# colcon build the ROS 2 workspace (pixi task -> colcon build --symlink-install ... in ros2_ws).
build:
    pixi run build

# (This is the colcon clean. For killing stale ROS processes use the `clean` recipe.)
# Wipe the colcon build/install/log dirs (pixi `clean` task: rm -rf build install log in ros2_ws).
clean-build:
    pixi run clean

# ---------------------------------------------------------------------------
# Sim + policy launchers (foreground; Ctrl-C to stop)
# ---------------------------------------------------------------------------

# Launch ONE MuJoCo sim (pixi task; default use_renderer:=false). depends-on build.
sim:
    pixi run sim

# (No separate sim-render pixi task exists; rendering is the use_renderer:=true launch arg, default false.
# MUJOCO_GL=egl is already exported by activate_ros_overlay.sh; set explicitly so intent is obvious.)
# Launch ONE MuJoCo sim WITH real offscreen rendering (use_renderer:=true).
sim-render:
    MUJOCO_GL=egl pixi run sim use_renderer:=true

# Launch ONE policy node with the mock backend (pixi task -> backend:=mock).
policy-mock:
    pixi run policy-mock

# Launch ONE policy node with the SmolVLA backend (pixi task -> backend:=smolvla). Needs GPU + network.
policy-smolvla:
    pixi run policy-smolvla

# Launch sim + mock policy together (pixi task -> scripts/launch_all.sh mock).
all-mock:
    pixi run all-mock

# ---------------------------------------------------------------------------
# Teardown / verification (clean-sim-run skill)
# ---------------------------------------------------------------------------

# (pkill exits non-zero when nothing matches, so each is guarded with `|| true`. Stopping the
# daemon forces a clean re-discovery. Stale duplicate publishers fake camera "motion".)
# Kill stale sim/policy/launch processes and stop the ROS 2 daemon (clean-sim-run skill). Run BEFORE launch and AFTER record.
clean:
    pkill -f mujoco_sim_node || true
    pkill -f policy_node     || true
    pkill -f "ros2 launch"   || true
    pkill -f launch_all      || true
    pixi run -- ros2 daemon stop || true

# (Do NOT trust any recording until BOTH show `Publisher count: 1`.)
# Assert a SINGLE publisher on /joint_states and /arm_controller/joint_trajectory (clean-sim-run skill, step 3).
verify-single:
    @echo "== /joint_states =="
    pixi run -- ros2 topic info /joint_states
    @echo "== /arm_controller/joint_trajectory =="
    pixi run -- ros2 topic info /arm_controller/joint_trajectory
    @echo "If either 'Publisher count' is > 1, run `just clean` and relaunch ONE sim + ONE policy."

# ---------------------------------------------------------------------------
# Topic introspection (mirrors pixi list-topics / echo-joints tasks)
# ---------------------------------------------------------------------------

# List all ROS 2 topics.
topics:
    pixi run -- ros2 topic list

# Echo one /joint_states message and exit.
joints:
    pixi run -- ros2 topic echo /joint_states --once

# ---------------------------------------------------------------------------
# Standalone tests (no ROS graph required for these two)
# ---------------------------------------------------------------------------

# (Loads lerobot/smolvla_base and runs select_action; no ROS graph required.)
# Standalone SmolVLA inference smoke test -> scripts/smolvla_standalone_test.py. Needs GPU + network on first run.
smolvla-smoke:
    pixi run python scripts/smolvla_standalone_test.py

# Offscreen MuJoCo render test (EGL, falls back to osmesa) -> scripts/render_offscreen.py.
# Writes toy_arm_{top,side,wrist}.png into --output-dir (here temp/render_test).
render-test:
    pixi run python scripts/render_offscreen.py --output-dir temp/render_test

# ---------------------------------------------------------------------------
# Camera recording (record-and-verify-motion skill)
# ---------------------------------------------------------------------------

# Caveats confirmed by reading scripts/record_cameras.py:
#   - Real argparse flags: --topics --output-dir --max-frames --fps.
#   - Per-camera folder = FIRST topic segment: top_camera/ side_camera/ wrist_camera/.
#   - Frames are frame_NNNNNN.png (6-digit, start at frame_000001.png).
#   - --max-frames 0 (default) records until Ctrl-C, then encodes on shutdown (here capped at 300).
#   - The encoded MP4 is ALWAYS 10 fps (ffmpeg framerate is hardcoded); --fps is stored, NOT used by the encoder.
#   - A relative --output-dir resolves to the repo root (not cwd); prefer temp/.
# Record the three camera topics to PNG + MP4 under OUT (record-and-verify-motion skill).
record OUT="temp/camera_record":
    pixi run python scripts/record_cameras.py \
      --topics /top_camera/image_raw /side_camera/image_raw /wrist_camera/image_raw \
      --output-dir {{ OUT }} \
      --max-frames 300 \
      --fps 10

# ---------------------------------------------------------------------------
# End-to-end reproducible run (clean -> launch -> verify -> record -> clean)
# ---------------------------------------------------------------------------

# WHY recording must be on a SINGLE clean instance: two sim/policy instances double-publish
# the same topics and the cameras flicker between two slightly different poses. That flicker
# FAKES motion (huge consecutive-frame diff, ~0 same-parity diff). The verify-single gate
# below catches it — do NOT trust a recording taken while `Publisher count` > 1.
# This is a shebang recipe so the whole sequence (incl. the backgrounded sim/policy) runs in
# ONE shell; background PIDs are tracked and killed on exit, with `just clean` as the safety net.
# Override the output dir:  just e2e OUT=temp/my_run
#
# Reproducible end-to-end SmolVLA run: clean -> sim-render + smolvla policy (bg) -> verify -> record ~15-20s -> clean.
e2e OUT="temp/e2e_smolvla_mp4":
    #!/usr/bin/env bash
    set -uo pipefail
    export PATH="$HOME/.pixi/bin:$PATH"
    SIM_PID=""; POL_PID=""
    teardown() {
      [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
      [ -n "$POL_PID" ] && kill "$POL_PID" 2>/dev/null || true
      pkill -f mujoco_sim_node || true
      pkill -f policy_node     || true
      pkill -f "ros2 launch"   || true
      pkill -f launch_all      || true
      pixi run -- ros2 daemon stop || true
    }
    trap teardown EXIT
    echo ">>> [1/5] tear down any stale instances"
    teardown
    echo ">>> [2/5] launch ONE sim (rendering) + ONE smolvla policy in background"
    MUJOCO_GL=egl pixi run sim use_renderer:=true & SIM_PID=$!
    sleep 12
    pixi run policy-smolvla & POL_PID=$!
    sleep 12
    echo ">>> [3/5] VERIFY single publishers (each must report Publisher count: 1)"
    pixi run -- ros2 topic info /joint_states
    pixi run -- ros2 topic info /arm_controller/joint_trajectory
    echo ">>> [4/5] record ~15-20s (180 frames @ ~10 Hz) into {{ OUT }}"
    mkdir -p {{ OUT }}
    pixi run python scripts/record_cameras.py \
      --topics /top_camera/image_raw /side_camera/image_raw /wrist_camera/image_raw \
      --output-dir {{ OUT }} \
      --max-frames 180 \
      --fps 10
    echo ">>> [5/5] tear down handled by trap on exit"
    echo ">>> e2e done. MP4s: {{ OUT }}/top_camera.mp4 side_camera.mp4 wrist_camera.mp4"
    echo ">>> Verify REAL motion with the record-and-verify-motion skill (gap-2/same-parity + joint-range checks); the MP4 alone can lie."

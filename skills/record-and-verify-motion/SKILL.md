---
name: record-and-verify-motion
description: Record ROS camera topics to MP4 and rigorously decide whether there is real, continuous motion (not flicker, not static).
---

## When to use

Use this when you need to PROVE the policy actually drives continuous, directed
arm motion in the MuJoCo+ROS2 sandbox — not just claim "it moved" from eyeballing
a video. Eyeballing fails in two specific ways this runbook is built to catch:

- A 2-state topic FLICKER (multiple publishers alternating frames) that looks
  like motion but is just two repeating images.
- An arm that is SATURATED / frozen at its joint limits — the joint command
  changes but the rendered pose does not actually progress.

Run this after a clean single sim+policy launch when you must produce a defensible
verdict on motion.

## Steps

Every shell MUST start by putting pixi on PATH (PATH is not persistent):

```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

Commands that need network/GPU/pixi must run with the Bash tool's
`dangerouslyDisableSandbox=true`. Never touch the `main` branch (work on
`focal-cu127`).

### 1. Start one clean sim + policy first

Follow the `clean-sim-run` skill to bring up exactly one simulator + one policy
(no duplicate publishers — duplicates are the root cause of the flicker failure
mode). Save ALL artifacts under the repo's persistent `temp/` directory, NOT
`/tmp` (which is volatile and can vanish between steps):

```bash
export PATH="$HOME/.pixi/bin:$PATH"
mkdir -p /workspace/smolvla_mujoco_ros2_sandbox/temp/motion_check
```

### 2. Record the camera topics to PNG + MP4

`scripts/record_cameras.py` subscribes to the three image topics, writes numbered
PNGs, and encodes one MP4 per camera via ffmpeg. Use a persistent absolute
`--output-dir` under `temp/`. (If you pass a RELATIVE path it resolves to the
repo root, not the volatile cwd — but pass an absolute path under `temp/` to be
safe.)

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd /workspace/smolvla_mujoco_ros2_sandbox && \
pixi run python scripts/record_cameras.py \
  --topics /top_camera/image_raw /side_camera/image_raw /wrist_camera/image_raw \
  --output-dir /workspace/smolvla_mujoco_ros2_sandbox/temp/motion_check/cam \
  --max-frames 300 \
  --fps 10
```

Notes on the ACTUAL behavior (confirmed by reading the script):

- Per-camera frame dir is named from the FIRST path segment of the topic, so the
  folders are `top_camera/`, `side_camera/`, `wrist_camera/` (NOT `top/side/wrist`
  — the module docstring's example is wrong).
- Frames are `frame_NNNNNN.png`, zero-padded to 6 digits, starting at
  `frame_000001.png`.
- Output layout:
  `<output-dir>/top_camera/frame_000001.png`, ..., `<output-dir>/top_camera.mp4`
  (and likewise for `side_camera`, `wrist_camera`).
- `--max-frames 0` (the default) records until Ctrl-C / SIGINT / SIGTERM, then
  encodes on shutdown.
- The exact ffmpeg call is hardcoded to 10 fps and libx264:
  `ffmpeg -y -framerate 10 -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p <out>.mp4`.
  The `--fps` CLI flag is stored but NOT used by the encoder — output MP4 is
  always 10 fps regardless of `--fps`.

### 3. Compute motion metrics from the saved PNG sequence

Do NOT trust the MP4 by eye. Load the PNG sequence with numpy and compute, per
camera, ALL of the following. Point `FRAMES` at the `*_camera` frame dir.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd /workspace/smolvla_mujoco_ros2_sandbox && pixi run python - <<'PY'
import glob, numpy as np
from PIL import Image

FRAMES = "temp/motion_check/cam/top_camera"   # repeat for side_camera, wrist_camera
paths = sorted(glob.glob(f"{FRAMES}/frame_*.png"))
assert len(paths) >= 4, f"need >=4 frames, got {len(paths)}"
frames = np.stack([np.asarray(Image.open(p).convert("RGB"), np.float32) for p in paths])
N = len(frames)

# (a) first-vs-last mean |diff| / 255
first_last = np.abs(frames[0] - frames[-1]).mean() / 255.0

# (b) consecutive-frame diff vs SAME-PARITY (gap-2) diff.
# If consecutive is large but same-parity is ~0 -> 2-state FLICKER, not motion.
consec = np.array([np.abs(frames[i+1]-frames[i]).mean()/255.0 for i in range(N-1)])
parity = np.array([np.abs(frames[i+2]-frames[i]).mean()/255.0 for i in range(N-2)])
flicker = consec.mean() > 0.01 and parity.mean() < 0.1 * consec.mean()

# (c) arm-region diff: arm may be ~1% of an external view, so restrict the
# diff to the pixels that actually changed (full-frame mean hides small motion).
fl_abs = np.abs(frames[0] - frames[-1]).mean(axis=2)  # per-pixel
changed = fl_abs > (0.05 * 255)
arm_region = fl_abs[changed].mean()/255.0 if changed.any() else 0.0
arm_frac = changed.mean()

print(f"{FRAMES}: N={N}")
print(f"  first-vs-last mean|diff|/255 = {first_last:.5f}")
print(f"  consec mean = {consec.mean():.5f}  min = {consec.min():.5f}")
print(f"  same-parity(gap2) mean = {parity.mean():.5f}  min = {parity.min():.5f}")
print(f"  FLICKER artifact = {flicker}")
print(f"  changed-pixel fraction = {arm_frac:.4f}  arm-region mean|diff|/255 = {arm_region:.5f}")
PY
```

Also compute the joint-trajectory metrics from the recorded joint states /
command log produced by the sim run (e.g. echo `/joint_states` to a file, or use
the trajectory the policy emitted):

- joint trajectory `range > 0` for the driven joints (range == 0 => no command
  reaching the arm).
- net-displacement / path-length ratio per joint:
  `|q_last - q_first| / sum(|q_{i+1} - q_i|)`.
  - ratio near 1.0 => DIRECTED motion (good).
  - ratio near 0 => jitter / oscillation / static (the arm wiggles but goes
    nowhere — e.g. saturated at a limit).

### 4. Render the verdict

Declare REAL continuous motion ONLY if BOTH hold:

- same-parity (gap-2) diff is ELEVATED THROUGHOUT the clip (check `parity.min()`,
  not just the mean — a high mean can hide a frozen tail), AND
- joint `range > 0` for the driven joints.

If consecutive-frame diff is high but same-parity diff is ~0 => verdict is
FLICKER (multi-publisher), NOT motion. If joint range > 0 but the
net/path ratio is ~0 and arm-region diff collapses over the clip => the arm is
oscillating or saturated/frozen at a joint limit, NOT making real progress.

### Synthetic-vs-real image check (detect a fake/placeholder render)

Confirm the camera is a real EGL MuJoCo render and not a synthetic placeholder.
A synthetic test frame has a fixed structure:

- R channel = a per-COLUMN ramp => per-column variance ~0 (every row identical).
- G channel = a per-ROW ramp => per-row variance ~0 (every column identical).
- B channel = a single uniform value => B variance ~0.

A real EGL render breaks ALL THREE. Check it:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd /workspace/smolvla_mujoco_ros2_sandbox && pixi run python - <<'PY'
import numpy as np
from PIL import Image
img = np.asarray(Image.open("temp/motion_check/cam/top_camera/frame_000001.png").convert("RGB"), np.float32)
R, G, B = img[...,0], img[...,1], img[...,2]
r_colvar = R.var(axis=0).mean()   # variance down each column -> ~0 if R is a column ramp
g_rowvar = G.var(axis=1).mean()   # variance across each row   -> ~0 if G is a row ramp
b_var    = B.var()                # ~0 if B is uniform
print(f"R per-column var = {r_colvar:.4f} (synthetic ~0)")
print(f"G per-row var    = {g_rowvar:.4f} (synthetic ~0)")
print(f"B var            = {b_var:.4f} (synthetic ~0)")
synthetic = r_colvar < 1.0 and g_rowvar < 1.0 and b_var < 1.0
print(f"SYNTHETIC placeholder = {synthetic}  (real EGL render breaks all three)")
PY
```

If `SYNTHETIC` is True, the renderer is producing a placeholder — fix the EGL
render before trusting ANY motion metric.

## Why / gotchas

- This methodology exists because eyeballing the MP4 lied twice:
  1. A false "motion" that was actually TOPIC FLICKER — two publishers (a
     duplicate sim/policy) alternating frames. Consecutive-frame diff was big,
     but the same-parity (gap-2) diff was ~0: the clip was really just two
     images repeating. The gap-2 / same-parity check is what catches this; a
     plain consecutive-frame diff does NOT.
  2. A "moving" arm that was actually SATURATED / FROZEN at its joint limits. The
     joint command changed (range > 0) but net-displacement/path-length was ~0
     and the arm-region diff collapsed — wiggle, not progress. Requiring an
     elevated same-parity diff THROUGHOUT the clip (check the min) plus a
     directed net/path ratio is what catches this.
- The arm can occupy only ~1% of an external (top/side) view, so a full-frame
  mean|diff| can read near-zero even during real motion. Always also report the
  arm-region diff (mean over changed pixels) AND the changed-pixel fraction.
- Use a PERSISTENT path under the repo `temp/` for artifacts. `/tmp` is volatile
  and frames recorded there can disappear between recording and analysis,
  destroying the evidence.
- Folder names follow the topic's first path segment (`top_camera`, `side_camera`,
  `wrist_camera`), not the short `top/side/wrist` the docstring suggests — point
  your analysis at the right dirs.
- The encoded MP4 is always 10 fps (hardcoded in the ffmpeg call); `--fps` does
  not change it. The MP4 is for human spot-checks only — the VERDICT comes from
  the numpy metrics on the PNG sequence, not the video.
- Run pixi/GPU/ROS commands with `dangerouslyDisableSandbox=true`, start every
  shell with `export PATH="$HOME/.pixi/bin:$PATH"`, and never touch `main`.

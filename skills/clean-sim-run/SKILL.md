---
name: clean-sim-run
description: Launch ONE clean MuJoCo sim + policy on branch focal-cu127 and verify single publishers to avoid fake camera "motion" from topic flicker.
---

## When to use

Use this whenever you need a trustworthy MuJoCo sim + policy run for recording,
measuring, or debugging camera/joint behavior on the `focal-cu127` branch.
Specifically when:

- You are about to `ros2 bag record` or otherwise measure topic data and need
  the data to come from exactly one publisher.
- You see suspicious "motion" in camera frames (large consecutive-frame diffs)
  that may actually be two nodes flickering on the same topic.
- You are restarting after a crashed or backgrounded previous run.

Never run any of this against the `main` branch. Confirm you are on
`focal-cu127` first (`git branch --show-current`).

## Steps

Every shell must export the pixi PATH first (it is NOT persistent), and any
command that needs network/GPU/pixi must run with the Bash tool's
`dangerouslyDisableSandbox=true`.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

### 1. Tear down stale instances BEFORE launching

Stale sim/policy processes are the root cause of the flicker incident. Kill them
first, then stop the ROS 2 daemon so it re-discovers a clean graph. `pkill`
returns non-zero when nothing matches, so guard with `|| true`. The `ros2` CLI
and daemon come from the pixi env, so run via `pixi run`.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pkill -f mujoco_sim_node || true
pkill -f policy_node     || true
pkill -f launch_all      || true
pkill -f "ros2 launch"   || true
pixi run -- ros2 daemon stop || true
```

### 2. Launch ONE sim and ONE policy (background)

Launch a single sim with real rendering, then a single policy. `MUJOCO_GL=egl`
is already exported by `scripts/activate_ros_overlay.sh`, but set it explicitly
so the intent is obvious. Both run in the background.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
MUJOCO_GL=egl pixi run sim use_renderer:=true   # ONE sim, real rendering
pixi run policy-smolvla                          # or: pixi run policy-mock
```

`use_renderer:=true` is forwarded to the `use_renderer` launch arg
(default is `false`). `policy-smolvla` runs `backend:=smolvla`; `policy-mock`
runs `backend:=mock`.

### 3. VERIFY a single instance before recording/measuring

Do NOT trust any recording until each of these shows exactly one publisher.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi run -- ros2 topic info /joint_states                      # Publisher count: 1
pixi run -- ros2 topic info /arm_controller/joint_trajectory   # Publisher count: 1
pixi run -- ros2 node list                                     # one mujoco_sim_node + one policy_node
```

If either `Publisher count` is greater than 1, or `ros2 node list` shows
duplicate `mujoco_sim_node` / `policy_node` entries, STOP: go back to step 1,
kill everything, and retry. Only proceed once both counts equal 1.

### 4. Tear down again after use and re-verify

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pkill -f mujoco_sim_node || true
pkill -f policy_node     || true
pkill -f launch_all      || true
pkill -f "ros2 launch"   || true
pixi run -- ros2 daemon stop || true
pixi run -- ros2 node list   # expect: no mujoco_sim_node, no policy_node
```

## Why / gotchas

- **Real incident:** stale `mujoco_sim_node` + `policy_node` processes left
  running (launched from `/tmp`) double-published to the same topics. The two
  sims rendered slightly different camera poses, so frames flickered between the
  two nodes' views. This looked like large motion — the **consecutive-frame diff
  was huge** — while the **same-parity (every-other-frame) diff was ~0**, the
  tell-tale signature of two interleaved publishers, not real movement.
- **The mitigation is the verification, not just the kill.** `pkill` can miss a
  process or a launch can race; always confirm `Publisher count: 1` on
  `/joint_states` AND `/arm_controller/joint_trajectory`, plus a single node of
  each name, before trusting ANY recording or measurement.
- `pkill` exits non-zero when no process matches — always append `|| true` so
  the step does not abort an `set -e` shell.
- Stop the ROS 2 daemon after killing processes; otherwise it can keep serving a
  stale discovery graph and `ros2 node list` may show ghosts.
- PATH is not persistent: `export PATH="$HOME/.pixi/bin:$PATH"` at the top of
  every shell. Run network/GPU/pixi commands with
  `dangerouslyDisableSandbox=true`.
- Never operate on `main`; this runbook is for `focal-cu127`.

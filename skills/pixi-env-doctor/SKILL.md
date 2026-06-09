---
name: pixi-env-doctor
description: Diagnose pixi + RoboStack(ROS2 Humble) + CUDA env health on this Focal/A5000 box; run first after any env or dependency change.
---

## When to use

Run this whenever something "won't import", `cuda_available` is unexpectedly
`False`, or you just changed `pixi.toml` / re-solved the lock. On this machine
(Ubuntu 20.04 Focal, NVIDIA RTX A5000, driver 565 / CUDA 12.7) almost every
import or CUDA failure traces back to a single dependency pin drifting. The
doctor is the first thing to run after any environment change, before launching
sim, the policy node, or training.

Never operate on the `main` branch; this repo's working branch is `focal-cu127`.

## Steps

All commands need network/GPU/pixi, so run them with the Bash tool's
`dangerouslyDisableSandbox=true`. PATH is NOT persistent — every shell must
start by exporting the pixi bin dir.

1) Run the full doctor:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi run doctor
```

Expect every line prefixed `[OK]` (failures show `[NG]`):
- `platform` Linux
- `python` 3.12
- `PIXI_PROJECT_ROOT` set, `MUJOCO_GL` set
- `ros2 --help`, `rclpy`, `sensor_msgs`, `trajectory_msgs`, `mujoco`, `lerobot` import/run OK
- `torch` line showing `torch=2.7.x`, `cuda_available=True`, `device=NVIDIA RTX A5000`

`pixi run doctor` exits non-zero if any check is `[NG]`.

2) Direct CUDA sanity check (bypasses doctor's wrapper):

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)"
```

Expect: `True NVIDIA RTX A5000 12.6` (CUDA runtime baked into the torch wheel;
the host driver reports 12.7 via `system-requirements`).

## Why / gotchas

The pins below exist for hard, real reasons — if a re-solve drifts off any of
them, the doctor will go `[NG]`. Pins are in `pixi.toml`:

- `python >=3.12,<3.13`: forced by `lerobot[smolvla]` (PyPI 0.5.1 requires
  Python >=3.12) and by RoboStack humble shipping cp39/cp311/cp312 builds (no 3.10).
- `setuptools >=68,<80`: upstream's `setuptools<=58` pin breaks on 3.12 because
  stdlib `distutils` was removed and setuptools 58 does not vendor it; the upper
  bound `<80` keeps `setup.py develop` (needed by `colcon build --symlink-install`),
  which setuptools 80 removed.
- `numpy >=2.0.0,<2.3.0`: lerobot 0.5.x requires numpy >=2.0,<2.3.
- `packaging >=24.2,<26.0`: lerobot requires packaging <26.0, but conda-forge
  defaults to 26.x which causes a PyPI solver conflict.
- `lerobot >=0.5.1,<0.6.0` (extras `smolvla`): needs
  opencv-python-headless >=4.9.0,<4.14.0 (conda's 4.13.0 satisfies this).
- `torch >=2.7.0,<2.8.0` / `torchvision >=0.22.0,<0.23.0`: the cu128 NIGHTLY
  (torch 2.10.0+cu128) removed `torch._dynamo.utils.NP_SUPPORTED_MODULES`, which
  torchvision 0.25.0 depends on, breaking the import chain. Pinning torch to a
  stable 2.7.x build makes the solver pick a compatible torchvision 0.22.x.
- `system-requirements.cuda = "12.7"`: lets the solver pick GPU-enabled builds
  and fails loudly if a package needs a newer toolkit than the host driver
  (565 / CUDA 12.7) supports.

If `lerobot`/`torch` are `[NG]` but ros2/rclpy are `[OK]`, mock-policy ROS 2
plumbing still works, but SmolVLA inference will not — fix the pin before
running `policy-smolvla` / `train-smolvla`.

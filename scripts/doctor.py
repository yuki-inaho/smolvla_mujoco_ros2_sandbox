#!/usr/bin/env python3
"""Environment diagnostics for the pixi ROS 2 + MuJoCo + SmolVLA sandbox."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def try_import(module: str) -> CheckResult:
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "version unknown")
        return CheckResult(module, True, str(version))
    except Exception as exc:  # noqa: BLE001 - diagnostics should show any import failure.
        return CheckResult(module, False, f"{type(exc).__name__}: {exc}")


def run_command(cmd: list[str]) -> CheckResult:
    if shutil.which(cmd[0]) is None:
        return CheckResult(" ".join(cmd), False, "command not found")
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
        first_line = (proc.stdout or proc.stderr).strip().splitlines()[0:1]
        return CheckResult(" ".join(cmd), True, first_line[0] if first_line else "ok")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(" ".join(cmd), False, f"{type(exc).__name__}: {exc}")


def check_torch() -> CheckResult:
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = f"torch={torch.__version__}, cuda_available={cuda}"
        if cuda:
            detail += f", device={torch.cuda.get_device_name(0)}"
        return CheckResult("torch", True, detail)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("torch", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    checks = [
        CheckResult("platform", platform.system() == "Linux", platform.platform()),
        CheckResult("python", sys.version_info[:2] == (3, 10), sys.version.replace("\n", " ")),
        CheckResult("PIXI_PROJECT_ROOT", bool(os.environ.get("PIXI_PROJECT_ROOT")), os.environ.get("PIXI_PROJECT_ROOT", "not set")),
        CheckResult("MUJOCO_GL", bool(os.environ.get("MUJOCO_GL")), os.environ.get("MUJOCO_GL", "not set")),
        run_command(["ros2", "--help"]),
        try_import("rclpy"),
        try_import("sensor_msgs"),
        try_import("trajectory_msgs"),
        try_import("mujoco"),
        try_import("lerobot"),
        check_torch(),
    ]

    width = max(len(c.name) for c in checks)
    failed = False
    for check in checks:
        status = "OK" if check.ok else "NG"
        print(f"[{status}] {check.name:<{width}} {check.detail}")
        failed = failed or not check.ok

    if failed:
        print("\nSome checks failed. Mock-policy ROS 2 plumbing may still work, but SmolVLA inference requires lerobot/torch to import correctly.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

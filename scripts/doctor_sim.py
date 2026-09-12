#!/usr/bin/env python3
"""Environment diagnostics for the tomato_harvest_sim work (sim-scope only).

This intentionally excludes lerobot/torch, which are out of scope for the
harvest simulation and are documented as not installed in pixi.toml.
"""
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
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        first_line = (proc.stdout or proc.stderr).strip().splitlines()[0:1]
        return CheckResult(" ".join(cmd), True, first_line[0] if first_line else "ok")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(" ".join(cmd), False, f"{type(exc).__name__}: {exc}")


def check_numpy_pin() -> CheckResult:
    try:
        import numpy as np

        ok = int(np.__version__.split(".")[0]) < 2
        return CheckResult("numpy<2", ok, f"numpy={np.__version__}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("numpy<2", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    checks = [
        CheckResult("platform", platform.system() == "Linux", platform.platform()),
        CheckResult("python", sys.version_info[:2] == (3, 11), sys.version.replace("\n", " ")),
        CheckResult("PIXI_PROJECT_ROOT", bool(os.environ.get("PIXI_PROJECT_ROOT")), os.environ.get("PIXI_PROJECT_ROOT", "not set")),
        run_command(["ros2", "--help"]),
        try_import("rclpy"),
        try_import("sensor_msgs"),
        try_import("trajectory_msgs"),
        try_import("mujoco"),
        try_import("duckdb"),
        try_import("numpy"),
        check_numpy_pin(),
    ]

    width = max(len(c.name) for c in checks)
    failed = False
    for check in checks:
        status = "OK" if check.ok else "NG"
        print(f"[{status}] {check.name:<{width}} {check.detail}")
        failed = failed or not check.ok

    if failed:
        print("\nSome checks failed. See pixi.toml comments for the documented scope.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

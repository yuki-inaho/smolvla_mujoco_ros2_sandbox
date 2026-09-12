#!/usr/bin/env python3
"""Drive playwright-cli against the harvest viewer and write a JSON report.

Checks (all must be true for exit 0):
  1. `phase` is present in window.__harvestState
  2. the EEF moved more than 0.05 m between two samples
  3. scene point count > 1000
  4. `frame_index` increased between two samples
  5. three screenshots were written and each is larger than 1 KiB
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "webcheck"
URL = "http://127.0.0.1:8765/"
SESSION = "harvest"
EVAL = "() => window.__harvestState"
CONFIG = ROOT / ".playwright" / "cli.config.json"


def _run(*args: str, timeout: float = 90.0, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["playwright-cli", f"-s={SESSION}", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"playwright-cli {' '.join(args)} failed rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _read_state(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"eval output file was not created: {path}")
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise RuntimeError(f"could not find JSON in eval output: {text[:200]}")
        return json.loads(text[start : end + 1])


def _distance(a, b) -> float:
    if not a or not b:
        return 0.0
    return math.dist(list(a), list(b))


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    state_dir = ARTIFACTS / "state"
    state_dir.mkdir(exist_ok=True)
    report: dict = {"url": URL, "samples": [], "screenshots": [], "checks": {}}

    _run("open", URL, "--browser", "chrome", "--config", str(CONFIG))
    try:
        samples = []
        for attempt in range(4):
            sample_path = state_dir / f"sample_{attempt + 1}.json"
            _run("eval", EVAL, "--filename", str(sample_path))
            samples.append(_read_state(sample_path))
            if attempt < 3:
                time.sleep(1.2)
        report["samples"] = samples

        screenshots = []
        for index in range(3):
            target = ARTIFACTS / f"viewer_{index + 1}.png"
            _run("screenshot", "--filename", str(target))
            screenshots.append(
                {"path": str(target), "bytes": target.stat().st_size if target.exists() else 0}
            )
        report["screenshots"] = screenshots
    finally:
        _run("close", check=False)

    max_move = 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            max_move = max(
                max_move, _distance(samples[i].get("eef"), samples[j].get("eef"))
            )
    frame_increased = any(
        int(samples[j].get("frame_index", 0)) > int(samples[i].get("frame_index", 0))
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
    )
    last = samples[-1]
    report["eef_move_m"] = max_move
    report["checks"] = {
        "phase_present": bool(last.get("phase")),
        "eef_moved_gt_0_05m": max_move > 0.05,
        "points_gt_1000": int(last.get("points_count", 0)) > 1000,
        "frame_index_increased": frame_increased,
        "screenshots_over_1kb": len(report["screenshots"]) == 3
        and all(item["bytes"] > 1024 for item in report["screenshots"]),
    }
    report["all_true"] = all(report["checks"].values())
    report_path = ARTIFACTS / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_true"] else 1


if __name__ == "__main__":
    sys.exit(main())

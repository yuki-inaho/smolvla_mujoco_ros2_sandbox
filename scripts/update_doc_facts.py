#!/usr/bin/env python3
"""Regenerate the machine-checked fact block in the spec (A-5).

Facts are generated from reality (pytest, the rendered artifact, ``wc -l``) and
written between the ``BEGIN/END GENERATED:facts`` markers. Run with ``--check``
in CI: it exits non-zero when the block is stale, which is the docs equivalent
of ``git diff --exit-code`` after regeneration.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_TESTS = ROOT / "ros2_ws" / "src" / "tomato_harvest_sim" / "tests"
SPEC = Path(
    "/home/kasm-user/Desktop/workdocs/spec_Sep12-2026_tomato_harvest_sim_3d_fidelity.md"
)
VIDEO = ROOT / "artifacts" / "video" / "harvest_realtime_30s_faithful.mp4"
SOURCES = {
    "node": ROOT
    / "ros2_ws"
    / "src"
    / "tomato_harvest_sim"
    / "tomato_harvest_sim"
    / "harvest_sim_node.py",
    "render": ROOT / "scripts" / "render_harvest_video.py",
    "viewer": ROOT
    / "ros2_ws"
    / "src"
    / "tomato_harvest_sim"
    / "tomato_harvest_sim"
    / "viewer_node.py",
}
BEGIN = "<!-- BEGIN GENERATED:facts -->"
END = "<!-- END GENERATED:facts -->"


def passed_count() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(PKG_TESTS), "-q"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    match = re.search(r"(\d+) passed", result.stdout)
    if result.returncode != 0 or match is None:
        raise SystemExit(f"pytest is not clean (rc={result.returncode})")
    return int(match.group(1))


def video_facts() -> tuple[int, int, float]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=nb_frames",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(VIDEO),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(f"ffprobe failed: {probe.stderr.strip()}")
    values = [line for line in probe.stdout.splitlines() if line.strip()]
    frames, duration = int(values[0]), float(values[1])
    return VIDEO.stat().st_size, frames, duration


def render_block() -> str:
    size, frames, duration = video_facts()
    lines = {name: len(path.read_text().splitlines()) for name, path in SOURCES.items()}
    return "\n".join(
        [
            BEGIN,
            "| 事実 | 値（自動生成: `pixi run doc-facts`） |",
            "| :--- | :--- |",
            f"| pytest | {passed_count()} passed |",
            f"| faithful video | {size:,} B / {frames} f / {duration:.3f} s / 1280×720 |",
            (
                "| line counts | "
                f"node {lines['node']} / render {lines['render']} / "
                f"viewer {lines['viewer']} |"
            ),
            END,
        ]
    )


def replace_block(text: str, block: str) -> str:
    pattern = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(f"generated block markers not found in {SPEC}")
    return pattern.sub(block, text, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()

    original = SPEC.read_text()
    updated = replace_block(original, render_block())
    if updated == original:
        print("doc facts are current")
        return 0
    if args.check:
        print("doc facts are stale; run `pixi run doc-facts`")
        return 1
    SPEC.write_text(updated)
    print("doc facts regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

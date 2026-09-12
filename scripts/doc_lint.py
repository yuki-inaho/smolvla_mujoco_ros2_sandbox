#!/usr/bin/env python3
"""Deterministic documentation lint for the tomato harvest workdocs.

Checks (no LLM, no network):
  1. No broken table rows (``||``) inside markdown tables (inline code spans,
     ``~~~`` fences, indented code and blockquotes are handled).
  2. Every table block keeps a consistent number of columns.
  3. Spec facts match reality by executing the checks, not by trimming:
     - the faithful-render byte size equals the artifact size;
     - the spec's pytest count equals a real ``pytest -q`` pass count;
     - the spec's machine-readable line counts equal ``wc -l`` of the sources.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_TESTS = ROOT / "ros2_ws" / "src" / "tomato_harvest_sim" / "tests"
SPEC = Path(
    os.environ.get(
        "DOC_LINT_SPEC",
        "/home/kasm-user/Desktop/workdocs/spec_Sep12-2026_tomato_harvest_sim_3d_fidelity.md",
    )
)
VIDEO = ROOT / "artifacts" / "video" / "harvest_realtime_30s_faithful.mp4"
LINE_COUNT_TARGETS = {
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
def default_targets() -> list[Path]:
    """Maintained docs only: workdocs plus our own temp workdocs/reports.

    Agent-written review fragments and transcripts under temp are evidence,
    not maintained documents, so they are intentionally out of scope.
    """
    temp = Path("/home/kasm-user/Desktop/temp")
    targets = [Path("/home/kasm-user/Desktop/workdocs")]
    targets.extend(sorted(temp.glob("workdoc*.md")))
    targets.extend(sorted(temp.glob("report*.md")))
    return targets


def iter_markdown(targets: Iterable[Path]) -> Iterable[Path]:
    for target in targets:
        if target.is_dir():
            yield from sorted(target.rglob("*.md"))
        elif target.suffix == ".md" and target.exists():
            yield target


def strip_inline_code(line: str) -> str:
    return re.sub(r"`[^`]*`", "", line)


def count_pipes(line: str) -> int:
    """Count real table separators, ignoring escaped ``\\|`` cells."""
    return len(re.findall(r"(?<!\\)\|", line))


def find_table_problems(path: Path) -> list[str]:
    problems: list[str] = []
    in_fence: str | None = None
    block_columns: int | None = None
    block_start = 0
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if in_fence is not None:
            if stripped.startswith(in_fence):
                in_fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            in_fence = stripped[:3]
            continue
        if line.startswith(("    ", "\t")):
            continue
        candidate = line.lstrip()
        if candidate.startswith(">"):
            candidate = candidate[1:].lstrip()
        if not candidate.startswith("|"):
            block_columns = None
            continue
        candidate = strip_inline_code(candidate)
        if re.search(r"(?<!\\)\|\|", candidate):
            problems.append(f"{path}:{number}: broken table row (double pipe)")
        columns = count_pipes(candidate)
        if block_columns is None:
            block_columns = columns
            block_start = number
        elif columns != block_columns:
            problems.append(
                f"{path}:{number}: table column mismatch (block from line "
                f"{block_start}: expected {block_columns} pipes, got {columns})"
            )
    return problems


def run_test_count() -> int | None:
    """Execute the suite; return the passed count or None when it is unhealthy."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(PKG_TESTS), "-q"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    match = re.search(r"(\d+) passed", result.stdout)
    if result.returncode != 0 or match is None:
        tail = (result.stdout or result.stderr).strip().splitlines()[-3:]
        print(f"NG: pytest is not clean (rc={result.returncode})")
        for item in tail:
            print(f"    {item}")
        return None
    return int(match.group(1))


def check_video_size(text: str) -> list[str]:
    line = next(
        (
            item
            for item in text.splitlines()
            if "harvest_realtime_30s_faithful.mp4" in item
            and re.search(r"[0-9,]+ B", item)
        ),
        None,
    )
    if line is None:
        line = next(
            (
                item
                for item in text.splitlines()
                if "動画" in item and re.search(r"[0-9,]+ B", item)
            ),
            None,
        )
    if line is None:
        return ["spec: no row with the faithful-render byte size"]
    if not VIDEO.exists():
        return [f"video artifact missing: {VIDEO}"]
    match = re.search(r"([0-9][0-9,]*) B", line)
    if match is None:
        return ["spec: cannot parse the faithful-render byte size"]
    claimed = int(match.group(1).replace(",", ""))
    actual = VIDEO.stat().st_size
    if claimed != actual:
        return [f"spec: video size claimed {claimed} != actual {actual}"]
    return []


def check_line_counts(text: str) -> list[str]:
    match = re.search(
        r"実測）: node (\d+)、render (\d+)、viewer (\d+)", text
    )
    if match is None:
        return [
            (
                "spec: no machine-readable line-count row "
                "(expected: 実測）: node N、render N、viewer N)"
            )
        ]
    problems: list[str] = []
    for key, value in zip(("node", "render", "viewer"), match.groups()):
        actual = len(LINE_COUNT_TARGETS[key].read_text().splitlines())
        if int(value) != actual:
            problems.append(
                f"spec: {key} line count claimed {value} != actual {actual}"
            )
    return problems


def check_spec_facts() -> list[str]:
    if not SPEC.exists():
        return [f"spec missing: {SPEC}"]
    text = SPEC.read_text()
    problems = check_video_size(text)
    problems.extend(check_line_counts(text))

    match = re.search(r"\*\*(\d+) passed\*\*", text)
    passed = run_test_count()
    if match is None:
        problems.append("spec: no `**N passed**` claim")
    elif passed is None:
        problems.append("spec: pytest is not clean; cannot verify the passed claim")
    elif int(match.group(1)) != passed:
        problems.append(f"spec: says {match.group(1)} passed, pytest reports {passed}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        type=Path,
        nargs="*",
        default=default_targets(),
        help="markdown files or directories (default: workdocs + our temp docs)",
    )
    args = parser.parse_args()

    problems: list[str] = []
    for path in iter_markdown(args.targets):
        problems.extend(find_table_problems(path))
    problems.extend(check_spec_facts())

    if problems:
        for problem in problems:
            print(f"NG: {problem}")
        print(f"doc lint failed: {len(problems)} problem(s)")
        return 1
    print("doc lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

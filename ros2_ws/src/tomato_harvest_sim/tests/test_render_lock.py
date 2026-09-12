"""Renderer output contract (A-3).

Renders to a given output path must be single-writer, and the final file may
only appear after the temp render fully decodes with the expected shape.
These tests exercise the helpers directly; the truncation test uses ffmpeg.
"""

import fcntl
import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "render_harvest_video.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "render_harvest_video_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exclusive_output_publishes_after_success(tmp_path, monkeypatch):
    render = _module()
    monkeypatch.setattr(render, "_verify_video", lambda *args, **kwargs: None)
    out = tmp_path / "video.mp4"
    with render.exclusive_output(out) as temp_path:
        assert temp_path != out
        temp_path.write_bytes(b"rendered")
    assert out.read_bytes() == b"rendered"


def test_exclusive_output_rejects_second_writer(tmp_path):
    render = _module()
    out = tmp_path / "video.mp4"
    lock_path = out.with_suffix(out.suffix + ".lock")
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError), render.exclusive_output(out):
            pass


def test_exclusive_output_cleans_temp_on_failure(tmp_path, monkeypatch):
    render = _module()
    monkeypatch.setattr(render, "_verify_video", lambda *args, **kwargs: None)
    out = tmp_path / "video.mp4"
    with pytest.raises(ValueError), render.exclusive_output(out) as temp_path:
        temp_path.write_bytes(b"partial")
        raise ValueError("render failed")
    assert not out.exists()
    assert not temp_path.exists()


def test_exclusive_output_cleans_stale_temp_and_keeps_suffix(tmp_path, monkeypatch):
    render = _module()
    monkeypatch.setattr(render, "_verify_video", lambda *args, **kwargs: None)
    out = tmp_path / "clip.mkv"
    stale = tmp_path / "clip.partial-999.mkv"
    stale.write_bytes(b"stale")
    with render.exclusive_output(out, expected_frames=1) as temp_path:
        assert temp_path.suffix == ".mkv"
        assert not stale.exists()
        temp_path.write_bytes(b"fresh")
    assert out.read_bytes() == b"fresh"


def test_verify_video_rejects_truncated_output(tmp_path):
    render = _module()
    good = tmp_path / "good.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x48:r=10",
            "-t", "0.5", "-pix_fmt", "yuv420p", str(good),
        ],
        check=True,
    )
    render._verify_video(good, expected_frames=5, width=64, height=48)
    bad = tmp_path / "bad.mp4"
    data = good.read_bytes()
    bad.write_bytes(data[: max(len(data) // 8, 1)])
    with pytest.raises(RuntimeError):
        render._verify_video(bad, expected_frames=5, width=64, height=48)

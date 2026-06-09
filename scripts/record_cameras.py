#!/usr/bin/env python3
"""Subscribe to camera image topics and save frames as PNG, then encode to MP4.

Subscribes to /top_camera/image_raw, /side_camera/image_raw, and
/wrist_camera/image_raw (configurable) and writes numbered PNG frames under
<output_dir>/<camera_name>/frame_NNNNNN.png.

After recording finishes (SIGINT or --max-frames reached), each camera's
frames are assembled into an MP4 using ffmpeg.

Usage:
    # Source ROS and pixi env, then:
    pixi run python scripts/record_cameras.py --output-dir /tmp/cam_record

    # Record at most 300 frames per camera (default 0 = unlimited):
    pixi run python scripts/record_cameras.py --max-frames 300 --fps 10

    # Custom topics:
    pixi run python scripts/record_cameras.py \\
        --topics /top_camera/image_raw /side_camera/image_raw /wrist_camera/image_raw

Output structure:
    <output_dir>/top/frame_000001.png ...
    <output_dir>/side/frame_000001.png ...
    <output_dir>/wrist/frame_000001.png ...
    <output_dir>/top.mp4
    <output_dir>/side.mp4
    <output_dir>/wrist.mp4
"""

from __future__ import annotations

import argparse
import os
import signal
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from threading import Lock
from typing import Dict, List

# ---------------------------------------------------------------------------
# Minimal PNG writer (no PIL dep needed at import time)
# ---------------------------------------------------------------------------


def _write_png(path: Path, rgb_bytes: bytes, width: int, height: int) -> None:
    """Write raw RGB bytes as a PNG file."""
    import numpy as np
    arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
    _numpy_to_png(arr, path)


def _numpy_to_png(arr, path: Path) -> None:
    import numpy as np

    arr = np.asarray(arr, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image as PILImage
        PILImage.fromarray(arr).save(str(path))
        return
    except ImportError:
        pass

    h, w = arr.shape[:2]

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw_rows = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    with open(str(path), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(_chunk(b"IDAT", zlib.compress(raw_rows, 6)))
        f.write(_chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------


class CameraRecorder:
    """ROS2 node that records camera topics to PNG frames."""

    def __init__(
        self,
        topics: List[str],
        output_dir: Path,
        max_frames: int = 0,
        fps: float = 10.0,
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        rclpy.init()
        self._node = Node("camera_recorder")
        self._output_dir = output_dir
        self._max_frames = max_frames
        self._fps = fps
        self._lock = Lock()
        self._counts: Dict[str, int] = {}
        self._done = False

        # Derive a short name from each topic for folder naming.
        for topic in topics:
            # e.g. /top_camera/image_raw -> top_camera
            name = topic.strip("/").split("/")[0]
            self._counts[name] = 0
            (output_dir / name).mkdir(parents=True, exist_ok=True)

            def _cb(msg: "Image", _name: str = name) -> None:
                self._on_image(msg, _name)

            self._node.create_subscription(Image, topic, _cb, 10)
            self._node.get_logger().info(f"Subscribed to {topic} -> {output_dir / name}/")

    def _on_image(self, msg, camera_name: str) -> None:
        with self._lock:
            if self._done:
                return
            n = self._counts[camera_name]
            if self._max_frames > 0 and n >= self._max_frames:
                return
            self._counts[camera_name] = n + 1

        path = self._output_dir / camera_name / f"frame_{n + 1:06d}.png"

        # Convert sensor_msgs/Image to numpy and save.
        import numpy as np
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if msg.encoding in ("rgb8", "bgr8"):
            arr = arr.reshape(msg.height, msg.width, 3)
            if msg.encoding == "bgr8":
                arr = arr[..., ::-1]
        elif msg.encoding == "mono8":
            arr = np.stack([arr.reshape(msg.height, msg.width)] * 3, axis=-1)
        else:
            # Attempt raw reshape
            arr = arr.reshape(msg.height, msg.width, -1)[..., :3]

        _numpy_to_png(arr, path)

        with self._lock:
            if self._max_frames > 0 and all(
                v >= self._max_frames for v in self._counts.values()
            ):
                self._done = True

    def spin(self) -> None:
        import rclpy
        node = self._node

        def _shutdown(_sig, _frame):
            nonlocal self
            with self._lock:
                self._done = True

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            while not self._done:
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.shutdown()
            self._encode_all()

    def _encode_all(self) -> None:
        """Assemble PNGs into MP4 using ffmpeg for each camera."""
        for camera_name, count in self._counts.items():
            frame_dir = self._output_dir / camera_name
            mp4_path = self._output_dir / f"{camera_name}.mp4"
            if count == 0:
                print(f"  {camera_name}: no frames received, skipping MP4 encode")
                continue
            self._ffmpeg_encode(frame_dir, mp4_path)
            print(f"  {camera_name}: {count} frames -> {mp4_path}")

    @staticmethod
    def _ffmpeg_encode(frame_dir: Path, out_mp4: Path) -> None:
        """Run ffmpeg to encode numbered PNGs into MP4."""
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "10",
            "-i", str(frame_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_mp4),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ffmpeg failed for {out_mp4}:\n{result.stderr[-500:]}")
        else:
            print(f"  ffmpeg OK: {out_mp4}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topics",
        nargs="+",
        default=[
            "/top_camera/image_raw",
            "/side_camera/image_raw",
            "/wrist_camera/image_raw",
        ],
        help="ROS2 image topics to subscribe to",
    )
    parser.add_argument(
        "--output-dir",
        default="temp/camera_record",
        help="Directory to write PNG frames and MP4s (default: temp/camera_record)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames per camera (0 = unlimited, Ctrl-C to stop)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback FPS for output MP4 (default: 10)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Recording {args.topics} -> {output_dir}")
    print("Press Ctrl-C to stop and encode MP4s.")

    recorder = CameraRecorder(
        topics=args.topics,
        output_dir=output_dir,
        max_frames=args.max_frames,
        fps=args.fps,
    )
    recorder.spin()


if __name__ == "__main__":
    main()

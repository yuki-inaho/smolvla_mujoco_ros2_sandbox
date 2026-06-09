#!/usr/bin/env python3
"""Standalone offscreen rendering test for toy_arm.xml.

Renders one frame per camera (top / side / wrist) using mujoco.Renderer and
saves the results as PNG files.  Tries MUJOCO_GL=egl first; falls back to
osmesa if EGL initialisation fails.

Usage (from repo root):
    pixi run python scripts/render_offscreen.py [--output-dir <dir>]

Output:
    <output_dir>/toy_arm_top.png
    <output_dir>/toy_arm_side.png
    <output_dir>/toy_arm_wrist.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _set_backend(backend: str) -> None:
    """Set MUJOCO_GL env var before mujoco is imported."""
    os.environ["MUJOCO_GL"] = backend
    # Remove any cached mujoco modules so the env var takes effect if this
    # function is called a second time (osmesa fallback path).
    for key in list(sys.modules.keys()):
        if key.startswith("mujoco"):
            del sys.modules[key]


def render_arm(model_path: Path, output_dir: Path, width: int = 320, height: int = 240) -> list[Path]:
    """Load toy_arm, run a step, render all cameras, save PNGs.

    Returns list of saved file paths.
    """
    import mujoco  # imported after MUJOCO_GL is set

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Advance a handful of steps so gravity settles the arm to a non-flat pose.
    for _ in range(200):
        mujoco.mj_step(model, data)

    camera_names = ["top", "side", "wrist"]
    saved: list[Path] = []

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for cam in camera_names:
            renderer.update_scene(data, camera=cam)
            img = renderer.render()  # returns numpy (H, W, 3) uint8

            if img is None or img.size == 0:
                raise RuntimeError(f"render() returned empty array for camera={cam}")

            out_path = output_dir / f"toy_arm_{cam}.png"
            _save_png(img, out_path)
            saved.append(out_path)
            print(f"  saved {out_path}  shape={img.shape}  "
                  f"min={img.min()} max={img.max()}")

    return saved


def _save_png(img, path: Path) -> None:
    """Save an HxWx3 uint8 numpy array as PNG.

    Uses PIL if available, otherwise writes a minimal PNG by hand.
    """
    import numpy as np

    arr = np.asarray(img, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image as PILImage
        PILImage.fromarray(arr).save(str(path))
        return
    except ImportError:
        pass

    # Minimal PNG writer (no external deps needed).
    import struct
    import zlib

    h, w = arr.shape[:2]

    def _make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw_rows = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    with open(str(path), "wb") as f:
        # PNG signature
        f.write(b"\x89PNG\r\n\x1a\n")
        # IHDR: width, height, bit depth=8, color type=2 (RGB), ...
        f.write(_make_chunk(b"IHDR",
                            struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        # IDAT
        f.write(_make_chunk(b"IDAT", zlib.compress(raw_rows, 9)))
        # IEND
        f.write(_make_chunk(b"IEND", b""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="temp/render_test",
        help="Directory to write PNG files into (default: temp/render_test)",
    )
    parser.add_argument(
        "--width", type=int, default=320, help="Render width in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=240, help="Render height in pixels"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "ros2_ws" / "src" / "vla_mujoco" / "assets" / "toy_arm.xml"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    if not model_path.exists():
        sys.exit(f"ERROR: model not found: {model_path}")

    print(f"model : {model_path}")
    print(f"output: {output_dir}")

    for backend in ("egl", "osmesa"):
        print(f"\nTrying MUJOCO_GL={backend} ...")
        _set_backend(backend)
        try:
            saved = render_arm(model_path, output_dir, width=args.width, height=args.height)
            print(f"\nRendering succeeded with backend={backend}")
            print("Saved files:")
            for p in saved:
                print(f"  {p}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  backend={backend} failed: {exc}")

    sys.exit("ERROR: All rendering backends failed.")


if __name__ == "__main__":
    main()

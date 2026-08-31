"""
backend/tests/generate_demo_pair.py
-----------------------------------
Generates lightweight synthetic before/after image pairs for local testing
and judge-ready demos without downloading external satellite datasets.

Usage:
  python generate_demo_pair.py
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw


def make_changed_pair(
    dest_dir: Path,
    size: Tuple[int, int] = (128, 128),
    change_rect: Tuple[int, int, int, int] = (30, 30, 90, 90),
) -> Tuple[Path, Path]:
    """
    Creates two RGB PNG files in dest_dir:
      - before.png: Uniform background (dark slate) + subtle grid/features
      - after.png: Same background + rectangular changed patch (bright contrast)

    Returns:
      (before_path, after_path)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    before_path = dest_dir / "before.png"
    after_path = dest_dir / "after.png"

    # Base image: slate background with grid lines
    before_img = Image.new("RGB", size, color=(45, 55, 72))
    draw_b = ImageDraw.Draw(before_img)
    # Draw reference features
    for y in range(0, size[1], 16):
        draw_b.line([(0, y), (size[0], y)], fill=(60, 70, 90), width=1)
    for x in range(0, size[0], 16):
        draw_b.line([(x, 0), (x, size[1])], fill=(60, 70, 90), width=1)

    before_img.save(before_path, format="PNG")

    # After image: identical base + high contrast patch
    after_img = before_img.copy()
    draw_a = ImageDraw.Draw(after_img)
    draw_a.rectangle(change_rect, fill=(220, 200, 180), outline=(255, 255, 255))

    after_img.save(after_path, format="PNG")

    return before_path, after_path


def make_changed_pair_bytes(
    size: Tuple[int, int] = (64, 64),
    change_rect: Tuple[int, int, int, int] = (16, 16, 48, 48),
) -> Tuple[bytes, bytes]:
    """Return in-memory PNG bytes for (before, after) with a clear changed patch."""
    before_img = Image.new("RGB", size, color=(40, 50, 65))
    draw_b = ImageDraw.Draw(before_img)
    draw_b.rectangle((10, 10, 25, 25), fill=(70, 80, 100))

    after_img = before_img.copy()
    draw_a = ImageDraw.Draw(after_img)
    draw_a.rectangle(change_rect, fill=(240, 220, 200))

    buf_b = io.BytesIO()
    before_img.save(buf_b, format="PNG")
    buf_b.seek(0)

    buf_a = io.BytesIO()
    after_img.save(buf_a, format="PNG")
    buf_a.seek(0)

    return buf_b.read(), buf_a.read()


if __name__ == "__main__":
    demo_dir = Path(__file__).resolve().parent / "demo_images"
    b_path, a_path = make_changed_pair(demo_dir)
    print(f"Generated demo pair:\n  Before: {b_path}\n  After:  {a_path}")

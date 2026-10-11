"""Overlay column/row labels on the 2x3 clay comparison grid.

Reproduces the labels on results/compare_3quant_clay.png as published:
columns bf16 / int8 / NVFP4 on top, STANDARD / OPTIMIZED row tags on the left.
Layout targets the 3840x2160 grid; positions scale with image size.

Usage:
  python scripts/label_usergrid.py <unlabeled-grid.png> <labeled-out.png>

Requires Pillow. Font: tries Arial (Windows), then DejaVu (Linux),
then Pillow's built-in bitmap font.
"""
from PIL import Image, ImageDraw, ImageFont
import os
import sys


def load_font(size):
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    img = Image.open(src).convert("RGB")
    W, H = img.size
    s = max(W / 3840.0, H / 2160.0)
    d = ImageDraw.Draw(img, "RGBA")
    font_big = load_font(int(92 * s))
    font_row = load_font(int(80 * s))

    for i, name in enumerate(["bf16", "int8", "NVFP4"]):
        x = int(W * (1 + 2 * i) / 6)
        bb = d.textbbox((0, 0), name, font=font_big)
        tw = bb[2] - bb[0]
        pad = int(28 * s)
        d.rectangle([x - tw // 2 - pad, 30, x + tw // 2 + pad,
                     30 + (bb[3] - bb[1]) + int(40 * s)], fill=(0, 0, 0, 130))
        d.text((x - tw // 2, 44), name, font=font_big, fill=(255, 255, 255))

    for name, fy in [("STANDARD", 0.115), ("OPTIMIZED", 0.50)]:
        y = int(H * fy)
        bb = d.textbbox((0, 0), name, font=font_row)
        d.rectangle([48, y, 48 + (bb[2] - bb[0]) + int(56 * s),
                     y + (bb[3] - bb[1]) + int(40 * s)], fill=(0, 0, 0, 130))
        d.text((72, y + 12), name, font=font_row, fill=(255, 215, 120))

    img.save(dst)
    print("saved", dst, img.size)


if __name__ == "__main__":
    main()

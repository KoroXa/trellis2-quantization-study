"""Overlay grid labels on comparison renders."""
from PIL import Image, ImageDraw, ImageFont
import sys

src, dst = sys.argv[1], sys.argv[2]
img = Image.open(src).convert("RGB")
W, H = img.size
d = ImageDraw.Draw(img, "RGBA")
font_big = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 46)
font_row = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 40)

cols = ["bf16", "int8", "NVFP4"]
for i, name in enumerate(cols):
    x = int(W * (1 + 2 * i) / 6)
    t = name
    bb = d.textbbox((0, 0), t, font=font_big)
    tw = bb[2] - bb[0]
    d.rectangle([x - tw // 2 - 14, 18, x + tw // 2 + 14, 18 + (bb[3] - bb[1]) + 20], fill=(0, 0, 0, 130))
    d.text((x - tw // 2, 24), t, font=font_big, fill=(255, 255, 255))

rows = [("STANDARD", 0.24), ("OPTIMIZED", 0.77)]
for name, fy in rows:
    y = int(H * fy)
    bb = d.textbbox((0, 0), name, font=font_row)
    d.rectangle([24, y, 24 + (bb[2] - bb[0]) + 28, y + (bb[3] - bb[1]) + 20], fill=(0, 0, 0, 130))
    d.text((36, y + 6), name, font=font_row, fill=(255, 215, 120))

img.save(dst)
print("saved", dst, img.size)

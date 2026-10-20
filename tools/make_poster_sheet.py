"""把三组对照结果拼成一张图，便于肉眼比较变形与否。"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "poster_ratio_exp")

PANELS = [
    ("A_旧拉伸_512", "A 旧行为 512×512（硬拉伸 → 变形）"),
    ("B_新填充_sd15_3x4", "B 新 384×512（3:4，等比+留白）"),
    ("C_新填充_sdxl_3x4", "C 新 768×1024（3:4，SDXL）"),
]

HEIGHT = 620
PAD = 16
CAPTION = 34

tiles = []
for folder, caption in PANELS:
    hits = glob.glob(os.path.join(OUT, folder, "*.png"))
    if not hits:
        print(f"跳过（无结果）：{folder}")
        continue
    im = Image.open(sorted(hits)[0]).convert("RGB")
    scale = HEIGHT / im.height
    im = im.resize((max(1, round(im.width * scale)), HEIGHT), Image.LANCZOS)
    tiles.append((im, caption))

if not tiles:
    raise SystemExit("没有可用的结果图")

total_w = sum(t.width for t, _ in tiles) + PAD * (len(tiles) + 1)
sheet = Image.new("RGB", (total_w, HEIGHT + CAPTION + PAD * 2), (245, 243, 238))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
except Exception:  # noqa: BLE001
    font = ImageFont.load_default()

x = PAD
for im, caption in tiles:
    sheet.paste(im, (x, PAD))
    draw.text((x, PAD + HEIGHT + 8), caption, fill=(40, 40, 40), font=font)
    x += im.width + PAD

path = os.path.join(OUT, "contact_sheet.png")
sheet.save(path)
print("已生成:", os.path.relpath(path, PROJECT), f"{sheet.width}x{sheet.height}")

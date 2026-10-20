"""把若干张图拼成带标题的对照图：python tools/sheet.py out.png "标题1=路径1" "标题2=路径2" """

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1]
items = []
for arg in sys.argv[2:]:
    caption, _, path = arg.partition("=")
    if not os.path.exists(path):
        print("跳过（不存在）:", path)
        continue
    items.append((caption, Image.open(path).convert("RGB")))

if not items:
    raise SystemExit("没有可用图片")

H = 900
PAD = 14
CAP = 34
tiles = []
for cap, im in items:
    s = H / im.height
    tiles.append((cap, im.resize((max(1, round(im.width * s)), H), Image.LANCZOS)))

W = sum(t.width for _, t in tiles) + PAD * (len(tiles) + 1)
sheet = Image.new("RGB", (W, H + CAP + PAD * 2), (246, 244, 239))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
except Exception:  # noqa: BLE001
    font = ImageFont.load_default()

x = PAD
for cap, t in tiles:
    sheet.paste(t, (x, PAD))
    draw.text((x, PAD + H + 6), cap, fill=(35, 35, 35), font=font)
    x += t.width + PAD
sheet.save(out)
print("已生成", out, sheet.size)

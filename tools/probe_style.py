"""诊断：同一句英文提示词在不同模型/挡位下的风格响应。

提示词：A panda is running on the grassland on a sunny day, American comic style.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "style_probe")
PROMPT = "A panda is running on the grassland on a sunny day, American comic style."

CASES = [
    ("sd-turbo", "bf16", "turbo", 512),      # 默认挡位
    ("sd-turbo", "bf16", "balanced", 512),   # turbo 模型 + 有 CFG 的挡位
    ("sd15", "bf16", "balanced", 512),
    ("sdxl", "fp16", "balanced", 1024),
]

os.makedirs(OUT, exist_ok=True)
engine = Engine(project_root=PROJECT, verbose=False)
rows = []
current = [None]

for model_key, prec_key, gear, size in CASES:
    prof = resolve_model(model_key)
    prec = resolve_precision(prec_key)
    q = resolve_quality(gear, prof)
    if current[0] and current[0] != prof.key:
        engine.unload()
    current[0] = prof.key

    req = GenRequest(
        mode="txt2img",
        prompt=PROMPT,
        model=prof.key,
        precision=prec.key,
        gear=q.key,
        width=size,
        height=size,
        seed=20261020,
        out_dir=os.path.join(OUT, f"{model_key}_{gear}"),
        size_source="诊断",
    )
    res = engine.run(req, prof, prec, q)
    if not res.ok:
        print(f"  {model_key}/{gear}: 失败 {res.error}")
        continue
    # 第一次是预热，再跑一次取热态
    res = engine.run(req, prof, prec, q)
    m = res.meta or {}
    print(
        f"  {model_key:<9} {gear:<9} {size}px  steps={m.get('steps')}  cfg={m.get('guidance')}  "
        f"{res.seconds:5.2f}s",
        flush=True,
    )
    rows.append((f"{model_key} / {gear} / cfg={m.get('guidance')}", res.images[0]))

# 拼图
h = 420
tiles = []
for cap, p in rows:
    im = Image.open(p).convert("RGB")
    k = h / im.height
    tiles.append((cap, im.resize((round(im.width * k), h), Image.LANCZOS)))

pad, cap_h = 12, 46
sheet = Image.new("RGB", (sum(t.width for _, t in tiles) + pad * (len(tiles) + 1), h + cap_h + pad), (250, 249, 246))
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
draw.text((pad, 8), PROMPT, fill=(30, 30, 30), font=font)
x = pad
for cap, t in tiles:
    sheet.paste(t, (x, cap_h))
    draw.text((x, cap_h + h + 6), cap, fill=(40, 40, 40), font=font)
    x += t.width + pad
path = os.path.join(OUT, "probe.png")
sheet.save(path)
print("已生成:", os.path.relpath(path, PROJECT))

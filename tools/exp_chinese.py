"""对照实验：SDXL 里如何让 Chinese 不再默认成日式动漫。

A 基线：只说 Chinese + 传统服装
B 加文化锚点：汉服 / 唐制 / 中国建筑 / 国画背景
C 基线 + 负向压制动漫
D 文化锚点 + 负向压制
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "chinese_style")
W, H, SEED = 768, 1024, 20261022

NEG = "anime, manga, japanese, ukiyo-e, cel shading, cartoon, illustration, flat colors"

CASES = [
    ("A 基线", "a Chinese woman in traditional clothing, portrait, soft daylight", ""),
    ("B 文化锚点", "a Chinese woman in hanfu, Tang dynasty hairstyle, Chinese courtyard background, "
                   "traditional Chinese painting aesthetic, soft daylight", ""),
    ("C 基线+负向", "a Chinese woman in traditional clothing, portrait, soft daylight", NEG),
    ("D 锚点+负向", "a Chinese woman in hanfu, Tang dynasty hairstyle, Chinese courtyard background, "
                    "traditional Chinese painting aesthetic, soft daylight", NEG),
]

os.makedirs(OUT, exist_ok=True)
engine = Engine(project_root=PROJECT, verbose=False)
prof = resolve_model("sdxl")
prec = resolve_precision("fp16")
q = resolve_quality("balanced", prof)

tiles = []
for label, prompt, neg in CASES:
    tag = label.split()[0]
    req = GenRequest(
        mode="txt2img",
        prompt=prompt,
        negative_prompt=neg,
        model=prof.key,
        precision=prec.key,
        gear=q.key,
        width=W,
        height=H,
        seed=SEED,
        out_dir=os.path.join(OUT, tag),
        style_expand="off",  # 关掉自动展开，避免干扰对照
    )
    res = engine.run(req, prof, prec, q)
    if not res.ok:
        print(f"  {label}: 失败 {res.error}")
        continue
    print(f"  {label:<10} {res.seconds:5.2f}s -> {os.path.relpath(res.images[0], PROJECT)}", flush=True)
    tiles.append((label, res.images[0]))

h = 460
imgs = []
for label, p in tiles:
    im = Image.open(p).convert("RGB")
    k = h / im.height
    imgs.append((label, im.resize((round(im.width * k), h), Image.LANCZOS)))

pad, cap = 12, 30
sheet = Image.new("RGB", (sum(t.width for _, t in imgs) + pad * (len(imgs) + 1), h + cap + pad), (248, 247, 244))
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
x = pad
for label, im in imgs:
    sheet.paste(im, (x, pad))
    draw.text((x, pad + h + 5), label, fill=(35, 35, 35), font=font)
    x += im.width + pad
path = os.path.join(OUT, "compare.png")
sheet.save(path)
print("已生成:", os.path.relpath(path, PROJECT))

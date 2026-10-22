"""第二组：泛化人像（不带服装锚点），看是否滑向日式/动漫。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "chinese_style2")
W, H, SEED = 768, 1024, 20261022

NEG = "anime, manga, japanese, ukiyo-e, cel shading, cartoon, illustration"

CASES = [
    ("A 泛化", "a beautiful Chinese girl, portrait", ""),
    ("B 泛化+负向", "a beautiful Chinese girl, portrait", NEG),
    ("C 加写实", "a beautiful Chinese girl, portrait, realistic photograph, 85mm lens, natural skin texture", ""),
    ("D 加写实+负向", "a beautiful Chinese girl, portrait, realistic photograph, 85mm lens, natural skin texture", NEG),
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
        style_expand="off",
    )
    res = engine.run(req, prof, prec, q)
    if not res.ok:
        print(f"  {label}: 失败 {res.error}")
        continue
    print(f"  {label:<14} {res.seconds:5.2f}s", flush=True)
    tiles.append((label, res.images[0]))

h = 460
imgs = []
for label, p in tiles:
    im = Image.open(p).convert("RGB")
    k = h / im.height
    imgs.append((label, im.resize((round(im.width * k), h), Image.LANCZOS)))

pad, cap = 12, 30
sheet = Image.new("RGB", (sum(t.width for _, t in imgs) + pad * (len(imgs) + 1), h + cap + pad + 10), (248, 247, 244))
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

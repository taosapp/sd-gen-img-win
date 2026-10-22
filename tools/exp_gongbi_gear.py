"""对照：同一格在 lightning(4步,CFG=0) 与 balanced(25步,CFG=7) 下的工笔还原度。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "gongbi_gear")
W, H = 768, 1344

STYLE = (
    "gongbi painting, traditional Chinese fine-line brushwork, iron-wire outlines, "
    "flat mineral pigments of azurite blue, malachite green and cinnabar red, "
    "silk scroll texture, decorative patterns"
)
NEG = (
    "anime, manga, cartoon, photo, photorealistic, 3d render, western oil painting, "
    "perspective shading, japanese kimono, ukiyo-e, modern clothing, text, watermark, "
    "adult woman, woman, female, textile pattern, extra limbs, deformed hands"
)
SHOT = (
    "a four-year-old boy in pale blue hanfu holding one small pear in both hands, "
    "an elder brother beside him holding a much larger pear, the size difference clearly visible, "
    "Eastern Han dynasty, crossed-collar hanfu with wide sleeves, topknot hair"
)

os.makedirs(OUT, exist_ok=True)
engine = Engine(project_root=PROJECT, verbose=False)
prof = resolve_model("sdxl")
prec = resolve_precision("fp16")

tiles = []
for gear in ("lightning", "balanced"):
    q = resolve_quality(gear, prof)
    req = GenRequest(
        mode="txt2img",
        prompt=f"{SHOT}, {STYLE}",
        negative_prompt=NEG,
        model=prof.key,
        precision=prec.key,
        gear=q.key,
        width=W,
        height=H,
        seed=20261022,
        out_dir=os.path.join(OUT, gear),
        style_expand="off",
    )
    res = engine.run(req, prof, prec, q)
    if not res.ok:
        print(f"  {gear}: 失败 {res.error}")
        continue
    print(f"  {gear:<10} {res.seconds:5.2f}s  cfg={res.meta.get('guidance')}", flush=True)
    tiles.append((f"{gear} / cfg={res.meta.get('guidance')} / {res.seconds:.1f}s", res.images[0]))

h = 480
imgs = []
for label, p in tiles:
    im = Image.open(p).convert("RGB")
    k = h / im.height
    imgs.append((label, im.resize((round(im.width * k), h), Image.LANCZOS)))

pad, cap = 12, 30
sheet = Image.new("RGB", (sum(t.width for _, t in imgs) + pad * (len(imgs) + 1), h + cap + pad), (250, 248, 243))
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

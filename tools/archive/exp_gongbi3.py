"""孔融让梨 · 工笔分镜 v3（风格前置 + 每条 ≤38 词元，确保不被截断）。"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = _ROOT
OUT = os.path.join(PROJECT, "results", "gongbi_v3")
W, H = 768, 1344
GEAR = os.environ.get("GONGBI_GEAR", "balanced")

# 风格块放在最前：CLIP 对前部权重更高，且短（10 词元），不会被截断
STYLE = "gongbi painting, fine ink outlines, mineral pigments, silk scroll"

NEG = (
    "anime, manga, cartoon, photo, photorealistic, 3d render, oil painting, "
    "perspective shading, japanese kimono, ukiyo-e, modern clothing, text, watermark, "
    "adult woman, female portrait, textile pattern, extra limbs, deformed hands"
)

SHOTS = [
    ("01 梨树", "a pear tree full of yellow pears over a Han dynasty courtyard, two servants beneath it"),
    ("02 摘梨", "a servant lifting a basket of pears, a father in dark robe and three brothers watching"),
    ("03 盛梨", "a small boy in blue hanfu looks at a lacquer tray of pears of different sizes"),
    ("04 环坐", "a father and three brothers sit on mats in age order, the youngest boy sits last"),
    ("05 父命", "the father gestures to the pear tray, asking his youngest son to choose first"),
    ("06 取小", "a four-year-old boy in blue hanfu holds the smallest pear, his brother holds a big one"),
    ("07 父问", "the father leans down to question his small son, brothers listening quietly"),
    ("08 作答", "the small boy bows with folded hands, answering his father, brothers watching"),
    ("09 兄惭", "the three elder brothers look ashamed at the little boy, the father nods approval"),
    ("10 分梨", "the family sits together sharing pears, empty silk space above for a colophon"),
]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    engine = Engine(project_root=PROJECT, verbose=False)
    prof = resolve_model("sdxl")
    prec = resolve_precision("fp16")
    q = resolve_quality(GEAR, prof)

    paths = []
    for idx, (label, shot) in enumerate(SHOTS, start=1):
        req = GenRequest(
            mode="txt2img",
            prompt=f"{STYLE}, {shot}",
            negative_prompt=NEG,
            model=prof.key,
            precision=prec.key,
            gear=q.key,
            width=W,
            height=H,
            seed=20261022,
            out_dir=os.path.join(OUT, "shots"),
            style_expand="off",
        )
        res = engine.run(req, prof, prec, q)
        if not res.ok:
            print(f"  {label}: 失败 {res.error}")
            continue
        dst = os.path.join(OUT, "shots", f"shot{idx:02d}.png")
        os.replace(res.images[0], dst)
        paths.append((label, dst))
        print(f"  {label:<8} {res.seconds:5.2f}s", flush=True)

    if not paths:
        return 1

    h, cols = 420, 5
    tiles = []
    for label, p in paths:
        im = Image.open(p).convert("RGB")
        k = h / im.height
        tiles.append((label, im.resize((round(im.width * k), h), Image.LANCZOS)))

    pad, cap = 10, 26
    rows = (len(tiles) + cols - 1) // cols
    tw = max(t.width for _, t in tiles)
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (h + cap + pad) + pad), (250, 248, 243))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    for i, (label, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (h + cap + pad)
        sheet.paste(im, (x, y))
        draw.text((x, y + h + 4), label, fill=(40, 40, 40), font=font)
    path = os.path.join(OUT, "storyboard.png")
    sheet.save(path)
    print("已生成:", os.path.relpath(path, PROJECT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

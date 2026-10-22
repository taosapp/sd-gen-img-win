"""孔融让梨 · 工笔分镜 v2：主语前置 + 强制人物在场。

v1 的教训：
- 人物块放句尾 → CLIP 权重大概率落到"梨/盘/庭院"上，第 3/4/10 格直接变成纹样或空景；
- "Kong Rong" 没有紧跟 "four-year-old boy" → 第 6/8 格画成成年女子。

v2 规则：
1. 每格**第一句就是人**（谁 + 几岁 + 在做什么），风格块后置；
2. 儿童场景重复写 boy / child，并在负向里压 adult / woman；
3. 纯物件镜头（梨盘）也必须有手或人在画面里，否则模型会退化成花纹。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "gongbi_v2")
W, H = 768, 1344
GEAR = os.environ.get("GONGBI_GEAR", "balanced")

STYLE = (
    "gongbi painting, traditional Chinese fine-line brushwork, iron-wire outlines, "
    "flat mineral pigments of azurite blue, malachite green and cinnabar red, "
    "silk scroll texture, decorative patterns"
)
NEG = (
    "anime, manga, cartoon, photo, photorealistic, 3d render, western oil painting, "
    "perspective shading, japanese kimono, ukiyo-e, modern clothing, text, watermark, "
    "adult woman, woman, female, lady, portrait of a lady, textile pattern, "
    "empty landscape, extra limbs, deformed hands"
)
ERA = "Eastern Han dynasty, crossed-collar hanfu with wide sleeves, topknot hair"

# 每个镜头：主语在前，动作在中间，时代与画风在最后
SHOTS = [
    ("01 梨树", "a pear tree heavy with ripe yellow pears in an autumn courtyard, two servants in short robes "
                f"standing under it, {ERA}, wooden pavilion"),
    ("02 摘梨", "a servant lifting a basket of pears to a middle-aged father in a dark brown robe and black cap, "
                f"three elder brothers watching nearby, {ERA}"),
    ("03 盛梨", "a small boy in pale blue hanfu looking down at a round lacquered tray of pears of clearly "
                f"different sizes on a low table, his father's hand resting beside the tray, {ERA}"),
    ("04 环坐", "a father and three elder brothers seated in order of age on woven mats, the youngest four-year-old "
                f"boy in pale blue sitting last and smallest, all facing a pear tray, {ERA}"),
    ("05 父命", "a middle-aged father in dark robe gesturing toward a tray of pears, inviting his youngest son to "
                f"choose first, three elder brothers seated beside him, {ERA}"),
    ("06 取小", "a four-year-old boy in pale blue hanfu reaching with both small hands for the smallest pear on a "
                f"lacquered tray, his father and brothers watching, {ERA}"),
    ("07 父问", "a father leaning down to ask his four-year-old son a question, the boy holding a small pear, "
                f"elder brothers listening quietly, {ERA}"),
    ("08 作答", "a four-year-old boy in pale blue hanfu bowing with hands folded in respect, answering his father, "
                f"three elder brothers and the father watching him, {ERA}"),
    ("09 兄惭", "three elder brothers looking at a small boy with surprised and ashamed faces, the father nodding "
                f"approval, the boy holding the smallest pear, {ERA}"),
    ("10 分梨", "a father, three elder brothers and a four-year-old boy seated together around a low table sharing "
                f"pears, harmonious family scene, empty silk space at the top for a colophon, {ERA}"),
]


def build(shot: str) -> str:
    return f"{shot}, {STYLE}"


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
            prompt=build(shot),
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

"""孔融让梨 · 工笔分镜 10 张（SDXL / 9:16 / balanced）。

设计原则：
- 画风块（STYLE）与负向（NEG）每镜完全一致 —— 风格漂移主要来自这里被改动；
- 人物块（KONG / FATHER / BROTHERS）逐字复用 —— 换词就等于换人；
- 每镜只写"这一格发生了什么"，不重复描述环境细节（留给 SETTING 块）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "gongbi_kongrong")
W, H = 768, 1344
GEAR = os.environ.get("GONGBI_GEAR", "balanced")

STYLE = (
    "gongbi painting, traditional Chinese fine-line brushwork, iron-wire outlines, "
    "flat mineral pigments of azurite blue, malachite green and cinnabar red, "
    "silk scroll texture, no cast shadows, intricate decorative patterns"
)
NEG = (
    "anime, manga, cartoon, photo, photorealistic, 3d render, western oil painting, "
    "perspective shading, japanese kimono, ukiyo-e, modern clothing, text, watermark, "
    "extra limbs, deformed hands"
)
ERA = (
    "Eastern Han dynasty China, crossed-collar hanfu with wide sleeves, topknot hair, "
    "wooden courtyard with a pear tree, low lacquered table, woven mats, paper screen"
)
KONG = (
    "Kong Rong, a small four-year-old Chinese boy, topknot hair, pale blue crossed-collar robe"
)
FATHER = "his father, a middle-aged Chinese scholar in dark brown robe with a black cap"
BROTHERS = "his three elder brothers in grey and ochre robes"

SHOTS = [
    ("01 梨树", f"a pear tree heavy with ripe yellow pears in an autumn courtyard, {ERA}"),
    ("02 摘梨", f"a servant in short robe lifting a basket of pears, {FATHER} standing under the eaves, {ERA}"),
    ("03 盛梨", f"a round lacquered tray of pears of clearly different sizes on a low table, {ERA}"),
    ("04 环坐", f"{FATHER} and {BROTHERS} seated in order of age on woven mats, {KONG} seated last and smallest, {ERA}"),
    ("05 父命", f"{FATHER} gesturing toward the tray of pears, inviting the youngest to choose first, {ERA}"),
    ("06 取小", f"{KONG} reaching with both hands for the smallest pear on the tray, {ERA}"),
    ("07 父问", f"{FATHER} leaning down to ask the boy a question, {BROTHERS} watching quietly, {ERA}"),
    ("08 作答", f"{KONG} bowing with hands folded in respect, answering his father calmly, {ERA}"),
    ("09 兄惭", f"{BROTHERS} looking at the small boy with surprised and ashamed expressions, {FATHER} nodding, {ERA}"),
    ("10 分梨", f"the family seated together sharing the pears, warm and harmonious, empty space at the top for a colophon, {ERA}"),
]


def build(shot_text: str) -> str:
    return f"{STYLE}, {shot_text}, {KONG}"


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
        # 按镜头号重命名
        dst = os.path.join(OUT, "shots", f"shot{idx:02d}.png")
        os.replace(res.images[0], dst)
        paths.append((label, dst))
        print(f"  {label:<8} {res.seconds:5.2f}s", flush=True)

    if not paths:
        return 1

    # 5 列 × 2 行
    h = 420
    cols = 5
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

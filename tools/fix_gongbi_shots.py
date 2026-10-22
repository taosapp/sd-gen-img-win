"""只重画第 01、06 格，并重建整组分镜图。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "gongbi_v2")
SHOTS_DIR = os.path.join(OUT, "shots")
W, H = 768, 1344

STYLE = (
    "gongbi painting, traditional Chinese fine-line brushwork, iron-wire outlines, "
    "flat mineral pigments of azurite blue, malachite green and cinnabar red, "
    "silk scroll texture, decorative patterns"
)
NEG = (
    "anime, manga, cartoon, photo, photorealistic, 3d render, western oil painting, "
    "perspective shading, japanese kimono, ukiyo-e, modern clothing, text, watermark, "
    "adult woman, woman, female, lady, textile pattern, extra limbs, deformed hands"
)
ERA = "Eastern Han dynasty, crossed-collar hanfu with wide sleeves, topknot hair"

RETRY = {
    1: (
        "a pear tree heavy with ripe yellow pears filling the upper half of the picture, "
        f"two servants in short robes standing beneath it in an autumn courtyard, {ERA}"
    ),
    6: (
        "a four-year-old boy in pale blue hanfu holding one small pear in both hands, "
        "an elder brother beside him holding a much larger pear, the size difference clearly visible, "
        f"a lacquered tray on the table, {ERA}"
    ),
}


def main() -> int:
    engine = Engine(project_root=PROJECT, verbose=False)
    prof = resolve_model("sdxl")
    prec = resolve_precision("fp16")
    q = resolve_quality("balanced", prof)

    for shot, text in RETRY.items():
        req = GenRequest(
            mode="txt2img",
            prompt=f"{text}, {STYLE}",
            negative_prompt=NEG,
            model=prof.key,
            precision=prec.key,
            gear=q.key,
            width=W,
            height=H,
            seed=20261022,
            out_dir=SHOTS_DIR,
            style_expand="off",
        )
        res = engine.run(req, prof, prec, q)
        if not res.ok:
            print(f"  第 {shot:02d} 格失败: {res.error}")
            continue
        dst = os.path.join(SHOTS_DIR, f"shot{shot:02d}.png")
        os.replace(res.images[0], dst)
        print(f"  重画 shot{shot:02d}  {res.seconds:5.2f}s", flush=True)

    # 重建分镜
    labels = [
        "01 梨树", "02 摘梨", "03 盛梨", "04 环坐", "05 父命",
        "06 取小", "07 父问", "08 作答", "09 兄惭", "10 分梨",
    ]
    h, cols = 420, 5
    tiles = []
    for i, label in enumerate(labels, start=1):
        p = os.path.join(SHOTS_DIR, f"shot{i:02d}.png")
        if not os.path.exists(p):
            continue
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
    print("已重建:", os.path.relpath(path, PROJECT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

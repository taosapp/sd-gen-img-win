"""对照：同一提示词/种子，只改分辨率与精度，看 SDXL 在哪一档"回到照片"。

512 / 768 / 832 / 1024，各跑 fp16；再补一个 1024 + bf16 排除精度干扰。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = _ROOT
OUT = os.path.join(PROJECT, "results", "sdxl_resolution")
PROMPT = "a panda running on a green grassland, sunny day"
SEED = 20261022

CASES = [
    (512, "fp16"),
    (768, "fp16"),
    (832, "fp16"),
    (1024, "fp16"),
    (1024, "bf16"),
]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    engine = Engine(project_root=PROJECT, verbose=False)
    prof = resolve_model("sdxl")

    tiles = []
    for size, prec_key in CASES:
        prec = resolve_precision(prec_key)
        q = resolve_quality("balanced", prof)
        req = GenRequest(
            mode="txt2img",
            prompt=PROMPT,
            model=prof.key,
            precision=prec.key,
            gear=q.key,
            width=size,
            height=size,
            seed=SEED,
            out_dir=os.path.join(OUT, f"{size}_{prec_key}"),
            style_expand="off",
        )
        res = engine.run(req, prof, prec, q)
        if not res.ok:
            print(f"  {size}/{prec_key}: 失败 {res.error}")
            continue
        print(f"  {size}px {prec_key:<5} {res.seconds:6.2f}s", flush=True)
        tiles.append((f"{size}px / {prec_key}", res.images[0]))

    h = 420
    imgs = []
    for label, p in tiles:
        im = Image.open(p).convert("RGB")
        k = h / im.height
        imgs.append((label, im.resize((round(im.width * k), h), Image.LANCZOS)))

    pad, cap = 10, 28
    sheet = Image.new("RGB", (sum(t.width for _, t in imgs) + pad * (len(imgs) + 1), h + cap + pad), (250, 248, 243))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    x = pad
    for label, im in imgs:
        sheet.paste(im, (x, pad))
        draw.text((x, pad + h + 4), label, fill=(35, 35, 35), font=font)
        x += im.width + pad
    path = os.path.join(OUT, "compare.png")
    sheet.save(path)
    print("已生成:", os.path.relpath(path, PROJECT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""为每个模型生成"挡位对照条"：一行挡位（或精度），标注耗时。

SDXL 用原生 1024，SD1.5 / SD-Turbo 用 512 —— 低于原生分辨率会让 SDXL 掉进插画先验（见 README §7.5）。

输出：results/gear_compare/<model>.png
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import (  # noqa: E402
    PRECISION_GEARS,
    quality_gears_for,
    resolve_model,
    resolve_precision,
    resolve_quality,
)

PROJECT = _ROOT
OUT = os.path.join(PROJECT, "results", "gear_compare")
PROMPT = "a panda running on a green grassland, sunny day"
SEED = 20261022

# 每个模型：分辨率 + 要对比的维度
PLAN = {
    "sdxl": {"size": 1024, "precisions": ["fp16"], "axes": "gears"},
    "sd15": {"size": 512, "precisions": ["fp16"], "axes": "gears"},
    "sd-turbo": {"size": 512, "precisions": ["fp32", "tf32", "bf16", "fp16"], "axes": "precisions"},
}


def compose(title: str, tiles: list[tuple[str, str, float]], path: str) -> None:
    h = 420
    pad, cap, head = 12, 52, 54
    imgs = []
    for label, p, secs in tiles:
        im = Image.open(p).convert("RGB")
        k = h / im.height
        imgs.append((label, secs, im.resize((round(im.width * k), h), Image.LANCZOS)))

    width = sum(t.width for _, _, t in imgs) + pad * (len(imgs) + 1)
    sheet = Image.new("RGB", (width, head + h + cap), (250, 248, 243))
    draw = ImageDraw.Draw(sheet)
    f_title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
    f_cap = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)

    draw.text((pad, 12), title, fill=(25, 25, 25), font=f_title)
    x = pad
    for label, secs, im in imgs:
        sheet.paste(im, (x, head))
        draw.text((x, head + h + 6), label, fill=(35, 35, 35), font=f_cap)
        draw.text((x, head + h + 28), f"{secs:.2f} s", fill=(110, 110, 110), font=f_cap)
        x += im.width + pad
    sheet.save(path)
    print("已生成:", os.path.relpath(path, PROJECT), sheet.size)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    engine = Engine(project_root=PROJECT, verbose=False)

    titles = {
        "sdxl": f"SDXL · 1024×1024（原生分辨率）· prompt: {PROMPT}",
        "sd15": f"SD1.5 · 512×512（原生分辨率）· prompt: {PROMPT}",
        "sd-turbo": f"SD-Turbo · 512×512 · turbo 档下各精度对比 · prompt: {PROMPT}",
    }

    for model_key, cfg in PLAN.items():
        prof = resolve_model(model_key)
        gears = quality_gears_for(prof)
        tiles: list[tuple[str, str, float]] = []

        if cfg["axes"] == "precisions":
            gear = gears[0]
            q = resolve_quality(gear, prof)
            for prec_key in cfg["precisions"]:
                prec = resolve_precision(prec_key)
                req = GenRequest(
                    mode="txt2img",
                    prompt=PROMPT,
                    model=prof.key,
                    precision=prec.key,
                    gear=q.key,
                    width=cfg["size"],
                    height=cfg["size"],
                    seed=SEED,
                    out_dir=os.path.join(OUT, model_key),
                    style_expand="off",
                )
                res = engine.run(req, prof, prec, q)
                if not res.ok:
                    print(f"  {model_key}/{prec.key}: 失败 {res.error}")
                    continue
                print(f"  {model_key:<9} {prec.key:<5} {gear:<10} {res.seconds:6.2f}s", flush=True)
                tiles.append((f"{prec_key} · {gear}", res.images[0], float(res.seconds or 0)))
        else:
            prec = resolve_precision(cfg["precisions"][0])
            for gear_key in gears:
                q = resolve_quality(gear_key, prof)
                req = GenRequest(
                    mode="txt2img",
                    prompt=PROMPT,
                    model=prof.key,
                    precision=prec.key,
                    gear=q.key,
                    width=cfg["size"],
                    height=cfg["size"],
                    seed=SEED,
                    out_dir=os.path.join(OUT, model_key),
                    style_expand="off",
                )
                res = engine.run(req, prof, prec, q)
                if not res.ok:
                    print(f"  {model_key}/{gear_key}: 失败 {res.error}")
                    continue
                print(f"  {model_key:<9} {gear_key:<11} {res.seconds:6.2f}s", flush=True)
                tiles.append(
                    (f"{gear_key} · {q.steps}步/CFG{q.guidance}", res.images[0], float(res.seconds or 0))
                )

        if tiles:
            compose(titles[model_key], tiles, os.path.join(OUT, f"{model_key}.png"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

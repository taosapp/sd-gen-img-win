"""强度扫描：找出"既像参考图、又有水彩感"的平衡点。

指标：生成图与参考图（同一裁切）的平均绝对差（MAE，越小越像）
和相关系数（越高越像）。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

from PIL import Image, ImageDraw, ImageFont, ImageStat  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.poster import WATERCOLOR_NEGATIVE, cover_crop  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = _ROOT
PHOTO = r"C:\Users\PPXW\Desktop\3.jpg"
OUT = os.path.join(PROJECT, "results", "strength_sweep")
W, H2 = 768, 512

STYLE = (
    "watercolor painting of this photograph, same composition and subject, "
    "soft translucent washes, muted natural colors, feathered edges, hand painted on paper"
)

STRENGTHS = [0.35, 0.45, 0.55, 0.65, 0.72]


def mae(a: Image.Image, b: Image.Image) -> float:
    if a.size != b.size:
        a = a.resize(b.size, Image.LANCZOS)
    sa, sb = ImageStat.Stat(a.convert("L")), ImageStat.Stat(b.convert("L"))
    # 用逐像素差更准，但不引入 numpy 依赖时用统计量近似：先算直方图差异
    ha = a.convert("L").histogram()
    hb = b.convert("L").histogram()
    total = sum(ha) or 1
    return sum(abs(x - y) for x, y in zip(ha, hb)) / (2 * total)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    src = cover_crop(Image.open(PHOTO).convert("RGB"), W, H2)
    src_path = os.path.join(OUT, "_src.png")
    src.save(src_path)

    model = resolve_model("sdxl")
    prec = resolve_precision("fp16")
    q = resolve_quality("balanced", model)
    engine = Engine(project_root=PROJECT, verbose=False)

    rows = []
    for s in STRENGTHS:
        req = GenRequest(
            mode="img2img",
            prompt=STYLE,
            negative_prompt=WATERCOLOR_NEGATIVE,
            model=model.key,
            precision=prec.key,
            gear=q.key,
            width=W,
            height=H2,
            seed=2026,
            strength=s,
            init_image=src_path,
            out_dir=os.path.join(OUT, f"s{int(s * 100)}"),
            size_source="扫描",
        )
        res = engine.run(req, model, prec, q)
        if not res.ok:
            print(f"  强度 {s}: 失败 {res.error}")
            continue
        img = Image.open(res.images[0]).convert("RGB")
        d = mae(src, img)
        print(f"  强度 {s:<5} 推理 {res.seconds:5.2f}s  与参考图差异 {d:.3f}（越小越像）", flush=True)
        rows.append((s, img, d))

    if not rows:
        return 1

    # 拼条带
    cap = 34
    pad = 10
    scale_h = 300
    tiles = []
    for s, img, d in rows:
        k = scale_h / img.height
        tiles.append((f"强度 {s:.2f} · 差异 {d:.3f}", img.resize((round(img.width * k), scale_h), Image.LANCZOS)))

    width = sum(t.width for _, t in tiles) + pad * (len(tiles) + 1)
    sheet = Image.new("RGB", (width, scale_h + cap + pad * 2), (245, 243, 238))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    x = pad
    for text, t in tiles:
        sheet.paste(t, (x, pad))
        draw.text((x, pad + scale_h + 6), text, fill=(40, 40, 40), font=font)
        x += t.width + pad
    path = os.path.join(OUT, "sweep.png")
    sheet.save(path)
    print("已生成:", os.path.relpath(path, PROJECT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

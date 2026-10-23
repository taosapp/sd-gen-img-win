"""把参考矩阵拼成对照大图。

行 = 模型 × 精度，列 = 画质挡位；缺图为浅灰占位。
输出：
  results/ref_matrix/matrix_A.png
  results/ref_matrix/matrix_B.png
  results/ref_matrix/matrix_combined.png   （A 在上、B 在下）
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

PROJECT = _ROOT
ROOT = os.path.join(PROJECT, "results", "ref_matrix")

MODELS = [("sdxl", "SDXL"), ("sd15", "SD1.5"), ("sd-turbo", "SD-Turbo")]
PRECISIONS = ["fp32", "tf32", "bf16", "fp16"]
GEARS = ["turbo", "lightning", "lightning8", "draft", "balanced", "quality", "max"]

CELL = 168
GAP = 6
LEFT = 150   # 行标签宽
TOP = 92     # 标题 + 列头
CAPTION = 26  # 每组下方留白

TITLES = {
    "A": "Prompt A: a panda running on a green grassland, sunny day",
    "B": "Prompt B: same + comic book illustration, bold black ink outlines, flat colors, halftone dots",
}


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def build(tag: str) -> Image.Image:
    rows = [(m, ml, p) for m, ml in MODELS for p in PRECISIONS]
    w = LEFT + len(GEARS) * (CELL + GAP) + GAP
    h = TOP + len(rows) * (CELL + GAP) + CAPTION

    img = Image.new("RGB", (w, h), (250, 249, 246))
    d = ImageDraw.Draw(img)
    f_title, f_head, f_cell = font(20), font(14), font(13)

    d.text((GAP + 4, 10), TITLES[tag], fill=(25, 25, 25), font=f_title)
    d.text((GAP + 4, 38), "行 = 模型 / 精度　列 = 画质挡位　尺寸统一 512×512　种子统一",
           fill=(110, 110, 110), font=f_cell)

    # 列头
    for c, gear in enumerate(GEARS):
        x = LEFT + c * (CELL + GAP) + CELL // 2
        d.text((x - 26, TOP - 22), gear, fill=(60, 60, 60), font=f_head)

    for r, (mkey, mlabel, prec) in enumerate(rows):
        y = TOP + r * (CELL + GAP)
        d.text((GAP, y + CELL // 2 - 16), mlabel, fill=(30, 30, 30), font=f_head)
        d.text((GAP, y + CELL // 2 + 2), prec, fill=(120, 120, 120), font=f_cell)
        for c, gear in enumerate(GEARS):
            x = LEFT + c * (CELL + GAP)
            path = os.path.join(ROOT, tag, f"{mkey}__{prec}__{gear}.png")
            if os.path.exists(path):
                im = Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
                img.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + CELL - 1, y + CELL - 1], fill=(236, 236, 236))
                d.text((x + CELL // 2 - 6, y + CELL // 2 - 8), "—", fill=(160, 160, 160), font=f_head)

    return img


def main() -> int:
    sheets = {}
    for tag in ("A", "B"):
        img = build(tag)
        path = os.path.join(ROOT, f"matrix_{tag}.png")
        img.save(path)
        sheets[tag] = img
        print("已生成:", os.path.relpath(path, PROJECT), img.size)

    a, b = sheets["A"], sheets["B"]
    w = max(a.width, b.width)
    gap = 24
    combined = Image.new("RGB", (w, a.height + b.height + gap), (255, 255, 255))
    combined.paste(a, (0, 0))
    combined.paste(b, (0, a.height + gap))
    path = os.path.join(ROOT, "matrix_combined.png")
    combined.save(path)
    print("已生成:", os.path.relpath(path, PROJECT), combined.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

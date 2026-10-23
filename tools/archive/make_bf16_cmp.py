"""把 bf16 与 fp16 的差异放大 8 倍，用于肉眼判断"是否看得出区别"。"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
PROJECT = _ROOT
OUT = os.path.join(PROJECT, "results", "bf16_vs_fp16")
PAIRS = [
    ("sd-turbo", "sd-turbo_fp16", "sd-turbo_bf16"),
    ("sd15", "sd15_fp16", "sd15_bf16"),
]


def first_png(folder: str) -> str | None:
    d = os.path.join(OUT, folder)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    return os.path.join(d, files[0]) if files else None


def main() -> int:
    rows = []
    for label, fa, fb in PAIRS:
        pa, pb = first_png(fa), first_png(fb)
        if not pa or not pb:
            print("缺图:", label)
            continue
        a = Image.open(pa).convert("RGB")
        b = Image.open(pb).convert("RGB")
        na = np.asarray(a, dtype=np.float32)
        nb = np.asarray(b, dtype=np.float32)
        diff = np.clip(np.abs(na - nb) * 8.0, 0, 255).astype(np.uint8)
        rows.append((label, a, b, Image.fromarray(diff)))

    if not rows:
        return 1

    tile = 320
    pad, head, cap = 10, 46, 26
    width = pad + 3 * (tile + pad)
    height = head + len(rows) * (tile + cap + pad)
    sheet = Image.new("RGB", (width, height), (250, 248, 243))
    draw = ImageDraw.Draw(sheet)
    f = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    fh = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)

    draw.text((pad, 14), "同种子同提示词：bf16 与 fp16 的差别（右图为差异 ×8）", fill=(25, 25, 25), font=fh)
    labels = ["fp16", "bf16", "差异 ×8"]
    y = head
    for label, a, b, d in rows:
        for i, im in enumerate((a, b, d)):
            sheet.paste(im.resize((tile, tile), Image.LANCZOS), (pad + i * (tile + pad), y))
        draw.text((pad, y + tile + 4), f"{label} · 左 fp16 / 中 bf16 / 右 差异 ×8", fill=(90, 90, 90), font=f)
        y += tile + cap + pad

    path = os.path.join(OUT, "cmp.png")
    sheet.save(path)
    print("已生成:", os.path.relpath(path, PROJECT), sheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

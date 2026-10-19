"""把样图拼成带标注的对照图（网格），用于直观比较各挡位的出图差异。

用法：
    python tools\\make_contact_sheet.py results\\bench\\samples
    python tools\\make_contact_sheet.py results\\lowres_exp -o results\\lowres_sheet.png --cols 2
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_font(size: int = 16):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description="生成样图对照网格")
    ap.add_argument("src", help="样图目录")
    ap.add_argument("-o", "--out", default=None, help="输出文件，默认 <src> 同级的 contact_sheet.png")
    ap.add_argument("--cols", type=int, default=4, help="每行张数")
    ap.add_argument("--cell", type=int, default=320, help="单格像素边长")
    ap.add_argument("--title", default=None, help="整张图的标题")
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(PROJECT, args.src)
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(".png"))
    if not files:
        print(f"目录里没有 png：{src}")
        return 1

    out = args.out or os.path.join(os.path.dirname(src), "contact_sheet.png")
    out = out if os.path.isabs(out) else os.path.join(PROJECT, out)

    cols = max(1, args.cols)
    rows = (len(files) + cols - 1) // cols
    cell = args.cell
    cap_h = 34
    pad = 10
    title_h = 44 if args.title else 0

    width = cols * (cell + pad) + pad
    height = title_h + rows * (cell + cap_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (24, 26, 30))
    draw = ImageDraw.Draw(sheet)
    font = load_font(16)
    title_font = load_font(24)

    if args.title:
        draw.text((pad, 10), args.title, fill=(235, 238, 245), font=title_font)

    for idx, fname in enumerate(files):
        r, c = divmod(idx, cols)
        x = pad + c * (cell + pad)
        y = title_h + pad + r * (cell + cap_h + pad)
        img = Image.open(os.path.join(src, fname)).convert("RGB").resize((cell, cell), Image.LANCZOS)
        sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(70, 76, 88))
        label = os.path.splitext(fname)[0].replace("_", " ")
        draw.text((x + 2, y + cell + 6), label[:46], fill=(196, 204, 220), font=font)

    sheet.save(out)
    print(f"已生成 {out}（{len(files)} 张，{cols} 列）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

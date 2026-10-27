"""把 gen_matrix.py 产出的图拼成一张带标注的效果参考图（模型无关）。

用法：
    python tools/make_sheet.py --manifest outputs/ref_matrix/sd15-sd15a/manifest.json
    python tools/make_sheet.py --manifest ... --out results/ref_matrix/sd15.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from genimg.presets import MODEL_PROFILES, QUALITY_GEARS  # noqa: E402

PRECISION_ORDER = ["fp16", "bf16", "tf32", "fp32", "fp8"]
PRECISION_NOTE = {
    "fp16": "权重/计算 fp16，VAE 解码用 bf16",
    "bf16": "权重/计算 bf16，VAE 同精度",
    "tf32": "fp32 权重 + TF32 张量核",
    "fp32": "全精度 fp32",
    "fp8": "fp8（本机不支持）",
}
PIXELS_DEFAULT = [512, 768, 1024]

PAD = 20
TILE_H = 236
LABEL_W = 190
COL_GAP = 26
CAP_H = 26
ROW_GAP = 16
BLOCK_TITLE_H = 34
COL_HDR_H = 26
BLOCK_GAP = 30

F_REG = "C:/Windows/Fonts/msyh.ttc"
F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def wrap(d: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """按像素宽度换行（中文没有空格，只能逐字量）。"""
    lines: list[str] = []
    cur = ""
    for ch in text:
        if d.textlength(cur + ch, font=f) <= max_w or not cur:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def gear_meta(gear: str) -> tuple[str, str]:
    g = QUALITY_GEARS[gear]
    note = f"{g.steps} 步 · CFG {g.guidance:g}"
    if g.guidance <= 1.0:
        note += "（无引导）"
    return g.key, note


def load_manifest(path: str | None) -> dict:
    if path is None:
        cands = sorted(glob.glob(os.path.join(PROJECT, "outputs", "ref_matrix", "*", "manifest.json")))
        if not cands:
            raise SystemExit("找不到 manifest.json，请先跑 gen_matrix.py")
        path = cands[-1]
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=0, help="精度分块排几列（0=自动：≤2 个精度单列，否则两列）")
    args = ap.parse_args()

    data = load_manifest(args.manifest)
    model = data.get("model", "sdxl")
    profile = MODEL_PROFILES[model]
    recs = {(r["precision"], r["gear"], r["pixels"]): r for r in data["records"] if r.get("ok")}

    gears = [g for g in QUALITY_GEARS if any(k[1] == g for k in recs)]
    precisions = [p for p in PRECISION_ORDER if any(k[0] == p for k in recs)]
    pixels = sorted({k[2] for k in recs})

    miss = 0
    ncols = args.cols or (1 if len(precisions) <= 2 else 2)
    nrows = (len(precisions) + ncols - 1) // ncols

    tile_w = TILE_H
    block_w = LABEL_W + len(pixels) * (tile_w + COL_GAP)
    block_h = BLOCK_TITLE_H + COL_HDR_H + len(gears) * (TILE_H + CAP_H + ROW_GAP)
    width = PAD + ncols * block_w + (ncols - 1) * BLOCK_GAP + PAD

    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title = f"{profile.label} 效果参考矩阵：自动比例 × 像素档 × 精度 × 画质挡位"
    f_title = font(F_BOLD, 27)
    for size in (27, 25, 23, 21, 19):
        f_title = font(F_BOLD, size)
        if probe.textlength(title, font=f_title) <= width - 2 * PAD:
            break
    f_sub = font(F_REG, 16)
    prompt = data.get("prompt", "")
    native = profile.native_size
    head_lines: list[str] = []
    for raw in (
        f'提示词："{prompt}"',
        "比例＝自动：提示词里没有画幅关键词 → 回落到模型原生 1:1，即 "
        + " / ".join(f"{p}×{p}" for p in pixels)
        + f"；种子固定 {data.get('seed')}。",
        "每格等比缩放到同高排版（看起来一样大），角标＝实际分辨率 · 单张纯推理耗时（不含加载权重）。",
    ):
        head_lines.extend(wrap(probe, raw, f_sub, width - 2 * PAD))

    foot_lines: list[str] = []
    foot_raw = [
        f"模型原生 {native}px："
        + ("512 档为原生分辨率，768/1024 属超原生面积出图。" if native == 512 else "1024 档为原生分辨率。"),
        "fp8 本机硬件不支持（sm_86 无 fp8 张量核），未纳入；同一格对比请只信同种子同尺寸。",
    ]
    missing_total = len(precisions) * len(gears) * len(pixels) - len(recs)
    if missing_total:
        foot_raw.append(f"留空（{missing_total} 格）＝该组合本轮未跑完：fp32 在 768/1024 下实测打满 12GB 显存（12–20 秒/步，另出现过 OOM），仅 512 档稳定。")
    for raw in foot_raw:
        foot_lines.extend(wrap(probe, raw, f_sub, width - 2 * PAD))

    head_h = 16 + 36 + 6 + len(head_lines) * 24 + 18
    foot_h = 12 + len(foot_lines) * 24
    height = head_h + nrows * block_h + (nrows - 1) * BLOCK_GAP + foot_h + PAD

    sheet = Image.new("RGB", (width, height), (247, 245, 240))
    d = ImageDraw.Draw(sheet)
    f_blk = font(F_BOLD, 21)
    f_blk_note = font(F_REG, 14)
    f_col = font(F_BOLD, 16)
    f_gear = font(F_BOLD, 18)
    f_meta = font(F_REG, 14)
    f_cap = font(F_REG, 14)

    d.text((PAD, 14), title, fill=(24, 24, 24), font=f_title)
    ty = 14 + 36 + 6
    for i, line in enumerate(head_lines):
        d.text((PAD, ty), line, fill=(58, 58, 58) if i == 0 else (104, 102, 98), font=f_sub)
        ty += 24
    d.line([(PAD, head_h - 14), (width - PAD, head_h - 14)], fill=(214, 210, 202), width=1)

    for bi, prec_key in enumerate(precisions):
        col = bi % ncols
        row = bi // ncols
        x0 = PAD + col * (block_w + BLOCK_GAP)
        y0 = head_h + row * (block_h + BLOCK_GAP)

        d.text((x0, y0), profile.label.split("（")[0] + f" · {prec_key}", fill=(28, 28, 28), font=f_blk)
        d.text(
            (x0 + d.textlength(profile.label.split("（")[0] + f" · {prec_key}", font=f_blk) + 10, y0 + 4),
            f"（{PRECISION_NOTE.get(prec_key, '')}）",
            fill=(112, 109, 104),
            font=f_blk_note,
        )
        yy = y0 + BLOCK_TITLE_H

        for ci, px in enumerate(pixels):
            d.text((x0 + LABEL_W + ci * (tile_w + COL_GAP), yy), f"{px} 像素", fill=(84, 82, 78), font=f_col)
        yy += COL_HDR_H

        for gi, gear in enumerate(gears):
            ytop = yy + gi * (TILE_H + CAP_H + ROW_GAP)
            name, meta = gear_meta(gear)
            d.text((x0, ytop + TILE_H // 2 - 20), name, fill=(24, 24, 24), font=f_gear)
            d.text((x0, ytop + TILE_H // 2 + 6), meta, fill=(110, 107, 102), font=f_meta)

            for ci, px in enumerate(pixels):
                x = x0 + LABEL_W + ci * (tile_w + COL_GAP)
                rec = recs.get((prec_key, gear, px))
                if rec is None:
                    miss += 1
                    d.rectangle([x, ytop, x + tile_w, ytop + TILE_H], fill=(238, 235, 229), outline=(203, 198, 190), width=1)
                    d.text((x + tile_w / 2, ytop + TILE_H / 2 - 8), "未生成", fill=(150, 145, 138), font=f_cap, anchor="mm")
                    d.text((x + tile_w / 2, ytop + TILE_H / 2 + 12), "超预算", fill=(170, 165, 158), font=f_cap, anchor="mm")
                    d.text((x, ytop + TILE_H + 5), f"{px}×{px} · —", fill=(150, 145, 138), font=f_cap)
                    continue
                im = Image.open(rec["path"]).convert("RGB")
                scale = TILE_H / im.height
                im = im.resize((max(1, round(im.width * scale)), TILE_H), Image.LANCZOS)
                sheet.paste(im, (x, ytop))
                d.rectangle([x, ytop, x + im.width - 1, ytop + TILE_H - 1], outline=(206, 202, 194), width=1)
                secs = rec.get("infer_seconds") or rec.get("seconds") or 0
                d.text((x, ytop + TILE_H + 5), f"{rec['width']}×{rec['height']} · {secs:.1f}s", fill=(95, 92, 88), font=f_cap)

    fy = head_h + nrows * block_h + (nrows - 1) * BLOCK_GAP + 6
    d.line([(PAD, fy - 6), (width - PAD, fy - 6)], fill=(214, 210, 202), width=1)
    for line in foot_lines:
        d.text((PAD, fy), line, fill=(104, 102, 98), font=f_sub)
        fy += 24

    out = args.out or os.path.join(PROJECT, "results", "ref_matrix", f"{model}_auto_ratio_matrix.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print("已生成", out, sheet.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

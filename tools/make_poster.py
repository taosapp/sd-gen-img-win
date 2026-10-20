"""两段式海报：上半照片、下半插画，严格 50/50。

用法：
    python tools\\make_poster.py --image C:\\Users\\PPXW\\Desktop\\1.jpg ^
        --prompt-file "C:\\Users\\PPXW\\Desktop\\水彩上下crad提示词.txt" ^
        --model sdxl --gear balanced --size 3:4 --strength 0.85 --out results\\poster_two_pass
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genimg.engine import Engine  # noqa: E402
from genimg.poster import condense_prompt, render  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402
from genimg.sizing import resolve as resolve_size  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description="两段式海报拼版")
    ap.add_argument("--image", nargs="+", required=True, help="参考照片，可多张")
    ap.add_argument("--prompt-file", required=True, help="长提示词文件（UTF-8）")
    ap.add_argument("--model", default="sdxl")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--gear", default="balanced")
    ap.add_argument("--size", default="3:4", help="画幅比例或 WxH，默认 3:4")
    ap.add_argument("--strength", type=float, default=0.82,
                    help="下半插画的改动强度：0.70 保主体可辨识，0.85 完全抽象")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--style", default="", help="覆盖画风提示词")
    ap.add_argument("--preset", choices=["watercolor", "paper"], default="watercolor",
                    help="watercolor=内置水彩描述；paper=从长提示词自动抽取手绘描述")
    ap.add_argument("--art-scale", type=float, default=0.52,
                    help="插画宽度占画布比例，0.52 ≈ 参考图的留白比例")
    ap.add_argument("--title", default="", help="主标题（大写衬线字距排版）")
    ap.add_argument("--subtitle", default="", help="副标题")
    ap.add_argument("--year", default="", help="年份")
    ap.add_argument("--out", default=os.path.join(PROJECT, "results", "poster_two_pass"))
    args = ap.parse_args()

    text = open(args.prompt_file, encoding="utf-8").read()
    style, negative = condense_prompt(text)
    print(f"提示词 {len(text)} 字符")
    if args.preset == "watercolor" and not args.style:
        from genimg.poster import WATERCOLOR_STYLE

        print(f"画风（preset=watercolor）：{WATERCOLOR_STYLE}")
    else:
        print(f"自动抽取的画风（{len(style.split())} 词）：{style}")
        print(f"自动抽取的负向（{len(negative.split(','))} 项）：{negative[:100]}...")

    model = resolve_model(args.model)
    precision = resolve_precision(args.precision)
    quality = resolve_quality(args.gear, model)
    decision = resolve_size(spec=args.size, model_key=model.key, model_native=model.native_size)
    w, h = decision.width, decision.height
    h -= h % 2  # 上下两半必须等高
    print(f"海报尺寸 {w}×{h}（每半 {w}×{h // 2}）\n")

    engine = Engine(project_root=PROJECT, verbose=False)
    for photo in args.image:
        name = os.path.splitext(os.path.basename(photo))[0]
        out_dir = os.path.join(args.out, f"{name}_{model.key}")
        try:
            res = render(
                engine,
                photo_path=photo,
                prompt_text=text,
                model=model,
                precision=precision,
                quality=quality,
                width=w,
                height=h,
                strength=args.strength,
                seed=args.seed,
                style_override=args.style,
                preset=args.preset,
                art_scale=args.art_scale,
                title=args.title,
                subtitle=args.subtitle,
                year=args.year,
                out_dir=out_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: 失败 {exc}")
            continue
        print(f"  {name:<12} → {os.path.relpath(res.path, PROJECT)}  （下半生成 {res.seconds:.2f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

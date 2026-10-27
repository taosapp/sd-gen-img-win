"""批量生成「精度 × 画质挡位 × 像素档」效果矩阵（自动比例）。

尺寸逻辑与网页界面完全一致：
    比例 = 自动 → resolve_size(base_override=像素档, prompt=提示词, model_native=模型原生)
本提示词里没有画幅关键词，因此自动回落到模型原生 1:1，三个像素档分别是
512x512 / 768x768 / 1024x1024。

用法（在项目根目录，三个模型各跑一次；挡位不填 = 该模型全部可用挡位）：
    python tools/gen_matrix.py --model sdxl     --precisions fp16,bf16
    python tools/gen_matrix.py --model sd15     --precisions fp16,bf16,tf32,fp32
    python tools/gen_matrix.py --model sd-turbo --precisions fp16,bf16,tf32,fp32

拼图：
    python tools/make_sheet.py --manifest outputs/ref_matrix/<model>-<stamp>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import (  # noqa: E402
    quality_gears_for,
    resolve_model,
    resolve_precision,
    resolve_quality,
)
from genimg.sizing import resolve as resolve_size  # noqa: E402

PROMPT = "A panda is running on the grassland. The weather is sunny."
SEED = 42


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdxl")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--precisions", default="fp16,bf16,tf32,fp32")
    ap.add_argument("--gears", default=None, help="留空 = 该模型全部可用画质挡位")
    ap.add_argument("--pixels", default="512,768,1024")
    ap.add_argument("--placement", default="auto", choices=["auto", "gpu", "offload"])
    ap.add_argument("--stamp", default=time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()

    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    pixels = [int(p) for p in args.pixels.split(",") if p.strip()]

    prompt = args.prompt
    profile = resolve_model(args.model)
    gears = (
        [g.strip() for g in args.gears.split(",") if g.strip()]
        if args.gears
        else quality_gears_for(profile)
    )
    out_root = os.path.join(PROJECT, "outputs", "ref_matrix", f"{profile.key}-{args.stamp}")
    engine = Engine(PROJECT, hf_source="mirror", verbose=True)

    total = len(precisions) * len(gears) * len(pixels)
    done = 0
    manifest_path = os.path.join(out_root, "manifest.json")
    # 支持断点续跑：先读出同目录已有记录，本次跑完按（精度, 挡位, 像素）合并
    records: list[dict] = []
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            records = json.load(fh).get("records", [])
        total += len(records)
    t_all = time.perf_counter()

    # 像素档放外层：Engine 的管线缓存键含 resolution，换分辨率必然重载权重，
    # 这样每个（精度, 像素档）只重载一次。
    for prec_key in precisions:
        precision = resolve_precision(prec_key)
        for px in pixels:
            for gear_key in gears:
                quality = resolve_quality(gear_key, profile)
                decision = resolve_size(
                    base_override=px,
                    prompt=prompt,
                    model_key=profile.key,
                    model_native=profile.native_size,
                )
                out_dir = os.path.join(out_root, prec_key, f"{px:04d}")
                req = GenRequest(
                    mode="txt2img",
                    prompt=prompt,
                    negative_prompt="",
                    model=profile.key,
                    precision=precision.key,
                    gear=quality.key,
                    width=decision.width,
                    height=decision.height,
                    size_source=decision.source,
                    size_note=decision.note,
                    seed=SEED,
                    batch=1,
                    placement=args.placement,
                    out_dir=out_dir,
                )
                done += 1
                print(
                    f"[{done}/{total}] {profile.key} / {prec_key} / {gear_key} / {px} "
                    f"-> {decision.width}x{decision.height} ({decision.ratio}) ...",
                    flush=True,
                )
                t0 = time.perf_counter()
                try:
                    res = engine.run(req, profile, precision, quality)
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! 异常 {type(exc).__name__}: {exc}", flush=True)
                    records.append(
                        {
                            "precision": prec_key,
                            "gear": gear_key,
                            "pixels": px,
                            "width": decision.width,
                            "height": decision.height,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                secs = time.perf_counter() - t0
                if not res.ok:
                    print(f"  !! 失败：{res.error}", flush=True)
                    records.append(
                        {
                            "precision": prec_key,
                            "gear": gear_key,
                            "pixels": px,
                            "width": decision.width,
                            "height": decision.height,
                            "ok": False,
                            "error": res.error,
                        }
                    )
                    continue
                print(f"  ok {secs:.1f}s -> {os.path.basename(res.images[0])} vram={res.vram.get('peak_gb')}", flush=True)
                records.append(
                    {
                        "precision": prec_key,
                        "gear": gear_key,
                        "pixels": px,
                        "steps": res.meta.get("steps"),
                        "guidance": res.meta.get("guidance"),
                        "width": decision.width,
                        "height": decision.height,
                        "ratio": decision.ratio,
                        "size_source": decision.source,
                        "seed": SEED,
                        "seconds": round(secs, 2),
                        "infer_seconds": round(float(res.seconds), 2),
                        "vram_peak_gb": res.vram.get("peak_gb"),
                        "placement": res.meta.get("placement"),
                        "prompt": prompt,
                        "path": res.images[0],
                        "ok": True,
                    }
                )
                merged = {(r["precision"], r["gear"], r["pixels"]): r for r in records}
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "prompt": prompt,
                            "model": profile.key,
                            "seed": SEED,
                            "size_mode": "auto",
                            "total_seconds": round(time.perf_counter() - t_all, 1),
                            "records": list(merged.values()),
                        },
                        fh,
                        ensure_ascii=False,
                        indent=2,
                    )

    ok = sum(1 for r in records if r.get("ok"))
    print(f"\n完成：{ok}/{total} 张，用时 {time.perf_counter() - t_all:.0f}s，输出目录 {out_root}", flush=True)
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

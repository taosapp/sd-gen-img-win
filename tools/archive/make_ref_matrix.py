"""生成 512×512 参考矩阵：模型 × 精度 × 画质挡位 × 两组提示词。

产物：
  results/ref_matrix/<A|B>/<model>__<precision>__<gear>.png
  results/ref_matrix/timing.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import (  # noqa: E402
    MODEL_PROFILES,
    PRECISION_GEARS,
    check_precision_support,
    probe_hardware,
    quality_gears_for,
    resolve_model,
    resolve_precision,
    resolve_quality,
)

PROJECT = _ROOT
OUT = os.path.join(PROJECT, "results", "ref_matrix")
SIZE = 512
SEED = 20261022

PROMPTS = {
    "A": "a panda running on a green grassland, sunny day",
    "B": "a panda running on a green grassland, sunny day, comic book illustration, "
         "bold black ink outlines, flat colors, halftone dots",
}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    hw = probe_hardware()
    engine = Engine(project_root=PROJECT, verbose=False)

    rows = []
    current = [None, None]  # (model, precision)
    total = 0
    t_all = time.perf_counter()

    # 循环顺序：模型 → 精度 → 挡位 → 提示词。
    # 同一精度下把 A/B 两组提示词连着跑，避免反复重载（SDXL fp32 一次加载要 90s）。
    for model_key, prof in MODEL_PROFILES.items():
        gears_all = quality_gears_for(prof)
        for prec_gear in PRECISION_GEARS.values():
            if check_precision_support(prec_gear, hw, prof):
                print(f"  跳过 {model_key}/{prec_gear.key}：{check_precision_support(prec_gear, hw, prof)}", flush=True)
                continue
            # 已被实测否决的组合（单张 > 1 分钟）不参与矩阵
            if prec_gear.key in getattr(prof, "blocked_precisions", ()):
                print(f"  跳过 {model_key}/{prec_gear.key}：单张 > 60 秒，已禁用", flush=True)
                continue
            gears = gears_all
            if current[0] != model_key or current[1] != prec_gear.key:
                engine.unload()  # 换模型/精度必须卸载，否则显存互挤
                current[:] = [model_key, prec_gear.key]
            for gear_key in gears:
                for tag, prompt in PROMPTS.items():
                    prof = resolve_model(model_key)
                    prec = resolve_precision(prec_gear.key)
                    q = resolve_quality(gear_key, prof)
                    req = GenRequest(
                        mode="txt2img",
                        prompt=prompt,
                        model=prof.key,
                        precision=prec.key,
                        gear=q.key,
                        width=SIZE,
                        height=SIZE,
                        seed=SEED,
                        out_dir=os.path.join(OUT, tag, "tmp"),
                        style_expand="auto",
                    )
                    t0 = time.perf_counter()
                    res = engine.run(req, prof, prec, q)
                    dt = time.perf_counter() - t0
                    total += 1
                    if not res.ok:
                        print(f"  {tag} {model_key}/{prec.key}/{gear_key}: 失败 {res.error}", flush=True)
                        rows.append([tag, model_key, prec.key, gear_key, "", f"{dt:.2f}", "error", res.error[:80]])
                        continue
                    dst = os.path.join(OUT, tag, f"{model_key}__{prec.key}__{gear_key}.png")
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.replace(res.images[0], dst)
                    # 计时口径：用引擎的 infer_total（只包住亚运，不含权重加载）
                    hot = float(res.seconds or 0.0)
                    rows.append([tag, model_key, prec.key, gear_key, f"{hot:.2f}", f"{dt:.2f}", "ok", ""])
                    print(
                        f"  [{total:3d}] {tag} {model_key:<9}{prec.key:<5}{gear_key:<11}"
                        f"热态 {hot:6.2f}s（含加载 {dt:6.2f}s）",
                        flush=True,
                    )

    # 建目录
    for tag in PROMPTS:
        os.makedirs(os.path.join(OUT, tag), exist_ok=True)

    # 清理空 tmp 目录
    for tag in PROMPTS:
        tmp = os.path.join(OUT, tag, "tmp")
        if os.path.isdir(tmp) and not os.listdir(tmp):
            os.rmdir(tmp)

    csv_path = os.path.join(OUT, "timing.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt", "model", "precision", "gear", "hot_seconds", "wall_seconds", "status", "error"])
        w.writerows(rows)

    print(f"\n完成 {total} 张，总耗时 {(time.perf_counter() - t_all) / 60:.1f} 分钟", flush=True)
    print("明细:", os.path.relpath(csv_path, PROJECT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

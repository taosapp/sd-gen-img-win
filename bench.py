"""基准测试：在 512x512 下逐挡位测量延迟与显存，验证 30 秒预算。

设计要点
--------
* 同一进程内复用管线：切画质挡位不重新加载，只有换精度挡位才重载。
* 每个配置先预热再计时，同时记录"冷启动（含加载）"与"稳态"两种口径。
* 跑不通的配置不跳过、不编数据：把失败原因记进结果表，作为选型依据的一部分。

用法：
    python bench.py                       # 默认矩阵
    python bench.py --reps 5 --warmup 2
    python bench.py --models sdxl --precisions fp16 bf16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import (  # noqa: E402
    check_precision_support,
    probe_hardware,
    resolve_model,
    resolve_precision,
    resolve_quality,
)
from genimg.runtime import configure_hf  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MATRIX: list[tuple[str, str, str]] = [
    # (模型, 精度挡位, 画质挡位)
    ("sd15", "fp32", "balanced"),
    ("sd15", "tf32", "balanced"),
    ("sd15", "bf16", "balanced"),
    ("sd15", "fp16", "balanced"),
    ("sd15", "fp16", "quality"),
    ("sdxl", "fp32", "balanced"),
    ("sdxl", "tf32", "balanced"),
    ("sdxl", "bf16", "balanced"),
    ("sdxl", "fp16", "balanced"),
    ("sdxl", "fp16", "draft"),
    ("sdxl", "fp16", "quality"),
    ("sdxl", "fp16", "lightning"),
    ("sdxl", "fp16", "lightning8"),
    ("sdxl", "fp8", "balanced"),
    ("sd-turbo", "fp32", "turbo"),
    ("sd-turbo", "tf32", "turbo"),
    ("sd-turbo", "bf16", "turbo"),
    ("sd-turbo", "fp16", "turbo"),
]

PROMPT = "a red panda astronaut floating in space, cinematic lighting, highly detailed"
NEGATIVE = "blurry, low quality, watermark"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="生图挡位基准测试")
    ap.add_argument("--models", nargs="*", default=None, help="只测这些模型")
    ap.add_argument("--precisions", nargs="*", default=None, help="只测这些精度挡位")
    ap.add_argument("--gears", nargs="*", default=None, help="只测这些画质挡位")
    ap.add_argument("--size", default="512x512", help="分辨率，默认 512x512")
    ap.add_argument("--reps", type=int, default=3, help="每种配置计时重复次数")
    ap.add_argument("--warmup", type=int, default=1, help="计时前预热次数")
    ap.add_argument(
        "--timeout-per-run",
        type=float,
        default=240.0,
        help="单张图硬预算（秒）。超过即中止并记为超时，避免被不实用的挡位拖住（0=不限制）",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join("results", "bench"), help="结果输出目录")
    ap.add_argument("--hf-source", choices=["huggingface", "mirror"], default="mirror")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--list-only", action="store_true", help="只打印将要跑的组合，不执行")
    return ap.parse_args()


def build_matrix(args) -> list[tuple[str, str, str]]:
    matrix = DEFAULT_MATRIX
    if args.models:
        matrix = [m for m in matrix if m[0] in set(args.models)]
    if args.precisions:
        matrix = [m for m in matrix if m[1] in set(args.precisions)]
    if args.gears:
        matrix = [m for m in matrix if m[2] in set(args.gears)]
    return matrix


def main() -> int:
    args = parse_args()
    w, h = (int(x) for x in args.size.lower().replace("×", "x").split("x"))
    per_run_timeout = args.timeout_per_run if args.timeout_per_run and args.timeout_per_run > 0 else None
    matrix = build_matrix(args)
    out_dir = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    hw = probe_hardware()
    print("=" * 110)
    print(f"基准测试 | {hw.summary()}")
    print(f"分辨率 {w}x{h} | 每配置 预热 {args.warmup} 次 + 计时 {args.reps} 次 | 共 {len(matrix)} 种配置")
    if per_run_timeout:
        print(f"单张硬预算 {per_run_timeout:.0f}s（超时记为 timeout 并继续）")
    print("=" * 110)
    for i, (m, p, g) in enumerate(matrix, 1):
        print(f"  {i:2d}. {m:<11} {p:<5} {g}")
    print()
    if args.list_only:
        return 0

    configure_hf(PROJECT_ROOT, source=args.hf_source, proxy=args.proxy, verbose=True)
    engine = Engine(PROJECT_ROOT, hf_source=args.hf_source, hf_proxy=args.proxy, verbose=False)

    rows: list[dict] = []
    t_start = time.perf_counter()

    for idx, (model_key, pkey, gkey) in enumerate(matrix, 1):
        profile = resolve_model(model_key)
        precision = resolve_precision(pkey)
        print("-" * 110)
        print(f"[{idx}/{len(matrix)}] {model_key} / {pkey} / {gkey}")

        why = check_precision_support(precision, hw)
        if why:
            print(f"  跳过（硬件不支持）：{why}")
            rows.append({"model": model_key, "precision": pkey, "gear": gkey, "status": "unsupported", "reason": why})
            continue

        try:
            quality = resolve_quality(gkey, profile)
        except ValueError as exc:
            print(f"  跳过（组合非法）：{exc}")
            rows.append({"model": model_key, "precision": pkey, "gear": gkey, "status": "invalid", "reason": str(exc)})
            continue

        req = GenRequest(
            mode="txt2img",
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            model=model_key,
            precision=pkey,
            gear=gkey,
            width=w,
            height=h,
            seed=args.seed,
        )

        # 预热（同时把管线 load 时间从稳态计时中剥离）
        load_seconds = None
        try:
            res = engine.run(
                req,
                profile,
                precision,
                quality,
                cudnn_benchmark=args.cudnn_benchmark,
                timeout_seconds=per_run_timeout,
            )
            load_seconds = res.meta.get("load_seconds") if res.ok else None
            if not res.ok:
                raise RuntimeError(res.error)
        except Exception as exc:  # noqa: BLE001
            text = f"{type(exc).__name__}:{exc}"
            status = "timeout" if ("TimeoutError" in text or "超过" in str(exc)) else "failed"
            print(f"  {'超时' if status == 'timeout' else '失败'}：{str(exc)[:140]}")
            rows.append(
                {
                    "model": model_key,
                    "precision": pkey,
                    "gear": gkey,
                    "status": status,
                    "reason": text[:400],
                }
            )
            continue

        # 保存一张样图
        sample_path = os.path.join(samples_dir, f"{model_key}_{pkey}_{gkey}.png")
        try:
            import shutil

            shutil.copyfile(res.images[0], sample_path)
            res.meta["sample"] = os.path.relpath(sample_path, PROJECT_ROOT)
        except Exception:  # noqa: BLE001
            pass

        print(
            f"  加载 {load_seconds:.1f}s | 预热推理 {res.stage_seconds.get('infer_total', 0):.2f}s "
            f"| 显存峰值 {res.vram.get('peak_allocated_gb', 0):.2f}GB | 放置 {res.meta.get('placement')}"
        )

        for _ in range(max(0, args.warmup - 1)):
            engine.run(
                req,
                profile,
                precision,
                quality,
                cudnn_benchmark=args.cudnn_benchmark,
                timeout_seconds=per_run_timeout,
            )

        times, peaks, denoises = [], [], []
        for r in range(args.reps):
            res = engine.run(
                req,
                profile,
                precision,
                quality,
                cudnn_benchmark=args.cudnn_benchmark,
                timeout_seconds=per_run_timeout,
            )
            if not res.ok:
                print(f"  第 {r + 1} 次重复失败：{res.error.splitlines()[0] if res.error else '未知'}")
                break
            times.append(res.stage_seconds.get("infer_total", 0.0))
            denoises.append(res.stage_seconds.get("denoise", 0.0))
            peaks.append(res.vram.get("peak_allocated_gb", 0.0))
            print(
                f"    第 {r + 1} 次：{times[-1]:6.2f}s（去噪 {denoises[-1]:5.2f}s，编解码 "
                f"{res.stage_seconds.get('encode_decode', 0):4.2f}s）"
            )

        if not times:
            rows.append(
                {"model": model_key, "precision": pkey, "gear": gkey, "status": "failed", "reason": "所有重复均失败"}
            )
            continue

        median = statistics.median(times)
        rows.append(
            {
                "model": model_key,
                "precision": pkey,
                "gear": gkey,
                "status": "ok",
                "steps": res.meta.get("steps"),
                "guidance": res.meta.get("guidance"),
                "scheduler": res.meta.get("scheduler"),
                "load_seconds": round(load_seconds or 0.0, 2),
                "median_infer_s": round(median, 3),
                "min_infer_s": round(min(times), 3),
                "max_infer_s": round(max(times), 3),
                "median_denoise_s": round(statistics.median(denoises), 3),
                "decode_s": round(res.stage_seconds.get("encode_decode", 0.0), 3),
                "peak_vram_gb": round(max(peaks), 2),
                "placement": res.meta.get("placement"),
                "within_30s": median <= 30.0,
                "sample": res.meta.get("sample", ""),
            }
        )

    # ------------------------------------------------------------------ 汇总输出
    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 110)
    print(f"汇总（总耗时 {elapsed / 60:.1f} 分钟）")
    print("=" * 110)
    head = (
        f"{'模型':<11}{'精度':<6}{'画质':<11}{'步数':>4}{'中位耗时':>10}{'去噪':>8}{'编解码':>8}"
        f"{'显存峰值':>10}{'放置':>16}{'≤30s':>7}"
    )
    print(head)
    print("-" * 110)
    for r in rows:
        if r["status"] != "ok":
            tag = {"unsupported": "硬件不支持", "invalid": "组合非法", "failed": "失败", "timeout": "超时中止"}.get(
                r["status"], r["status"]
            )
            print(
                f"{r['model']:<11}{r['precision']:<6}{r['gear']:<11}{'--':>4}{'--':>10}{'--':>8}{'--':>8}"
                f"{'--':>10}{tag:>16}{'--':>7}"
            )
            continue
        print(
            f"{r['model']:<11}{r['precision']:<6}{r['gear']:<11}{r['steps']:>4}"
            f"{r['median_infer_s']:>9.2f}s{r['median_denoise_s']:>7.2f}s{r['decode_s']:>7.2f}s"
            f"{r['peak_vram_gb']:>9.2f}G{r['placement']:>16}{'是' if r['within_30s'] else '否':>7}"
        )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        passed = [r for r in ok_rows if r["within_30s"]]
        print()
        print(
            f"可跑通配置 {len(ok_rows)} 个，其中 {len(passed)} 个满足 30 秒预算；"
            f"最快 {min(r['median_infer_s'] for r in ok_rows):.2f}s，"
            f"最慢 {max(r['median_infer_s'] for r in ok_rows):.2f}s"
        )

    # 落盘
    json_path = os.path.join(out_dir, "bench.json")
    csv_path = os.path.join(out_dir, "bench.csv")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "hardware": {
                    "gpu": hw.name,
                    "cc": f"sm_{hw.cc[0]}{hw.cc[1]}",
                    "vram_gb": round(hw.total_gb, 2),
                    "driver": hw.driver,
                    "torch": hw.torch_version,
                    "cuda": hw.cuda_version,
                },
                "config": {
                    "size": f"{w}x{h}",
                    "reps": args.reps,
                    "warmup": args.warmup,
                    "seed": args.seed,
                    "timeout_per_run": args.timeout_per_run,
                    "prompt": PROMPT,
                    "negative": NEGATIVE,
                },
                "rows": rows,
                "elapsed_minutes": round(elapsed / 60, 2),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    fields = [
        "model",
        "precision",
        "gear",
        "status",
        "steps",
        "guidance",
        "scheduler",
        "load_seconds",
        "median_infer_s",
        "min_infer_s",
        "max_infer_s",
        "median_denoise_s",
        "decode_s",
        "peak_vram_gb",
        "placement",
        "within_30s",
        "sample",
        "reason",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果已写入：{os.path.relpath(json_path, PROJECT_ROOT)}、{os.path.relpath(csv_path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

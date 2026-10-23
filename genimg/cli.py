"""命令行入口：解析参数、合并提示词内嵌指令、调用引擎、输出结果与元数据。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .advice import check_request
from .zh_prompt import contains_cjk
from .directives import describe as describe_directives
from .directives import parse_directives
from .engine import Engine, GenRequest, is_repo_cached
from .sizing import image_size
from .sizing import resolve as resolve_size
from .presets import (
    MODEL_PROFILES,
    PRECISION_GEARS,
    QUALITY_GEARS,
    check_precision_support,
    probe_hardware,
    quality_gears_for,
    resolve_model,
    resolve_precision,
    resolve_quality,
)
from .runtime import configure_hf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genimg",
        description="本机生图 agent：文生图 / 图生图，带精度挡位与画质挡位（默认 512x512）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "提示词内嵌指令示例：\n"
            '  genimg.py "a red panda astronaut, cinematic [precision=fp32] [gear=max]"\n'
            '  genimg.py "水墨山水 [尺寸=640x640] [模型=sdxl] [步数=30]"\n'
            "命令行参数优先级高于提示词内嵌指令。\n"
        ),
    )
    p.add_argument("prompt", nargs="?", default=None, help="提示词（可内嵌 [精度=fp16] 之类的指令）")
    p.add_argument("--mode", choices=["txt2img", "img2img"], default=None, help="生图模式，默认 txt2img")
    p.add_argument("--image", "-i", default=None, help="图生图的初始图片路径")
    p.add_argument("--strength", type=float, default=None, help="图生图改动强度 0~1，越大改动越多（默认 0.5，保主体）")
    p.add_argument("--model", "-m", default=None, help="底模：sdxl / sd15 / sd-turbo")
    p.add_argument("--precision", "-p", default=None, help="精度挡位：fp16 / bf16 / tf32 / fp32 / fp8")
    p.add_argument(
        "--gear", "-g", default=None, help="画质挡位：turbo / lightning / lightning8 / draft / balanced / quality / max"
    )
    p.add_argument(
        "--size",
        "-s",
        default=None,
        help="输出尺寸：固定像素 768x1024，或比例 3:4 / 2:3 / 16:9，或 match（跟随参考图）；默认按提示词/参考图/模型原生自动判断",
    )
    p.add_argument("--width", type=int, default=None, help="输出宽度（需与 --height 同时给出）")
    p.add_argument("--height", type=int, default=None, help="输出高度（需与 --width 同时给出）")
    p.add_argument("--steps", type=int, default=None, help="采样步数（覆盖画质挡位）")
    p.add_argument("--guidance", "--cfg", dest="guidance", type=float, default=None, help="CFG 引导强度")
    p.add_argument("--seed", type=int, default=None, help="随机种子，固定后可复现")
    p.add_argument("--batch", "-n", type=int, default=None, help="一次生成几张（默认 1）")
    p.add_argument("--negative", default=None, help="负向提示词")
    p.add_argument("--placement", choices=["auto", "gpu", "offload"], default=None, help="显存放置策略，默认 auto")
    p.add_argument("--out", default=None, help="输出目录（默认 outputs/<时间戳>）")
    p.add_argument("--repeat", type=int, default=1, help="重复生成若干次并汇总耗时（用于验收速度）")
    p.add_argument("--timeout", type=float, default=0.0, help="单张图的秒数硬预算，超时主动中止（0=不限制）")
    p.add_argument("--warmup", type=int, default=0, help="正式计时前先空跑几次（用于排除首次预热开销）")
    p.add_argument("--json", action="store_true", help="只输出 JSON（便于别的程序调用）")
    p.add_argument("--manifest", default=None, help="把完整元数据写入指定 JSON 文件")
    p.add_argument("--dry-run", action="store_true", help="只解析参数并打印执行计划，不真正生图")
    p.add_argument("--list", action="store_true", help="列出本机可用模型/精度挡位/画质挡位与硬件能力")
    p.add_argument("--hf-source", choices=["huggingface", "mirror"], default="mirror", help="权重下载源")
    p.add_argument("--proxy", default=None, help="HF 下载代理，例如 http://10.10.100.199:3128（默认自动探测）")
    p.add_argument("--cudnn-benchmark", action="store_true", help="开启 cuDNN 自动调优（固定尺寸下略快）")
    p.add_argument("--safety-checker", action="store_true", help="启用 NSFW 安全检查器（默认关闭，会额外下载安全模型）")
    p.add_argument(
        "--translate",
        choices=["dict", "off"],
        default=None,
        help="中文提示词处理：dict（默认，离线词典） | off（不翻译）",
    )
    p.add_argument(
        "--style-expand",
        choices=["auto", "off"],
        default="auto",
        help="抽象风格名自动展开成具体视觉特征（如 American comic → ink outlines/flat colors/halftone dots）；off 关闭",
    )
    return p


# --------------------------------------------------------------------------------------
# 参数合并
# --------------------------------------------------------------------------------------
def _parse_size(value: str):
    """保留给外部调用：把 ``3:4`` / ``768x1024`` 解成尺寸或比例。"""
    from .sizing import parse_spec

    return parse_spec(value)


def resolve_request(args, project_root: str):
    """把 CLI 参数 + 提示词指令 + 默认值合并成一个 GenRequest，并记录来源。"""
    raw_prompt = args.prompt or ""
    parsed = parse_directives(raw_prompt)
    d = parsed.values
    warnings = list(parsed.warnings)
    source: dict[str, str] = {}

    def pick(cli_value, key, default=None):
        """优先级：显式 CLI > 提示词指令 > 默认值。"""
        if cli_value is not None:
            source[key] = "cli"
            return cli_value
        if key in d:
            source[key] = "prompt"
            return d[key]
        source[key] = "default"
        return default

    model_key = pick(args.model, "model", "sd-turbo")
    if "model" in d and args.model is None:
        try:
            model_key = resolve_model(str(d["model"])).key
        except KeyError as exc:
            warnings.append(str(exc))
            model_key = "sd-turbo"
    profile = resolve_model(str(model_key))

    # 默认 bf16：本机实测与 fp16 同速或更快（sd-turbo 0.34s vs 0.45s）、显存更小、
    # 且动态范围与 fp32 一致，不需要为 VAE 单独升精度
    precision_name = pick(args.precision, "precision", "bf16")
    precision = resolve_precision(str(precision_name))

    gear_name = pick(args.gear, "gear", profile.default_gear)
    quality = resolve_quality(str(gear_name), profile)

    # 尺寸优先级：显式宽高 > --size（512x768 / 3:4 / match）> 提示词指令 > 提示词里的画幅描述 > 参考图 > 模型原生
    spec = args.size
    if spec is None and "size" in d:
        parsed_size = d["size"]
        if parsed_size[0] == "wh":
            spec = f"{parsed_size[1]}x{parsed_size[2]}"
        else:
            spec = f"{parsed_size[1]:g}:{parsed_size[2]:g}"

    init_size = image_size(args.image) if getattr(args, "image", None) else None
    decision = resolve_size(
        width=args.width,
        height=args.height,
        spec=spec,
        prompt=parsed.prompt,
        model_key=profile.key,
        init_size=init_size,
        model_native=profile.native_size,
    )
    width, height = decision.width, decision.height
    source["size"] = decision.source
    if decision.note:
        warnings.append(f"尺寸：{decision.note}")

    steps = pick(args.steps, "steps", None)
    guidance = pick(args.guidance, "guidance", None)
    seed = pick(args.seed, "seed", None)
    batch = pick(args.batch, "batch", 1)
    negative = pick(args.negative, "negative", "")
    strength = pick(args.strength, "strength", 0.5)
    placement = pick(args.placement, "placement", "auto")

    mode = args.mode or "txt2img"
    if args.image and args.mode is None:
        mode = "img2img"

    req = GenRequest(
        mode=mode,
        prompt=parsed.prompt or raw_prompt,
        negative_prompt=str(negative or ""),
        model=profile.key,
        precision=precision.key,
        gear=quality.key,
        width=int(width),
        height=int(height),
        size_source=decision.source,
        size_note=decision.note,
        steps=int(steps) if steps is not None else None,
        guidance=float(guidance) if guidance is not None else None,
        seed=int(seed) if seed is not None else None,
        batch=int(batch),
        strength=float(strength),
        init_image=args.image,
        placement=str(placement),
        out_dir=args.out,
        translate=args.translate or "dict",
        style_expand=args.style_expand,
    )
    return req, profile, precision, quality, parsed, source, warnings


# --------------------------------------------------------------------------------------
# 列表输出
# --------------------------------------------------------------------------------------
def print_catalog(project_root: str, hf_source: str, proxy: str | None) -> None:
    # 先落定 HF 缓存位置，否则缓存检测会去看默认目录（~/.cache）而误报未下载
    configure_hf(project_root, source=hf_source, proxy=proxy, verbose=False)

    hw = probe_hardware()
    print("=" * 96)
    print("硬件能力")
    print("=" * 96)
    print("  " + hw.summary())
    print()

    print("=" * 96)
    print("模型档案")
    print("=" * 96)
    for key, prof in MODEL_PROFILES.items():
        cached = "已缓存" if is_repo_cached(prof.repo, project_root) else "未下载"
        print(f"  {key:<12} {prof.label:<34} 原生 {prof.native_size}px  [{cached}]")
        print(f"               {prof.repo}")
        print(f"               可用画质挡位：{', '.join(quality_gears_for(prof))}")
        print(f"               {prof.note}")
    print()

    print("=" * 96)
    print("精度挡位")
    print("=" * 96)
    for key, gear in PRECISION_GEARS.items():
        why = check_precision_support(gear, hw)
        status = "可用" if why is None else "不可用"
        print(f"  {key:<6} {gear.label:<34} [{status}]")
        if why:
            print(f"         原因：{why}")
        else:
            print(f"         {gear.note}")
    print()

    print("=" * 96)
    print("画质挡位")
    print("=" * 96)
    for key, gear in QUALITY_GEARS.items():
        scope = "全部模型" if not gear.only_family else "、".join(gear.only_family)
        print(f"  {key:<12} {gear.label:<38} 步数 {gear.steps:<3} CFG {gear.guidance:<4} 适用：{scope}")
        print(f"               {gear.note}")
    print()


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)  # genimg/ 的上一级

    if args.list:
        print_catalog(project_root, args.hf_source, args.proxy)
        return 0

    if not args.prompt:
        build_parser().print_help()
        return 2

    try:
        req, profile, precision, quality, parsed, source, warnings = resolve_request(args, project_root)
    except (KeyError, ValueError) as exc:
        # 挡位/模型名写错或组合非法：给一句人话，外加可用取值，不抛堆栈
        msg = str(exc).strip('"')
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False, indent=2))
        else:
            print(f"[错误] {msg}")
        return 2

    if not args.json:
        print("=" * 96)
        print("执行计划")
        print("=" * 96)
        print(
            f"  模式          {req.mode}"
            + (f"（初始图 {req.init_image}，强度 {req.strength}）" if req.mode == "img2img" else "")
        )
        print(f"  提示词        {req.prompt!r}")
        if contains_cjk(req.prompt) and req.translate != "off":
            print(f"  中文提示      检测到中文，生成前自动转英文（方式：{req.translate}）")
        if req.negative_prompt:
            print(f"  负向提示词    {req.negative_prompt!r}")
        print(f"  模型          {profile.key}（{profile.repo}）")
        print(f"  精度挡位      {precision.key} — {precision.label}")
        print(f"  画质挡位      {quality.key} — {quality.label}")
        print(f"  分辨率        {req.width}x{req.height}")
        print(
            f"  步数 / CFG    {req.steps if req.steps is not None else quality.steps} / "
            f"{req.guidance if req.guidance is not None else quality.guidance}"
        )
        print(f"  种子 / 张数   {req.seed if req.seed is not None else '随机'} / {req.batch}")
        print(f"  显存放置      {req.placement}")
        if parsed.tokens:
            print(f"  {describe_directives(parsed)}")
        for w in warnings:
            print(f"  [告警] {w}")
        print("  参数来源      " + ", ".join(f"{k}={v}" for k, v in source.items()))
        tips = check_request(
            mode=req.mode,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            steps=req.steps if req.steps is not None else quality.steps,
            strength=req.strength,
            guidance=req.guidance if req.guidance is not None else quality.guidance,
            model_key=profile.key,
            skip_cjk_tip=bool(contains_cjk(req.prompt) and req.translate != "off"),
        )
        for tip in tips:
            print(f"  [提示] {tip}")
        print()

    hw = probe_hardware()
    why = check_precision_support(precision, hw, profile)
    if why:
        msg = f"精度挡位 {precision.key} 在本机不可用：{why}"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False, indent=2))
        else:
            print(f"[错误] {msg}")
            print(
                "[提示] 本机可用挡位："
                + "、".join(k for k, g in PRECISION_GEARS.items() if check_precision_support(g, hw, profile) is None)
            )
        return 3

    if args.dry_run:
        print("[dry-run] 参数解析完成，未执行推理。")
        return 0

    engine = Engine(
        project_root,
        hf_source=args.hf_source,
        hf_proxy=args.proxy,
        verbose=not args.json,
        use_safety_checker=args.safety_checker,
    )

    records = []
    # 预热阶段不计入统计
    for _ in range(max(0, args.warmup)):
        if not args.json:
            print(f"[warmup] 预热 {_ + 1}/{args.warmup} …")
        res = engine.run(req, profile, precision, quality, cudnn_benchmark=args.cudnn_benchmark)
        if not res.ok:
            print(f"[错误] 预热失败：{res.error}")
            return 4

    timeout = args.timeout if args.timeout and args.timeout > 0 else None
    for i in range(max(1, args.repeat)):
        if not args.json and args.repeat > 1:
            print(f"\n[运行 {i + 1}/{args.repeat}]")
        t0 = time.perf_counter()
        res = engine.run(
            req,
            profile,
            precision,
            quality,
            cudnn_benchmark=args.cudnn_benchmark,
            timeout_seconds=timeout,
        )
        wall = time.perf_counter() - t0
        if not res.ok:
            if args.json:
                print(json.dumps({"ok": False, "error": res.error}, ensure_ascii=False, indent=2))
            else:
                print(f"[错误] 生图失败：{res.error}")
            return 4
        records.append({"wall": wall, "result": res})

    last = records[-1]["result"]

    if args.json:
        payload = {
            "ok": True,
            "request": {
                "mode": req.mode,
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "width": req.width,
                "height": req.height,
                "batch": req.batch,
                "seed": req.seed,
            },
            "runs": [r["result"].meta for r in records],
            "wall_seconds": [round(r["wall"], 3) for r in records],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print()
        print("=" * 96)
        print("结果")
        print("=" * 96)
        for path in last.images:
            print(f"  输出          {os.path.relpath(path, project_root)}")
        print("\n  分阶段耗时（不含权重加载）：")
        print(f"    推理总计    {last.stage_seconds.get('infer_total', 0):7.3f} s")
        print(f"      ├ 去噪     {last.stage_seconds.get('denoise', 0):7.3f} s  ← UNet 采样循环")
        print(f"      └ 编解码   {last.stage_seconds.get('encode_decode', 0):7.3f} s  ← 文本编码 + VAE 解码")
        print(f"    权重加载    {last.meta.get('load_seconds', 0):7.3f} s（首次加载计入，之后复用）")
        steps_t = last.meta.get("step_seconds", [])
        if steps_t:
            print(
                f"    单步耗时    首步 {steps_t[0]:.3f} s，稳态均值 "
                f"{sum(steps_t[1:]) / max(1, len(steps_t[1:])):.3f} s，共 {len(steps_t)} 步"
            )
        print(
            f"    显存峰值    {last.vram.get('peak_allocated_gb', 0):.2f} GB 已分配 / "
            f"{last.vram.get('peak_reserved_gb', 0):.2f} GB 保留"
        )
        if args.repeat > 1:
            walls = [r["wall"] for r in records]
            print(
                f"\n  重复 {len(walls)} 次墙钟耗时：{', '.join(f'{w:.2f}s' for w in walls)}"
                f" → 中位数 {sorted(walls)[len(walls) // 2]:.2f} s"
            )
        infer = last.stage_seconds.get("infer_total", 0)
        verdict = "满足" if infer <= 30 else "不满足"
        print(f"\n  速度验收    512x512 单张推理 {infer:.2f} s → {verdict} 30 秒预算")
        print(
            f"  复现命令    python genimg.py {json.dumps(req.prompt, ensure_ascii=False)} "
            f"--model {profile.key} --precision {precision.key} --gear {quality.key} "
            f"--size {req.width}x{req.height}" + (f" --seed {req.seed}" if req.seed is not None else "")
        )

    if args.manifest:
        manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
        os.makedirs(manifest_dir, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "runs": [r["result"].meta for r in records],
                    "wall_seconds": [round(r["wall"], 3) for r in records],
                    "directives": parsed.values,
                    "source": source,
                    "warnings": warnings,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        if not args.json:
            print(f"\n  元数据已写入 {args.manifest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

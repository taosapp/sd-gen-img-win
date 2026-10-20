"""3:4 竖版尺寸下的耗时测量：确认哪种挡位仍能落在 30 秒内。"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO = r"C:\Users\PPXW\Desktop\1.jpg"
PROMPT = "watercolor illustration, minimal editorial poster, soft paper texture, negative space"

CASES = [
    ("sd15", "bf16", "balanced", 384, 512),
    ("sdxl", "fp16", "lightning", 768, 1024),
    ("sdxl", "fp16", "draft", 768, 1024),
    ("sdxl", "fp16", "balanced", 768, 1024),
]

engine = Engine(project_root=PROJECT, verbose=False)
current = [None]

print(f"{'模型':<8}{'精度':<6}{'挡位':<11}{'尺寸':<11}{'步数':>4}  {'推理':>8}  {'峰值显存':>8}  放置")
print("-" * 74)
for model_key, prec_key, gear, w, h in CASES:
    prof = resolve_model(model_key)
    prec = resolve_precision(prec_key)
    q = resolve_quality(gear, prof)
    if current[0] is not None and current[0] != prof.key:
        engine.unload()  # 换模型必须卸载，否则两条管线驻留会逼出顺序卸载
    current[0] = prof.key

    req = GenRequest(
        mode="img2img",
        prompt=PROMPT,
        model=prof.key,
        precision=prec.key,
        gear=q.key,
        width=w,
        height=h,
        seed=42,
        strength=0.55,
        init_image=PHOTO,
        out_dir=os.path.join(PROJECT, "results", "ratio_timing", f"{model_key}_{prec_key}_{gear}"),
        size_source="bench",
    )
    t0 = time.perf_counter()
    res = engine.run(req, prof, prec, q)  # 第一次：预热（会触发放置策略与内存分配）
    warm = time.perf_counter() - t0
    if not res.ok:
        print(f"{model_key:<8}{prec_key:<6}{gear:<11}{w}x{h:<8}  失败: {res.error[:40]}")
        continue
    t0 = time.perf_counter()
    res = engine.run(req, prof, prec, q)  # 第二次：计时（热态）
    total = time.perf_counter() - t0
    if not res.ok:
        print(f"{model_key:<8}{prec_key:<6}{gear:<11}{w}x{h:<8}{'-':>4}  失败: {res.error[:40]}")
        continue
    meta = res.meta or {}
    vram = (meta.get("vram") or {}).get("peak_gb")
    print(
        f"{model_key:<8}{prec_key:<6}{gear:<11}{f'{w}x{h}':<11}{str(meta.get('steps', '?')):>4}  "
        f"{res.seconds:6.2f}s  {vram if vram is not None else '-':>8}  {meta.get('placement', '-')}"
        f"   （预热次 {warm:.1f}s）",
        flush=True,
    )

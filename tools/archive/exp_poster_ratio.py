"""对照实验：长提示词 + 横构图参考图 → 3:4 竖版。

三组：
  A 旧行为：参考图硬拉伸到 512×512（复现变形）
  B 新行为：sd15 自动选 384×512（3:4），等比缩放 + 模糊留白
  C 新行为：sdXL 自动选 768×1024（3:4），推荐档

用真实提示词文件（桌面那份 5128 字符的英文长提示词）。
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

from PIL import Image  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402
from genimg.sizing import resolve as resolve_size  # noqa: E402

PROJECT = _ROOT
PROMPT_FILE = r"C:\Users\PPXW\Desktop\水彩上下crad提示词.txt"
PHOTO = r"C:\Users\PPXW\Desktop\1.jpg"
OUT = os.path.join(PROJECT, "results", "poster_ratio_exp")

prompt = open(PROMPT_FILE, encoding="utf-8").read().strip()
print(f"提示词 {len(prompt)} 字符 / {len(prompt.splitlines())} 行")
print(f"参考图 {PHOTO} → {Image.open(PHOTO).size}")

os.makedirs(OUT, exist_ok=True)

# A：复现旧行为 —— 先把图硬拉伸成 512×512，再原样送入
stretched = os.path.join(OUT, "init_stretched_512.png")
Image.open(PHOTO).convert("RGB").resize((512, 512), Image.LANCZOS).save(stretched)

engine = Engine(project_root=PROJECT, verbose=False)

RUNS = [
    ("A_旧拉伸_512", "sd15", "balanced", 512, 512, 0.55, stretched),
    ("B_新填充_sd15_3x4", "sd15", "balanced", 384, 512, 0.55, PHOTO),
    ("C_新填充_sdxl_3x4", "sdxl", "balanced", 768, 1024, 0.55, PHOTO),
]

rows = []
_loaded = [None]
for name, model_key, gear, w, h, strength, img in RUNS:
    prof = resolve_model(model_key)
    # 切模型前必须卸载：两条管线同时驻留会逼出顺序卸载，实测 SDXL 会从 15s 掉到 60s+
    if _loaded[0] is not None and _loaded[0] != prof.key:
        engine.unload()
    _loaded[0] = prof.key
    prec = resolve_precision("bf16" if model_key != "sdxl" else "fp16")
    q = resolve_quality(gear, prof)
    decision = resolve_size(
        prompt=prompt, model_key=model_key, init_size=Image.open(img).size, model_native=prof.native_size
    )
    # A 组用固定 512×512 覆盖自动判断，复现旧行为
    width = w or decision.width
    height = h or decision.height

    req = GenRequest(
        mode="img2img",
        prompt=prompt,
        model=prof.key,
        precision=prec.key,
        gear=q.key,
        width=width,
        height=height,
        seed=20261020,
        strength=strength,
        init_image=img,
        out_dir=os.path.join(OUT, name),
        size_source="实验",
    )
    t0 = time.perf_counter()
    res = engine.run(req, prof, prec, q)
    dt = time.perf_counter() - t0
    if not res.ok:
        print(f"  {name}: 失败 {res.error}")
        continue
    meta = res.meta
    print(
        f"  {name:<20} {width}x{height}  总 {dt:5.2f}s（推理 {res.seconds:.2f}s）  "
        f"init_fit={meta.get('init_fit', '')}",
        flush=True,
    )
    rows.append((name, res.images[0], f"{width}x{height}"))

print("\n输出目录:", OUT)
for r in rows:
    print("  ", r[0], "->", os.path.relpath(r[1], PROJECT))

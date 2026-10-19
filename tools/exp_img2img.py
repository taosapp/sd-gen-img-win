"""实验：图生图做"水彩风格"改造，为什么效果差？对照五个变量。

变量：提示词语言（中/英）、画质挡位（turbo=CFG 0 / 常规 CFG 7 / lightning CFG 1）、
改动强度（0.7 / 0.8）。输出到 results/img2img_exp/<tag>/。

重点验证一个容易被忽略的换算：diffusers 的图生图实际采样步数 = int(步数 × 强度)。
所以 sd-turbo 的 turbo 档（2 步）+ 强度 0.7 → 实际只跑 1 步，风格改造必然只是糊一层色。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "img2img_exp")
INIT = os.path.join(OUT, "input.png")

CN = "把图片改为水彩风格，细节丰富"
EN = "watercolor painting style, soft translucent washes, rich details, paper texture, artistic"

CASES = [
    # tag, 模型, 画质挡位, 强度, 提示词
    ("A_cn_sd15_bal_s070", "sd15", "balanced", 0.70, CN),
    ("B_en_sd15_bal_s070", "sd15", "balanced", 0.70, EN),
    ("C_en_turbo_s070", "sd-turbo", "turbo", 0.70, EN),
    ("D_en_sdxl_lightning_s070", "sdxl", "lightning", 0.70, EN),
    ("E_en_sdxl_bal_s070", "sdxl", "balanced", 0.70, EN),
    ("F_en_sd15_bal_s080", "sd15", "balanced", 0.80, EN),
]

engine = Engine(PROJECT, verbose=False)

print()
print(f"输入图：{os.path.relpath(INIT, PROJECT)}")
print(f"{'用例':<26}{'实际采样步':>10}{'耗时':>8}  说明")
print("-" * 96)

for tag, model_key, gear, strength, prompt in CASES:
    profile = resolve_model(model_key)
    prec = resolve_precision("bf16")
    q = resolve_quality(gear, profile)
    effective = int(q.steps * strength)
    req = GenRequest(
        mode="img2img",
        prompt=prompt,
        negative_prompt="blurry, low quality, deformed",
        model=profile.key,
        precision="bf16",
        gear=q.key,
        width=512,
        height=512,
        seed=2026,
        strength=strength,
        init_image=INIT,
        out_dir=os.path.join(OUT, tag),
    )
    res = engine.run(req, profile, prec, q)
    if not res.ok:
        print(f"{tag:<26}{'-':>10}{'-':>8}  失败：{res.error[:60]}")
        continue
    note = ""
    if q.guidance <= 1.0:
        note = "CFG≤1，提示词约束弱"
    if effective < 3:
        note = (note + "；" if note else "") + "实际采样步 < 3，几乎只是重绘一层色"
    print(f"{tag:<26}{effective:>10}{res.seconds:>7.2f}s  {note}")

print()
print(f"结果目录：{os.path.relpath(OUT, PROJECT)}")

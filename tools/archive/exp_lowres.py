"""实验：SDXL 在 512x512 下构图崩坏，能否用"原生分辨率条件化"修好？

背景
----
SDXL 原生训练分辨率是 1024x1024，它把 `original_size / crops_coord_top_left / target_size`
作为 micro-conditioning 输入。在 512 出图时，diffusers 默认把这些值填成 (512, 512)，
等于告诉模型"这是一张 512 的图"，与训练分布不符，容易画出多格拼贴式的怪异构图。

做法
----
同一提示词、同一种子，跑四组对比：
  A) SDXL @512，默认条件化
  B) SDXL @512，原生分辨率条件化（original_size=target_size=(1024,1024)）
  C) SD1.5 @512（原生 512 参照）
  D) SD-Turbo @512（原生 512，少步）

统一走 engine.run 而不是直接调 pipe，这样 VAE 精度/latent 转型等处理与正式生图一致。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = _ROOT
OUT = os.path.join(PROJECT, "results", "lowres_exp")
os.makedirs(OUT, exist_ok=True)

PROMPT = "a red panda astronaut floating in space, cinematic lighting, highly detailed"
NEG = "blurry, low quality, watermark"
SEED = 1234
SIZE = 512

engine = Engine(PROJECT, verbose=True)


def run(model_key, tag, native_cond=False, gear=None, note=""):
    profile = resolve_model(model_key)
    prec = resolve_precision("fp16")
    q = resolve_quality(gear or profile.default_gear, profile)
    extra = {}
    if native_cond:
        # SDXL micro-conditioning：告诉模型"这是 1024 尺度的图"
        extra = {"original_size": (1024, 1024), "target_size": (1024, 1024), "crops_coord_top_left": (0, 0)}

    req = GenRequest(
        mode="txt2img",
        prompt=PROMPT,
        negative_prompt=NEG,
        model=profile.key,
        precision="fp16",
        gear=q.key,
        width=SIZE,
        height=SIZE,
        seed=SEED,
        out_dir=os.path.join(OUT, tag),
        extra_call_kwargs=extra,
    )
    res = engine.run(req, profile, prec, q)
    if not res.ok:
        print(f"  {tag:34s} 失败：{res.error}")
        return None
    print(f"  {tag:34s} {res.seconds:6.2f}s -> {os.path.relpath(res.images[0], PROJECT)}  {note}")
    return res.images[0]


print()
print("== A) SDXL @512 默认条件化 ==")
run("sdxl", "A_sdxl512_default_cond", note="（预期：可能出现拼接/分格）")

print("== B) SDXL @512 原生分辨率条件化 ==")
run("sdxl", "B_sdxl512_native_cond", native_cond=True, note="（对比是否消除拼接）")

print("== C) SD1.5 @512（原生 512 参照） ==")
run("sd15", "C_sd15_512_native")

print("== D) SD-Turbo @512（原生 512，少步） ==")
run("sd-turbo", "D_sdturbo_512_turbo", gear="turbo")

print()
print(f"对比图已写入 {OUT}")

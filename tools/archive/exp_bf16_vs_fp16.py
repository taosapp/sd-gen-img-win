"""bf16 vs fp16：同种子同提示词，比较速度与像素差异。

只跑 sd-turbo（2 步，最省时），外加 sd15 的 balanced 档做二次确认。
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = _ROOT
OUT = os.path.join(PROJECT, "results", "bf16_vs_fp16")
PROMPT = "a panda running on a green grassland, sunny day"
SEED = 20261022
REPEAT = 5


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def run(engine, model_key, prec_key, gear_key, size):
    prof = resolve_model(model_key)
    prec = resolve_precision(prec_key)
    q = resolve_quality(gear_key, prof)
    req = GenRequest(
        mode="txt2img",
        prompt=PROMPT,
        model=prof.key,
        precision=prec.key,
        gear=q.key,
        width=size,
        height=size,
        seed=SEED,
        out_dir=os.path.join(OUT, f"{model_key}_{prec_key}"),
        style_expand="off",
    )
    return engine.run(req, prof, prec, q)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    engine = Engine(project_root=PROJECT, verbose=False)

    cases = [("sd-turbo", "turbo", 512), ("sd15", "balanced", 512)]
    print(f"{'模型':<10}{'精度':<7}{'首次':>8}{'热态均值':>10}{'热态最快':>10}", flush=True)
    print("-" * 48)

    images = {}
    for model_key, gear_key, size in cases:
        for prec_key in ("fp16", "bf16"):
            # 第一次含权重加载，单独记
            first = run(engine, model_key, prec_key, gear_key, size)
            if not first.ok:
                print(f"{model_key}/{prec_key} 失败: {first.error}")
                continue
            first_s = float(first.seconds or 0)
            hot = []
            path = first.images[0]
            for _ in range(REPEAT):
                res = run(engine, model_key, prec_key, gear_key, size)
                if res.ok:
                    hot.append(float(res.seconds or 0))
                    path = res.images[0]
            images[(model_key, prec_key)] = load(path)
            print(
                f"{model_key:<10}{prec_key:<7}{first_s:>7.2f}s"
                f"{np.mean(hot):>9.2f}s{min(hot):>9.2f}s",
                flush=True,
            )

    print()
    for model_key, _, _ in cases:
        a = images.get((model_key, "fp16"))
        b = images.get((model_key, "bf16"))
        if a is None or b is None or a.shape != b.shape:
            continue
        diff = float(np.mean(np.abs(a - b)))
        # 结构相似：用灰度图的相关系数
        ga = a.mean(axis=2)
        gb = b.mean(axis=2)
        corr = float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1])
        print(
            f"{model_key}: bf16 vs fp16 平均像素差 {diff:.2f}/255"
            f" · PSNR {psnr(a, b):.2f} dB · 灰度相关 {corr:.5f}"
        )

    print()
    print("图在:", os.path.relpath(OUT, PROJECT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

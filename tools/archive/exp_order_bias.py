"""顺序对照：先 fp16 还是先 bf16，看热态耗时是否随顺序变化。"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = _ROOT
OUT = os.path.join(PROJECT, "results", "bf16_vs_fp16")
PROMPT = "a panda running on a green grassland, sunny day"
SEED = 20261022
REPEAT = 8


def main() -> int:
    order = sys.argv[1].split(",") if len(sys.argv) > 1 else ["bf16", "fp16"]
    engine = Engine(project_root=PROJECT, verbose=False)
    prof = resolve_model("sd-turbo")
    q = resolve_quality("turbo", prof)

    print("顺序:", " → ".join(order), flush=True)
    for prec_key in order:
        prec = resolve_precision(prec_key)
        req = GenRequest(
            mode="txt2img",
            prompt=PROMPT,
            model=prof.key,
            precision=prec.key,
            gear=q.key,
            width=512,
            height=512,
            seed=SEED,
            out_dir=os.path.join(OUT, f"order_{prec_key}"),
            style_expand="off",
        )
        times = []
        for _ in range(REPEAT + 1):
            res = engine.run(req, prof, prec, q)
            if res.ok:
                times.append(float(res.seconds or 0))
        hot = times[1:]  # 丢掉同模型第一张（含预热）
        print(
            f"{prec_key:<6} 中位 {np.median(hot):.3f}s  均值 {np.mean(hot):.3f}s  "
            f"最快 {min(hot):.3f}s  最慢 {max(hot):.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

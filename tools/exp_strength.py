"""强度对照：同一张图、同一提示词，只改 strength，看主体保留度与风格强度的平衡。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genimg.engine import Engine, GenRequest  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results", "img2img_exp")
INIT = os.path.join(OUT, "input.png")
PROMPT = "watercolor painting style, soft translucent washes, rich details, paper texture, artistic"

engine = Engine(PROJECT, verbose=False)
profile = resolve_model("sd15")
prec = resolve_precision("bf16")
q = resolve_quality("balanced", profile)

for strength in (0.5, 0.6):
    tag = f"G_en_sd15_bal_s{int(strength*100):03d}"
    res = engine.run(
        GenRequest(
            mode="img2img",
            prompt=PROMPT,
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
        ),
        profile,
        prec,
        q,
    )
    print(f"{tag}: {'%.2fs' % res.seconds if res.ok else res.error[:80]}  实际采样 {int(q.steps*strength)} 步")

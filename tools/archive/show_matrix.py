"""列出可跑矩阵：模型 × 精度 × 画质挡位。"""

import sys

sys.path.insert(0, r"D:\ai\gen-img-agent")

from genimg.presets import (  # noqa: E402
    MODEL_PROFILES,
    PRECISION_GEARS,
    check_precision_support,
    probe_hardware,
    quality_gears_for,
)

hw = probe_hardware()
total = 0
print("模型        精度            画质挡位                                        张数")
print("-" * 96)
for key, prof in MODEL_PROFILES.items():
    gears = quality_gears_for(prof)
    ok = [p.key for p in PRECISION_GEARS.values() if not check_precision_support(p, hw)]
    n = len(ok) * len(gears)
    total += n
    print(f"{key:<10}  {','.join(ok):<15} {','.join(gears):<46} {n}")
print("-" * 96)
print(f"合计 {total} 张；两组提示词 = {total * 2} 张")

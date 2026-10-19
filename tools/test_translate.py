"""中文提示词词典翻译的自检（纯离线，无外部依赖）。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genimg.zh_prompt import translate  # noqa: E402

CASES = [
    "把图片改为水彩风格，细节丰富",
    "水彩风格，细节丰富，柔和光线，留白",
    "油画风格，暖色调，电影级灯光，8k，大师作品",
    "水墨山水，留白，远山，淡墨",
    "赛博朋克风格，霓虹灯，夜景，高对比",
    "一只戴墨镜的熊猫在月球上喝咖啡，赛博朋克风格",  # 主体词词典覆盖不到
    "watercolor painting style, rich details",  # 纯英文：不应改动
]

print(f"{'原始提示词':<44}{'方式':<6}{'效果':<60}")
print("-" * 118)
for text in CASES:
    r = translate(text)
    print(f"{text:<44}{r.method:<6}{r.effective[:60]:<60}")
    if r.unknown:
        print(f"{'':<44}未识别（已原样保留）：{'、'.join(r.unknown)}")

print()
print("== mode=off 时不翻译 ==")
r = translate("把图片改为水彩风格", mode="off")
print(f"  方式={r.method}  效果={r.effective}")

print()
print("== 词典未命中时保留原文，不应返回空串 ==")
r = translate("一只猫")
print(f"  方式={r.method}  效果={r.effective}")

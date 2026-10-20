"""尺寸解析自检：比例 → 尺寸桶、参考图跟随、填充不变形。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from genimg.sizing import (  # noqa: E402
    fit_image,
    parse_spec,
    ratio_from_prompt,
    resolve,
    size_for_ratio,
)

print("=== parse_spec ===")
for s in ["3:4", "768x1024", "match", "16:9", "乱七八糟"]:
    print(f"  {s:<12} -> {parse_spec(s)}")

print("\n=== size_for_ratio（比例优先，尽量贴近原生面积）===")
for ratio in [(1, 1), (3, 4), (4, 3), (2, 3), (16, 9)]:
    a = size_for_ratio(*ratio, 512)
    b = size_for_ratio(*ratio, 1024)
    print(f"  {ratio[0]}:{ratio[1]:<3} sd15 {a[0]}x{a[1]}={a[0] / a[1]:.3f}   sdxl {b[0]}x{b[1]}={b[0] / b[1]:.3f}")

print("\n=== ratio_from_prompt ===")
cases = [
    "Strict 3:4 vertical composition. Divide the canvas ... precise 1:1 height ratio",
    "a cat on a sofa, cinematic",
    "竖版海报，水彩",
    "no ratio here at all",
]
for c in cases:
    print(f"  {c[:46]:<48} -> {ratio_from_prompt(c)}")

print("\n=== resolve（优先级） ===")
print("  --size 3:4 + sd15          :", resolve(spec="3:4", model_key="sd15").describe())
print("  --size 3:4 + sdxl          :", resolve(spec="3:4", model_key="sdxl").describe())
print("  --size 768x1024            :", resolve(spec="768x1024", model_key="sd15").describe())
print("  img2img 无参数(1024x683)    :", resolve(model_key="sd15", init_size=(1024, 683)).describe())
print("  提示词 3:4 + 参考图 3:2     :", resolve(prompt="Strict 3:4 vertical", model_key="sdxl", init_size=(1024, 683)).describe())
print("  只给 --width 900           :", resolve(width=900, model_key="sdxl").describe())
print("  超面积 1536x1536           :", resolve(spec="1536x1536", model_key="sd15").describe())

print("\n=== fit_image：横图 → 竖画布，检查是否变形 ===")
src = Image.new("RGB", (1024, 683), (40, 60, 90))
# 画一个圆，若被拉伸会变成椭圆
from PIL import ImageDraw  # noqa: E402

d = ImageDraw.Draw(src)
d.ellipse((412, 241, 612, 441), fill=(220, 80, 60))  # 200x200 正圆
out, note = fit_image(src, 768, 1024)
print(f"  输出尺寸: {out.size}  说明: {note}")

# 检测填充区：中间那行像素应仍是深蓝（模糊背景），画面主体区应保持圆形
px = out.load()
print(f"  左上角像素: {px[5, 5]}（模糊延展背景）")
print(f"  中心像素  : {px[384, 512]}（主体区）")

# 直接把原图等比缩进画布再比对：圆心应落在画布中央
print("\n  说明：填充模式不会拉伸像素；若用旧的 resize((w,h)) 处理，正圆会变成 0.75 宽的椭圆")

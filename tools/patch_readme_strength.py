"""把 README 里的图生图强度默认值统一改为 0.5（保主体）。"""

import io
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
text = io.open(PATH, encoding="utf-8", newline="").read()

pairs = [
    ("powershell -File run.ps1 --image .\\in.png --strength 0.6 \"oil painting style\"",
     "powershell -File run.ps1 --image .\\in.png --strength 0.5 \"oil painting style\""),
    ("| 改动强度 | `[强度=0.65]` | 0–1（图生图） |",
     "| 改动强度 | `[强度=0.5]` | 0–1（图生图，默认 0.5 保主体） |"),
    ("   改动强度拖到 `0.65` → 生成。预期：构图保留、风格改变。",
     "   改动强度保持默认 `0.5` → 生成。预期：构图与主体保留、风格改变。"),
    ("| 改动强度 | 0.45–0.55 保主体；0.6–0.7 强风格化 | 见下表 |",
     "| 改动强度 | **0.5（默认）** 保主体；0.65–0.7 强风格化 | 见下表 |"),
    ("| 0.50 | 12 步 | 主体完整，呈现「淡彩水彩」效果 |",
     "| **0.50（默认）** | 12 步 | 主体完整，呈现「淡彩水彩」效果 |"),
]

changed = 0
for old, new in pairs:
    n = text.count(old)
    if n != 1:
        print(f"[跳过] 匹配 {n} 次：{old[:40]}")
        continue
    text = text.replace(old, new)
    changed += 1

io.open(PATH, "w", encoding="utf-8", newline="").write(text)
print(f"已替换 {changed}/{len(pairs)} 处")

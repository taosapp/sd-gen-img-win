"""把 §6.9 两段式海报一节插入 README（放在 §7 实测数据之前）。"""

import io
import os

PATH = r"D:\ai\gen-img-agent\README.md"
lines = io.open(PATH, encoding="utf-8").read().split("\n")
idx = next(i for i, ln in enumerate(lines) if ln.startswith("## 7. 实测数据"))

section = """### 6.9 两段式海报：上半照片 + 下半插画（严格 50/50）

§6.8 的结论是：**版式指令进不了 CLIP**（835 token → 只吃 77 个）。所以版式交给代码，
模型只负责「画什么」：

```
第 1 步  照片等比裁切（cover）填满上半区 —— 只裁不拉
第 2 步  同一张裁切图做 img2img（强度 0.70）—— 只让模型把照片变手绘插画
第 3 步  插画缩到 62% 贴到纸色底版 —— 「主体小、大量留白」由代码保证
第 4 步  上下等高直接拼，缝在正中间
```

**命令**

```powershell
python tools\\make_poster.py `
  --image "C:\\Users\\PPXW\\Desktop\\1.jpg" "C:\\Users\\PPXW\\Desktop\\3.jpg" `
  --prompt-file "C:\\Users\\PPXW\\Desktop\\水彩上下crad提示词.txt" `
  --model sdxl --gear balanced --size 3:4 --strength 0.72 --art-scale 0.62 `
  --out results\\poster_two_pass
```

**长提示词怎么压缩**：自动从长文本里抽两段——

- 正向：`BOTTOM HALF` 小节的首句 + `Use:` 列表（手绘线条 / 丙烯平涂 / 纸纹…），截到 50 词 ≈ 65 token；
- 负向：`AVOID` 小节按行转成负向词（crayon / bleeding watercolor / 3d render…）。

抽不到的段落（上半照片的保真要求）本来就由第 1 步的裁切保证了，不需要模型理解。

**实测（SDXL fp16 / balanced / 768×1024，每半 768×512）**

| 项 | 数值 |
|---|---|
| 第一张（含加载） | 18.0 s |
| 后续每张 | 6.6–6.9 s |
| 版式 | 精确 50/50，可验证（像素行数各 512） |

**强度取舍（同一张照片、同一种子）**

| 强度 | 效果 |
|---|---|
| **0.70** | 主体仍可辨认：帽子、手势、姿态都在，转成手绘+平涂风格 —— **默认** |
| 0.85 | 完全抽象成色块，主体丢失 |

**已知局限（如实记录）**

- 风格偏「高对比版画」而不是文件名的「水彩」——提示词本身写的是手绘纸本、并明确 `AVOID bleeding watercolor`；
- 插画自带的背景色（如熊猫那张的绿）不会被换成纸色；
- 没有文字排版（`TYPOGRAPHY` 小节尚未使用）；
- 下半是从照片 img2img 来的，因此构图跟随原图，只靠缩小制造留白。

## 7. 实测数据（512×512）"""

lines[idx] = section
io.open(PATH, "w", encoding="utf-8", newline="").write("\n".join(lines))
print(f"已插入 §6.9，共 {len(lines)} 行")

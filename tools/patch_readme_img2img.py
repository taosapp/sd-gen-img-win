"""在 README 的 §6 末尾插入「图生图风格改造」章节。"""

import io
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
lines = io.open(PATH, encoding="utf-8", newline="").read().split("\n")

anchor = next(i for i, ln in enumerate(lines) if ln.startswith("## 7. 实测数据"))

section = """### 6.6 图生图做「风格改造」的正确姿势（实测）

**症状**：上传一张图，提示词写「把图片改为水彩风格，细节丰富」，用 `sd-turbo` + `turbo` 档，
结果只是色块涂抹，完全不像水彩。

**三个原因**（都做过对照验证）：

1. **图生图的实际采样步数 = `int(步数 × 强度)`**，不是步数本身。
   `turbo` 档只有 2 步，强度 0.6 → `int(1.2)` = **1 步**。1 步只够把原图重涂一层，做不出风格。
   `lightning`（4 步）× 0.7 = 2 步，同样偏弱。
2. **`turbo` / `lightning` 的 CFG ≤ 1**，等于关闭引导：提示词与负向提示词几乎不起作用。
3. **中文提示词效果明显弱于英文**：SD 系列的 CLIP 文本编码器以英文语料为主。
   同参数、同种子下，中文「水彩风格」出的是糊在一起的色块；英文
   `watercolor painting style` 才出纸质边缘与湿画法晕染（对照图 `results/img2img_exp/contact_sheet.png`）。

**推荐配方**：

| 项目 | 建议值 | 说明 |
|---|---|---|
| 模型 | `sd15` | 原生 512，图生图最稳（`sdxl` 也行，但建议出 1024） |
| 画质挡位 | `balanced`（25 步 / CFG 7） | 有 CFG，提示词才起作用 |
| 改动强度 | 0.45–0.55 保主体；0.6–0.7 强风格化 | 见下表 |
| 提示词 | 英文 | `watercolor painting style, soft translucent washes, rich details, paper texture` |
| 负向提示词 | `blurry, low quality, deformed` | CFG>1 时才生效 |

**强度对照（同一张图、同一提示词、sd15 / balanced）**：

| 强度 | 实际采样步数 | 效果 |
|---|---|---|
| 0.50 | 12 步 | 主体完整，呈现「淡彩水彩」效果 |
| 0.60 | 15 步 | 主体开始变形（本例脸部和人像混淆） |
| 0.70 | 17 步 | 完全艺术化重画，主体基本丢失（本例变成花卉） |

**现在会自动提示**：命令行与网页都会在生成前检查这几个坑（规则在 `genimg/advice.py`），
网页端还多了一个「用风格改造推荐配置」按钮，一键切到 `sd15 / balanced / 0.7`。
若图生图的实际步数算出来是 0（步数×强度 < 1），会直接报错并说明原因，而不是丢一个底层异常。

"""
lines[anchor:anchor] = section.split("\n")
io.open(PATH, "w", encoding="utf-8", newline="").write("\n".join(lines))
print(f"已插入 §6.6，共 {len(lines)} 行")

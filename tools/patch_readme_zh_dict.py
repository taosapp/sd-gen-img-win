"""把 README 的 §6.7 改写为「纯离线词典」版本（移除本地大模型翻译相关内容）。"""

import io
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
lines = io.open(PATH, encoding="utf-8", newline="").read().split("\n")

start = next(i for i, ln in enumerate(lines) if ln.startswith("### 6.7"))
end = next(i for i, ln in enumerate(lines) if ln.startswith("## 7."))

new_section = """### 6.7 中文提示词支持（纯离线词典）

**现实**：`sd15` / `sdxl` / `sd-turbo` 的文本编码器都是 CLIP（英文语料为主），中文提示词过完分词
大多是无效 token。同一组参数、同一种子，中文写「水彩风格」得到的是糊在一起的色块，
英文写 `watercolor painting style` 才出纸质感。

**做法**：内置一个**纯离线词典**（116 条），把风格 / 画质 / 光影 / 构图 / 颜色 / 常见场景
翻成英文，并剔除「把图片改为」「请帮我」这类对 CLIP 无意义的指令性措辞。
**不联网、不依赖任何外部服务、不涉及任何凭据**，零延迟、结果可预期。

**翻译效果（可复现：`python tools\\test_translate.py`）**

| 输入 | 实际使用的提示词 |
|---|---|
| 把图片改为水彩风格，细节丰富 | `watercolor painting style, rich details` |
| 油画风格，暖色调，电影级灯光，8k，大师作品 | `oil painting style, warm color tones, cinematic lighting, 8k, masterpiece` |
| 水墨山水，留白，远山，淡墨 | `ink wash painting, landscape, negative space, distant mountains, light ink wash` |
| 赛博朋克风格，霓虹灯，夜景，高对比 | `cyberpunk style, neon lights, night scene, high contrast` |

验证方式：中文「把图片改为水彩风格，细节丰富」翻译后与**直接写这句英文**，
在同参数、同种子下出图 —— **像素完全一致**（SHA256 相同，最大像素差 0）。

**词典覆盖不到的部分不会丢**：例如「一只戴墨镜的熊猫在月球上喝咖啡，赛博朋克风格」会变成
`一只戴墨镜的熊猫在月球上喝咖啡, cyberpunk style`，并提示「未识别片段已原样保留」。

这类自由描述（主体、动作、关系）本就不是词典能解决的。真要覆盖，正确做法是换**中文原生底模**
（Taiyi / AltDiffusion 等，模型本身懂中文）——代价是额外约 4GB 下载，且画质低于现有的
sd-turbo / sd15 / SDXL。本项目**未集成**，也不打算在提示词层做机器翻译。

**开关**：默认开启；命令行 `--translate off` 可关掉，中文原样送进模型。

**透明性**：实际使用的英文提示词会打印在命令行、写进 PNG 元数据
（`prompt` = 实际使用值，`prompt_original` = 你输入的原文），网页端也会显示
「中文已转英文（dict）：…」以及未翻译片段，不用猜它拿什么去生图。
"""

lines[start:end] = new_section.split("\n")
io.open(PATH, "w", encoding="utf-8", newline="").write("\n".join(lines))
print(f"已替换 §6.7（原 {end - start} 行 → 新 {len(new_section.splitlines())} 行），共 {len(lines)} 行")

"""在 README 的 §6.6 之后插入「中文提示词支持」章节。"""

import io
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
lines = io.open(PATH, encoding="utf-8", newline="").read().split("\n")
anchor = next(i for i, ln in enumerate(lines) if ln.startswith("## 7. 实测数据"))

section = """### 6.7 中文提示词支持

**现实**：`sd15` / `sdxl` / `sd-turbo` 的文本编码器都是 CLIP（英文语料为主），中文提示词过完分词
大多是无效 token。同一组参数、同一种子，中文写「水彩风格」得到的是糊在一起的色块，
英文写 `watercolor painting style` 才出纸质感。

所以这件事分三层做，先做不花钱的那层：

| 层 | 能力 | 额外依赖 | 状态 |
|---|---|---|---|
| ① 离线词典翻译 | 风格 / 画质 / 光影 / 构图 / 颜色 / 常见场景等一百多个中文词 → 英文 | 无 | **默认启用** |
| ② 本地大模型翻译 | 任意中文描述（含主体、动作、场景） | LM Studio / Ollama / vLLM 等 OpenAI 兼容端点 | 已实现，配环境变量即用 |
| ③ 中文原生底模 | 模型本身懂中文（Taiyi / AltDiffusion 等） | 需下载约 4GB，画质低于现有模型 | 未集成，列为备选 |

**① 的效果（可复现）**

| 输入 | 实际使用的提示词 |
|---|---|
| 把图片改为水彩风格，细节丰富 | `watercolor painting style, rich details` |
| 水彩风格，细节丰富，柔和光线，留白 | `watercolor painting style, rich details, soft lighting, negative space` |
| 油画风格，暖色调，电影级灯光，8k，大师作品 | `oil painting, style, warm color tones, cinematic lighting, 8k, masterpiece` |

验证方式：中文「把图片改为水彩风格，细节丰富」翻译后与**直接写这句英文**，
在同参数、同种子下出图 —— **像素完全一致**（SHA256 相同，最大像素差 0）。
也就是说"说中文"和"说英文"在这套流程里是同一件事。

**词典覆盖不到的部分不会丢**：例如
「一只戴墨镜的熊猫在月球上喝咖啡，赛博朋克风格」会变成
`一只戴墨镜的熊猫在月球上喝咖啡, cyberpunk, style`，
并明确提示"未识别片段已原样保留"。想覆盖这类自由描述，请启用第 ② 层。

**② 启用本地大模型翻译**（LM Studio 演示，先在 LM Studio 里加载模型并打开本地服务器）

```powershell
$env:GENIMG_TRANSLATE = "llm"
$env:GENIMG_TRANSLATE_URL = "http://127.0.0.1:1234/v1"
$env:GENIMG_TRANSLATE_MODEL = "qwen2.5-7b-instruct"   # 改成你实际加载的模型名
powershell -File web.ps1
```

命令行也可以临时指定：`--translate llm --translate-url http://127.0.0.1:1234/v1 --translate-model <名字>`；
不想翻译就 `--translate off`。若端点不可用，会自动回退到词典并在结果里注明，不会让生成失败。

**透明性**：实际使用的英文提示词会打印在命令行、写进 PNG 元数据
（`prompt` = 实际使用值，`prompt_original` = 你输入的原文），网页端也会显示
「中文已转英文（dict）：…」以及未翻译片段，不用猜它到底拿什么去生图。

"""
lines[anchor:anchor] = section.split("\n")
io.open(PATH, "w", encoding="utf-8", newline="").write("\n".join(lines))
print(f"已插入 §6.7，共 {len(lines)} 行")

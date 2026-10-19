"""按行号/锚点更新 README（避开控制台乱码导致的字符串匹配问题）。"""

import io
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
lines = io.open(PATH, encoding="utf-8", newline="").read().split("\n")

NEW_TABLE = [
    "| 能跑吗 | 能。RTX 3060 12GB 足以在本机离线生图，无需联网推理 |",
    "| 512×512 快吗 | 极快。默认挡位（sd-turbo / bf16 / turbo）**0.34 秒**/张；SDXL fp16 约 5.1 秒/张 |",
    "| 30 秒预算 | 18 个配置中 15 个满足，最快有 **88 倍余量**；仅「SDXL + fp32/tf32」超预算（显存硬上限） |",
    "| 512 该用哪个模型 | **默认已改为 sd-turbo**（原生 512、主体最完整且最快）；需负向提示词用 `sd15`；要 SDXL 画质就出 1024（§7.3 有对照实验） |",
    "| 精度挡位 | **`bf16`（默认）** / `fp16` / `tf32` / `fp32`；`fp8` 本机硬件不支持，会明确报错。实测 bf16 在 sd-turbo 上比 fp16 快 25%（§7.2） |",
    "| 怎么指定挡位 | 命令行 `--precision fp16 --gear balanced`，**或直接在提示词里写** `[精度=fp16] [画质=max] [尺寸=640x640]` |",
    "| 有可视化界面吗 | 有。`web.ps1` 启动本地网页，支持**上传图片**与**文生图 / 图生图**切换，见 §6.5 |",
]

# 1) 替换结论速览表的数据行（第 10–16 行，1-based）
start = next(i for i, ln in enumerate(lines) if ln.startswith("| 能跑吗"))
end = next(i for i, ln in enumerate(lines) if ln.startswith("| 有可视化界面吗"))
lines[start : end + 1] = NEW_TABLE

# 2) 替换末尾"下次继续"章节
anchor = next((i for i, ln in enumerate(lines) if ln.startswith("## 11.")), None)
if anchor is not None:
    new_tail = [
        "## 11. 后续可选（非阻塞）",
        "",
        "本期目标（硬件评估 / 选型 / 512×512 30 秒内出图 / 多精度挡位 / 可上传的 Web 界面）已完成，",
        "以下是可继续深挖的方向，按性价比排序：",
        "",
        "1. **1024 出图调优**：SDXL 原生 1024 才是它的主场；目前只测了 512，可补一组 1024 的挡位矩阵",
        "   （显存占用会明显上升，fp16 下预计 9–11GB，可能需要 VAE tiling）。",
        "2. **SD1.5 微调模型/中文提示词适配**：默认模型对中文提示词支持一般，可挂中文 CLIP 或换社区精调模型。",
        "3. **图生图的局部重绘（inpaint）**：当前 img2img 是整图重绘，加 mask 可做局部修改。",
        "4. **量化挡位**：Windows 下 `torchao` / `bitsandbytes` 若可用，可再压显存或提速（当前未验证）。",
        "5. **提示词模板库**：把常用风格（海报 / 封面 / 产品图）沉淀成预设，界面上一键切换。",
    ]
    lines = lines[:anchor] + new_tail

io.open(PATH, "w", encoding="utf-8", newline="").write("\n".join(lines))
print(f"已更新 {PATH}，共 {len(lines)} 行")

"""更新 README 中与 Web 启动方式相关的段落（改为双击 web.bat + 预热说明）。"""

import io
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
lines = io.open(PATH, encoding="utf-8", newline="").read().split("\n")


def find(pred, desc):
    for i, ln in enumerate(lines):
        if pred(ln):
            return i
    raise SystemExit(f"没找到：{desc}")


# 1) 结论速览里的“有可视化界面吗”
i = find(lambda ln: ln.startswith("| 有可视化界面吗"), "结论表界面行")
lines[i] = "| 有可视化界面吗 | 有。**双击 `web.bat`** 即可（自动预热模型并打开浏览器）；支持**上传图片**与**文生图 / 图生图**切换，见 §6.5 |"

# 2) §6.5 的启动命令块
i = find(lambda ln: "web.ps1                    # http://127.0.0.1:7860" in ln, "启动命令块")
lines[i : i + 2] = [
    ":: 最简单：双击 web.bat —— 后端 + 界面一起启动，自动预热模型并打开浏览器",
    "",
    "# 也可以用命令行（等价，便于调参数）",
    "powershell -ExecutionPolicy Bypass -File web.ps1                    # 预热默认挡位",
    "powershell -ExecutionPolicy Bypass -File web.ps1 -Port 8080 -Open    # 换端口 + 自动开浏览器",
    "powershell -ExecutionPolicy Bypass -File web.ps1 -Preload:$false     # 不预热",
]

# 3) 手动测试步骤第 1 条
i = find(lambda ln: ln.startswith("1. 启动："), "手动测试步骤")
lines[i] = "1. 启动：**双击 `web.bat`**（或 `powershell -File web.ps1 -Open`），控制台会打印实际地址并自动打开浏览器。"

# 4) 目录结构里的 web.ps1
i = find(lambda ln: ln.startswith("├─ web.ps1"), "目录结构 web.ps1")
lines[i : i + 1] = [
    "├─ web.bat                    # 双击启动 Web 界面（推荐）",
    "├─ web.ps1                    # 启动脚本本体（-Port / -Open / -Preload:$false）",
]

# 5) 在端到端实测段后补一段预热说明
i = find(lambda ln: "挡位对照图见" in ln, "端到端实测段")
lines[i + 1 : i + 1] = [
    "",
    "**启动时预热**：`web.bat` 默认带 `--preload`，服务起来后会在后台把默认挡位",
    "（sd-turbo / bf16 / turbo）加载进显存，并空跑一步把首次前向的 kernel 选择也热掉。",
    "实测：预热约 12s 完成，之后**第一次点生成就是 0.32s**；若不预热，第一次生成要等约 10–12s 加载权重。",
    "页面右上会显示“模型已就绪 / 预热中”，想关掉预热就用 `web.ps1 -Preload:$false`。",
]

io.open(PATH, "w", encoding="utf-8", newline="").write("\n".join(lines))
print(f"已更新，共 {len(lines)} 行")

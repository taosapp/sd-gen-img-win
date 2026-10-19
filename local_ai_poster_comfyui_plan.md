# 本地 AI 海报生图应用：选型与实施方案

## 1. 硬件与目标

- **操作系统：** Windows 10/11
- **显卡：** NVIDIA RTX 3060 12GB
- **内存：** 32GB RAM
- **目标：** 生成 512×512 图片，尽量在 30 秒内完成，同时兼顾画质与中英文文字准确度。
- **主要用途：** 海报、封面、产品宣传图。
- **实施原则：** 先使用 ComfyUI 验证模型和工作流，不急于开发 Vue 前端。

> 性能说明：以下是建议的选型和测试方案，并非 RTX 3060 上的实测保证。实际速度受模型精度、采样步数、节点、驱动、CPU 卸载和内存压力影响。

## 2. 推荐选型

### A. 快速预览：SDXL 1.0 + SDXL Lightning

- **基础模型：** SDXL Base 1.0
- **精度：** 优先 FP16
- **加速：** SDXL Lightning 4-step LoRA
- **用途：** 快速探索构图、配色、背景、产品摆放。
- **优点：** 生态成熟，比较适合 12GB 显存。
- **注意：** 4 步不等于 4 秒，具体速度必须实测。按对应 Lightning 示例使用采样器、调度器和 CFG 设置。

官方入口：
- [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)

### B. 中文文字与高质量生成：Qwen-Image-2512

- **主模型候选：** Qwen-Image-2512 FP8 工作流
- **用途：** 中文海报、文字与画面融合、复杂宣传图。
- **加速：** 在官方工作流支持的前提下测试 Lightning 4-step LoRA。
- **主要风险：** Qwen-Image 模型较大，FP8 工作流的组件无法全部驻留在 12GB 显存中，可能需要 CPU 卸载，导致生成变慢并增加系统内存占用。
- **备选：** 如果 FP8 工作流无法稳定运行，可评估可信社区的 GGUF Q4 量化版本及对应加载节点。GGUF 不是直接替换文件，必须匹配模型结构、节点和版本。

官方入口：
- [Qwen-Image 官方仓库](https://github.com/QwenLM/Qwen-Image)
- [Qwen-Image-2512 ComfyUI 官方工作流说明](https://github.com/Comfy-Org/docs/blob/main/tutorials/image/qwen/qwen-image-2512.mdx)

### C. 不建议一开始追求“解除限制版”

优先关注：
- 量化格式与精度
- 显存占用
- 推理速度
- 工作流兼容性
- 模型来源、许可证和配套代码是否可信

第三方修改安全限制并不意味着画质更好、速度更快或文字更准确。不要运行来源不明的安装脚本。

## 3. 安装步骤

### 第一步：安装 ComfyUI

1. 更新 NVIDIA 显卡驱动并重启。
2. 从 [ComfyUI 官方下载页](https://www.comfy.org/download) 下载 Windows 版本，安装时选择 NVIDIA GPU。
3. 建议将程序和模型放在 SSD，例如 `D:\\AI\\ComfyUI`。
4. 建议为模型、缓存和输出预留至少约 100GB 磁盘空间。
5. 启动 ComfyUI，先运行简单示例，确认 GPU 工作正常。

建议先使用官方桌面版，不要一开始就混用系统 Python 3.12、3.13 与 ComfyUI 自己的环境。自定义节点确有需要时，再考虑便携版或独立环境。

### 第二步：安装 SDXL Lightning

1. 下载 SDXL Base 1.0。
2. 下载对应的 SDXL Lightning 4-step LoRA。
3. 将模型放入 ComfyUI 对应模型目录，或按所用版本的模型管理界面说明安装。
4. 使用官方 Lightning 示例工作流。
5. 将输出设为 512×512、Batch Size = 1。
6. 先跑通后，再比较 4 步和 8 步的速度与画质。

建议工作流：

`Checkpoint Loader → LoRA Loader → Prompt / Empty Latent Image → KSampler → VAE Decode → Save Image`

请以对应 Lightning 示例的采样器、调度器和 CFG 设置为准，不要直接照搬普通 SDXL 的参数。

### 第三步：安装 Qwen-Image-2512

先打开官方 ComfyUI 工作流文档，确认文件名和工作流要求。官方工作流所列组件包括：

| 组件 | 常见目录 |
|---|---|
| Qwen-Image-2512 FP8 主模型 | `models/diffusion_models/` |
| Qwen 文本编码器 | `models/text_encoders/` |
| Qwen Image VAE | `models/vae/` |
| Lightning LoRA（如使用） | `models/loras/` |

文件名和要求可能随官方工作流更新，请以当前官方文档为准，不要只下载主模型就假设工作流能运行。

安装后按顺序测试：
1. 更新 ComfyUI 到支持该工作流的版本。
2. 加载官方 Qwen-Image-2512 工作流。
3. 先运行标准工作流，确认所有节点和模型均正常。
4. 再启用工作流支持的加速 LoRA。
5. 先用 512×512 测试，再尝试更高分辨率。
6. 记录显存、系统内存、首次加载时间与后续生成时间。

如果系统内存接近耗尽、Windows 大量使用分页文件，或出现显存溢出，先降低内存压力或改用经过验证的量化工作流。

## 4. 显存与内存规划

下表是粗略规划，不是保证值：

| 项目 | 规划建议 |
|---|---|
| SDXL FP16 + Lightning | 相对适合 12GB 显存；实际峰值依节点与设置变化 |
| Qwen-Image-2512 FP8 | 12GB 显存无法完整容纳所有组件，可能需要 CPU 卸载 |
| Qwen GGUF Q4 | 可能降低权重内存需求，但速度与显存取决于加载器和卸载设置 |
| 32GB 系统内存 | 适合尝试轻量工作流；Qwen FP8 全套组件可能造成明显内存压力 |
| 磁盘 | 建议预留至少约 100GB |

## 5. 海报文字怎样做到容易修改

### 推荐方式：AI 生成背景，程序或编辑器叠加文字

这是文字可控性最好的方法：

1. 让 AI 生成不含文字的背景。
2. 使用 SVG、Canvas 或现成图片编辑软件叠加标题、副标题、按钮和品牌名。
3. 文字独立保存，可修改字体、字号、颜色、位置、对齐、描边和阴影。
4. 导出时将文字与背景合成为 PNG。
5. 如需继续编辑，保存项目 JSON 或 SVG。

示例文字对象：

```json
{
  "type": "text",
  "text": "秋季新品发布",
  "x": 256,
  "y": 110,
  "fontFamily": "Noto Sans SC",
  "fontSize": 32,
  "fontWeight": 700,
  "color": "#FFFFFF",
  "align": "center",
  "rotation": 0
}
```

例如，把“秋季新品发布”改成“冬季新品发布”，只需要修改文本字段，不需要重新生成背景。

### 另一种方式：AI 直接生成完整海报

使用 Qwen-Image-2512，让标题与画面融合。适合透视招牌、装饰字等场景，但 AI 生成的文字不能保证百分之百准确，尤其是小字和复杂排版。

建议将两种方式结合：
- 普通标题、正文、按钮：使用独立文字图层。
- 必须与画面材质或透视融合的文字：尝试 AI 生成或图像编辑模型。
- 需要精确修改时，优先覆盖为可编辑文字图层。

## 6. ComfyUI 还是 Vue？

**第一阶段只用 ComfyUI。** 它已经提供 Web 界面，可以填写提示词、选择模型、调整参数、管理节点工作流和保存结果。个人本地生图不需要另外开发 Vue。

只有当你需要专门的海报编辑体验时，才考虑增加 Vue：
- 选择模板并填写标题、副标题
- 点击文字直接编辑、拖动文字位置
- 调整字体和颜色并实时预览
- 一键生成、重绘背景和导出
- 批量制作、模板管理或自己的应用界面

建议分阶段实施：

| 阶段 | 工具 | 目标 |
|---|---|---|
| 1 | ComfyUI | 跑通 SDXL Lightning 和 Qwen-Image，测速度与画质 |
| 2 | ComfyUI + 现成图片编辑软件 | 验证实际的文字编辑需求 |
| 3（可选） | ComfyUI API + Vue | 只有确实需要模板化、批量化和专用 UI 时才开发 |

## 7. 性能基准测试

准备 4 条固定提示词，每条至少运行 3 次：

### A. 纯背景
`Minimalist premium skincare advertisement background, ivory and sage green palette, soft studio lighting, elegant glass bottle, clean composition, no text, square image.`

### B. 中文标题
`设计一张现代中文新品宣传海报，画面中央清晰呈现标题“秋季新品发布”，下方是“全新设计，焕然一新”，现代简约排版，暖橙色与米白色配色。`

### C. 中英文混排
`Create a premium technology poster with the exact English headline "NEXT GENERATION" and Chinese subtitle "重新定义效率", precise typography, dark navy background, cinematic lighting.`

### D. 复杂产品画面
`A premium wireless headphone product advertisement, metallic textures, dramatic side lighting, detailed reflections, clean visual hierarchy, square composition.`

记录以下数据：

| 指标 | 记录方式 |
|---|---|
| 首次生成时间 | 从提交任务到图片生成完成 |
| 后续生成时间 | 模型已加载后，连续生成的耗时 |
| 显存峰值 | GPU 监控工具 |
| 系统内存峰值 | Windows 任务管理器 |
| 文字准确度 | 对照目标文本，统计字符错误 |
| 画质 | 检查构图、细节、边缘和明显伪影 |

测试前先预热一次，正式测量至少 3 次并记录中位数。模型加载、生成和前端导出时间应分开记录。

建议的验收目标：
- 快速预览：512×512 单张中位生成时间不超过 30 秒。
- 中文文字：单独统计标题和副标题的字符错误。
- 可编辑文字：修改标题后，无需重新运行图像模型即可正确导出。
- 稳定性：连续生成 10 张，不出现显存溢出或服务崩溃。

这些是项目验收标准，不是对具体模型速度的保证。

## 8. 推荐实施顺序

1. 安装 ComfyUI。
2. 跑通 SDXL 1.0 + Lightning，验证 RTX 3060 的快速出图能力。
3. 安装 Qwen-Image-2512 FP8 工作流，测试中英文文字质量、显存和系统内存。
4. 如果 FP8 无法稳定运行，再尝试可信社区的 Q4 GGUF 工作流，并与 FP8 结果比较。
5. 先用现成图片编辑软件验证文字图层工作流。
6. 只有当确实需要模板管理、批量出图或专用交互界面时，再开发 Vue 前端。

## 9. 最终结论

对 RTX 3060 12GB + 32GB RAM 的 Windows 机器，建议先用 **SDXL Lightning + ComfyUI** 建立快速预览流程，再将 **Qwen-Image-2512** 作为中文文字融合和高质量生成的候选主模型。

不要把 Qwen 的速度当成已确定的事实：12GB 显存是主要限制，应以本机基准测试决定是否长期使用。若要求文字可精确修改，优先采用 **AI 生成背景 + 独立 SVG/Canvas 文字图层**，比依赖 AI 每次准确重绘文字更可靠。

**最简启动路线：先装 ComfyUI → 跑 SDXL Lightning → 测速 → 再试 Qwen-Image → 用可编辑文字图层完成海报。**

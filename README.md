# sd-gen-img-win — 本机离线生图 Agent

文生图 / 图生图，带 **4 档精度挡位** 与 **7 档画质挡位**，默认输出 512×512。
针对本机硬件（RTX 3060 12GB）选型与调优，所有速度数字均为本机实测。

---

## 1. 结论速览

| 问题 | 结论 |
|---|---|
| 能跑吗 | 能。RTX 3060 12GB 足以在本机离线生图，无需联网推理 |
| 512×512 快吗 | 极快。默认挡位（sd-turbo / bf16 / turbo）**0.34 秒**/张；SDXL fp16 约 5.1 秒/张 |
| 30 秒预算 | 18 个配置中 15 个满足，最快有 **88 倍余量**；仅「SDXL + fp32/tf32」超预算（显存硬上限） |
| 512 该用哪个模型 | **默认已改为 sd-turbo**（原生 512、主体最完整且最快）；需负向提示词用 `sd15`；要 SDXL 画质就出 1024（§7.3 有对照实验） |
| 精度挡位 | **`bf16`（默认）** / `fp16` / `tf32` / `fp32`；`fp8` 本机硬件不支持，会明确报错。实测 bf16 在 sd-turbo 上比 fp16 快 25%（§7.2） |
| 怎么指定挡位 | 命令行 `--precision fp16 --gear balanced`，**或直接在提示词里写** `[精度=fp16] [画质=max] [尺寸=640x640]` |
| 有可视化界面吗 | 有。**双击 `web.bat`** 即可（自动预热模型并打开浏览器）；支持**上传图片**与**文生图 / 图生图**切换，见 §6.5 |

> 与同机另一套 ComfyUI 工作流集（Qwen-Image-2.1，中文原生）的横向对比：
> **[docs/comparison-vs-comfyui.md](docs/comparison-vs-comfyui.md)** —— 谁适合出草稿、谁适合出成品，表格里写清楚了。

---

## 2. 本机硬件实测

| 项目 | 实测值 | 对方案的影响 |
|---|---|---|
| GPU | **NVIDIA GeForce RTX 3060 12GB**（Ampere，计算能力 **sm_86**） | 有 fp16 / bf16 张量核与 TF32；**没有 fp8 硬件单元**（需 Ada sm_89+） |
| 显卡驱动 | 617.42（CUDA UMD 13.4） | 兼容 CUDA 12.x / 13.x 运行时 |
| CPU / 内存 | Intel i7-10700K（8 核 16 线程）/ 32GB | 权重加载与 CPU offload 有富余 |
| 磁盘 | C: 244GB 可用，D: 1261GB 可用 | 多套权重（16GB）无压力 |
| Python 环境 | conda 26.7.1，新建环境 `genimg`（Python 3.12.15） | 与 AutoClaw 内置解释器隔离 |
| 推理栈 | torch **2.14.0+cu126**、diffusers **0.41.0**、transformers 5.20.0 | `torch.cuda.is_available() = True`，实测可出图 |
| 网络 | huggingface.co 需走代理；hf-mirror.com 可直连且更快（实测 6.69 MB/s vs 代理 2.1 MB/s） | 下载脚本默认走 hf-mirror，代理自动探测 |

---

## 3. 选型理由

### 3.1 瓶颈不在速度，在质量

30 秒/张 @512×512 对 3060 12GB 来说非常宽松：实测 SDXL fp16 跑 25 步只需 **5.73 秒**。
既然速度有巨大余量，选型就应该**以画质为先**，而不是去追更小的模型。

### 3.2 SDXL 在 512 下会崩构图 —— 这是选型的关键分水岭

SDXL 的原生训练分辨率是 1024×1024，它在训练时把 `original_size / target_size` 作为
micro-conditioning 输入。直接在 512×512 出图时，diffusers 默认把这些条件填成 (512, 512)，
与训练分布不符，实测会出现**多格拼贴式构图崩坏**：

| 配置 | 512×512 出图表现 |
|---|---|
| SDXL + 默认条件化 | 出现"三格贴纸"式构图崩坏（见 `results/lowres_exp/A_sdxl512_default_cond/`） |
| SDXL + 原生分辨率条件化 (1024) | 见 §7 实测（脚本 `tools/exp_lowres.py`） |
| SD1.5（原生 512） | 构图正常 |
| SD-Turbo（原生 512） | 构图正常，且最快 |

> 结论：**要求 512×512 时，默认应该用原生 512 的模型**；SDXL 保留给 1024 出图与质量上限场景。

### 3.3 模型清单

| 键 | 仓库 | 原生分辨率 | 定位 | 下载体积 |
|---|---|---|---|---|
| `sd-turbo` | `stabilityai/sd-turbo` | 512 | **512 极速**：1–4 步出图 | 3.3GB |
| `sd15` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 512 | **512 通用**：生态最大，图生图稳 | 6.0GB |
| `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` | 1024 | **高质量**：细节与语义最强 | 6.5GB |
| `sdxl-turbo` | `stabilityai/sdxl-turbo` | 512 | SDXL 蒸馏版（可选，未默认下载）| 6.9GB |
| — | `ByteDance/SDXL-Lightning` | — | SDXL 的 4/8 步加速 LoRA | 0.7GB |

### 3.4 为什么不用本机已有的 Qwen-Image

本机 ComfyUI 目录下已有 Qwen-Image 2.1（GGUF Q4/Q5，20B 级 MMDiT）。
它是质量上限更高的模型，但 20B 参数在 12GB 显存上需要重度 offload，
512 分辨率单张远超 30 秒预算，因此**不符合本任务约束**，未纳入。

---

## 4. 精度挡位

| 挡位 | 权重/计算 | 本机可用 | 说明 |
|---|---|---|---|
| `fp16` | fp16 | ✅ | **默认**。张量核吞吐最高、显存最小；VAE 单独用 fp32 解码，规避半精度黑图 |
| `bf16` | bf16 | ✅ | 动态范围与 fp32 相同，几乎不会溢出/出黑图；Ampere 上速度与 fp16 接近 |
| `tf32` | fp32 权重 + TF32 张量核 | ✅ | 权重保持 fp32，矩阵乘/卷积走 TF32（10 位尾数），兼顾保真与速度 |
| `fp32` | fp32 | ✅（大模型会 offload） | 画质基线。SDXL 的 fp32 权重需 13.9GB，超出 12GB 显存，会自动转 CPU offload 并且**明显变慢** |
| `fp8` | fp8 | ❌ **硬件不支持** | fp8 张量核需要 sm_89（Ada/Hopper）。本机是 sm_86，软件无法绕过；指定该挡位会明确报错并列出可用挡位 |

### 精度挡位的两个实现要点

1. **VAE 解码精度与主精度解耦**：fp16 挡位下 VAE 保持 fp32，并在采样的最后一步把 latent
   转到 fp32 再解码。这样既拿到 fp16 的采样速度，又避免半精度 VAE 偶发的黑图/花屏。
2. **放置策略自动决策**（`--placement auto`）：脚本按"权重体积 + 激活余量"估算显存占用，
   自动在「全量常驻显存 / 模型级 CPU offload / 逐层 CPU offload」之间选择，并把决策写进元数据。

---

## 5. 画质挡位

| 挡位 | 步数 | CFG | 适用模型 | 说明 |
|---|---|---|---|---|
| `turbo` | 2 | 0.0 | sd-turbo / sdxl-turbo | 极速。CFG=0 表示关闭引导，**负向提示词不生效**；图生图做风格改造请避开（§6.6）|
| `lightning8` | 8 | 1.5 | sdxl | 挂 Lightning 8 步 LoRA |
| `lightning` | 4 | 1.0 | sdxl | 挂 Lightning 4 步 LoRA。CFG=1 即关闭引导 |
| `draft` | 12 | 5.0 | 全部 | 快速试提示词，细节偏软 |
| `balanced` | 25 | 7.0 | 全部 | **默认**，画质/耗时平衡点 |
| `quality` | 35 | 7.5 | 全部 | 纹理与光影更完整 |
| `max` | 50 | 8.0 | 全部 | 质量上限参考 |

---

## 6. 用法

### 6.1 环境准备

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1              # 建环境 + 装依赖 + 下权重（约 16GB）
powershell -ExecutionPolicy Bypass -File setup.ps1 -SkipModels  # 只装环境
```

### 6.2 生图

```powershell
# 512×512 极速（推荐用于 512）
powershell -File run.ps1 "a red panda astronaut, cinematic" -m sd-turbo -g turbo -s 512x512

# SDXL 高质量（原生 1024）
powershell -File run.ps1 "a red panda astronaut, cinematic" -m sdxl -s 1024x1024 -g quality

# 在提示词里直接标挡位（与命令行等价，命令行优先级更高）
powershell -File run.ps1 "水墨山水，留白 [尺寸=640x640] [精度=fp32] [画质=max]"

# 图生图
powershell -File run.ps1 --image .\in.png --strength 0.5 "oil painting style"

# 验收速度：跑 5 次并给出中位数
powershell -File run.ps1 "a cat" -m sd-turbo -g turbo --warmup 1 --repeat 5

# 自检：本机硬件能力 + 可用模型/挡位
powershell -File run.ps1 --list
```

### 6.3 提示词内嵌指令

| 指令 | 中文写法 | 取值示例 |
|---|---|---|
| 精度挡位 | `[精度=fp16]` | fp16 / bf16 / tf32 / fp32 |
| 画质挡位 | `[画质=balanced]` | turbo / lightning / lightning8 / draft / balanced / quality / max |
| 尺寸 | `[尺寸=640x640]` | 512x512、768x768、1024x1024 |
| 模型 | `[模型=sdxl]` | sdxl / sd15 / sd-turbo / sdxl-turbo |
| 步数 | `[步数=30]` | 整数 |
| 引导强度 | `[cfg=7.5]` | 浮点 |
| 种子 | `[种子=1234]` | 整数 |
| 改动强度 | `[强度=0.5]` | 0–1（图生图，默认 0.5 保主体） |
| 负向提示词 | `[负向=blurry, low quality]` | 任意文本 |

* 优先级：**显式命令行参数 > 提示词内嵌指令 > 默认值**（来源会打印出来，便于核对）。
* 只识别已知键；`[blurry]`、`[cat:dog:0.5]` 这类 A1111 原生语法会原样保留，不会被吃掉。

### 6.4 其他能力

```powershell
# 单张图硬预算：超时主动中止（防止某个挡位拖死）
powershell -File run.ps1 "a cat" -m sdxl -p fp32 --timeout 30

# JSON 输出（便于别的程序调用）+ 元数据落盘
powershell -File run.ps1 "a cat" --json --manifest results\run.json

# 逐挡位基准测试
C:\Users\PPXW\miniconda3\envs\genimg\python.exe bench.py --size 512x512 --reps 3 --warmup 2
```

每张图输出到 `outputs/<时间戳>/`，文件名含模型/精度/画质/种子，PNG 元数据里写入完整参数，
可复现命令也会随结果打印。

### 6.5 Web 界面（上传图片 / 文生图 / 图生图）

```powershell
:: 最简单：双击 web.bat —— 后端 + 界面一起启动，自动预热模型并打开浏览器

# 也可以用命令行（等价，便于调参数）
powershell -ExecutionPolicy Bypass -File web.ps1                    # 预热默认挡位
powershell -ExecutionPolicy Bypass -File web.ps1 -Port 8080 -Open    # 换端口 + 自动开浏览器
powershell -ExecutionPolicy Bypass -File web.ps1 -Preload:$false     # 不预热
```

界面能力：

| 区域 | 功能 |
|---|---|
| 顶部 | 实时显示 GPU / 算力 / 显存 / torch / CUDA，以及 fp8 是否可用 |
| 模式切换 | **文生图** / **图生图**；切到图生图才出现上传区与“改动强度”滑块 |
| 上传区 | 点击选择或直接拖拽（jpg/png/webp，≤20MB），带缩略图预览 |
| 参数区 | 模型、精度挡位（不可用挡位置灰并显示原因）、画质挡位（按模型自动过滤）、尺寸预设/自定义、步数、CFG、种子、张数 |
| 进度 | 采样步进度条 + 已用时间；权重加载单独计时 |
| 结果 | 生成图（点击看原图）+ 推理/去噪/编解码耗时、显存峰值、放置策略、**30 秒预算判定** |
| 历史 | 最近 12 次生成缩略图，点击查看 |

三个实现选择，以及为什么：

* **单工作线程串行**：只有一块 GPU，并发只会互相抢显存、双双变慢；请求进队列排队执行。
* **管线常驻**：同精度下连续出图不重复加载权重（换精度才重载）。
* **不依赖外网**：前端是本地静态文件，无 CDN、无打包步骤，断网也能用。

HTTP 接口（脚本调用也方便）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/catalog` | 硬件能力 + 模型（含是否已下载）+ 精度挡位（含不可用原因）+ 画质挡位 |
| POST | `/api/generate` | multipart 提交，返回 `job_id` |
| GET | `/api/jobs/{id}` | 任务状态：`queued/running/done/error` + 步进度 + 结果元数据 |
| GET | `/api/history` | 最近完成的生成 |
| GET | `/outputs/...` | 生成的图片 |

已做端到端实测（`tools/test_web.py` 自动跑）：文生图 0.65s、上传图片做图生图 0.35s，
返回的图片是有效 PNG（校验了 PNG 签名），`/api/history` 可回列出历史记录。
挡位对照图见 `results/bench/contact_sheet.png`。

**启动时预热**：`web.bat` 默认带 `--preload`，服务起来后会在后台把默认挡位
（sd-turbo / bf16 / turbo）加载进显存，并空跑一步把首次前向的 kernel 选择也热掉。
实测：预热约 12s 完成，之后**第一次点生成就是 0.32s**；若不预热，第一次生成要等约 10–12s 加载权重。
页面右侧会显示“模型已就绪 / 预热中”，想关掉预热就用 `web.ps1 -Preload:$false`。

#### 手动测试步骤（建议按顺序点一遍）

1. 启动：**双击 `web.bat`**（或 `powershell -File web.ps1 -Open`），控制台会打印实际地址并自动打开浏览器。
2. 页面顶部应显示 `RTX 3060 · sm_86 · 12.0 GB`；若显示“正在检测硬件…”不消失，看控制台报错。
3. **文生图**：模型 `SD-Turbo`、精度 `fp16`、画质 `turbo`、尺寸 `512×512` → 点「生成」。
   预期：几秒内出现图片，指标区显示推理耗时且标注「满足 30 秒预算」。
4. **图生图**：切到「图生图」，拖入刚生成的图片，提示词改 `oil painting style`，
   改动强度保持默认 `0.5` → 生成。预期：构图与主体保留、风格改变。
5. **挡位对比**：同提示词同种子，分别用 `fp16` / `bf16` / `tf32` / `fp32` 各生成一次，
   对比耗时与显存峰值（sd15 上差距最大：fp32 ≈ 8.8s、bf16 ≈ 3.4s，见 §7.3）。
6. **边界验证**：
   * 精度选 `fp8` → 该项应为置灰状态（鼠标悬停显示硬件原因）；
   * 提示词留空 → 前端提示“请填写提示词”；
   * 尺寸填 `500` → 后端返回“需为 256–1536 之间且能被 8 整除”；
   * 图生图未上传图片 → 后端返回“图生图需要先上传一张图片”。
7. 出图位置：`outputs/web/<任务号>/`，PNG 内嵌完整参数元数据。

---

### 6.6 图生图做「风格改造」的正确姿势（实测）

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
| 改动强度 | **0.5（默认）** 保主体；0.65–0.7 强风格化 | 见下表 |
| 提示词 | 英文 | `watercolor painting style, soft translucent washes, rich details, paper texture` |
| 负向提示词 | `blurry, low quality, deformed` | CFG>1 时才生效 |

**强度对照（同一张图、同一提示词、sd15 / balanced）**：

| 强度 | 实际采样步数 | 效果 |
|---|---|---|
| **0.50（默认）** | 12 步 | 主体完整，呈现「淡彩水彩」效果 |
| 0.60 | 15 步 | 主体开始变形（本例脸部和人像混淆） |
| 0.70 | 17 步 | 完全艺术化重画，主体基本丢失（本例变成花卉） |

**现在会自动提示**：命令行与网页都会在生成前检查这几个坑（规则在 `genimg/advice.py`），
网页端还多了一个「用风格改造推荐配置」按钮，一键切到 `sd15 / balanced / 0.7`。
若图生图的实际步数算出来是 0（步数×强度 < 1），会直接报错并说明原因，而不是丢一个底层异常。


### 6.7 中文提示词支持（纯离线词典）

**现实**：`sd15` / `sdxl` / `sd-turbo` 的文本编码器都是 CLIP（英文语料为主），中文提示词过完分词
大多是无效 token。同一组参数、同一种子，中文写「水彩风格」得到的是糊在一起的色块，
英文写 `watercolor painting style` 才出纸质感。

**做法**：内置一个**纯离线词典**（116 条），把风格 / 画质 / 光影 / 构图 / 颜色 / 常见场景
翻成英文，并剔除「把图片改为」「请帮我」这类对 CLIP 无意义的指令性措辞。
**不联网、不依赖任何外部服务、不涉及任何凭据**，零延迟、结果可预期。

**翻译效果（可复现：`python tools\test_translate.py`）**

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

### 6.8 画幅：按比例出图，参考图不再被拉伸

**旧版的问题**：图生图把参考图直接 `resize((512, 512))`。一张 1024×683 的横图塞进方框，
横向被压扁 1.5 倍——人脸、建筑全部变形。

**现在的规则**（命令行 / 网页 / 提示词三处都生效）：

| 写法 | 效果 |
|---|---|
| `--size 3:4` 或 `[尺寸=3:4]` | 按比例选尺寸：SD1.5 → 384×512，SDXL → 768×1024 |
| `--size 768x1024` | 固定像素（会对齐到 8 的倍数） |
| `--size match` | 跟随参考图比例 |
| 不写 | 图生图跟随参考图；文生图读提示词里的 `3:4` / `竖版`；都没有则用模型原生 1:1 |

**参考图怎么装进画布**：等比缩放 + 自身放大模糊后的留白填充，**不拉伸**，主体比例保持不变。

**尺寸桶**（64 的倍数；比例优先，面积尽量贴近原生）：

| 比例 | SD1.5 / SD-Turbo | SDXL |
|---|---|---|
| 1:1 | 512×512 | 1024×1024 |
| 3:4 | 384×512 | 768×1024 |
| 2:3 | 384×576 | 768×1152 |
| 4:3 | 512×384 | 1024×768 |
| 16:9 | 576×320 | 1344×768 |

**实测（3:4、图生图、强度 0.55、参考图 1024×683）**

| 模型 / 挡位 | 尺寸 | 推理 |
|---|---|---|
| sd15 bf16 balanced | 384×512 | **1.46 s** |
| sdxl fp16 lightning | 768×1024 | **5.66 s** |
| sdxl fp16 draft | 768×1024 | 7.49 s |
| sdxl fp16 balanced | 768×1024 | 11.44 s |

> 768×1024 时 SDXL 权重放不进 12GB，放置策略自动降级为 `model_offload`，但仍在 30 秒内。

**顺带修掉一个真 bug**：fp16 挡位下图生图报 `Input type (Half) and bias type (float)`。
原因是 diffusers 会把输入图先 cast 成 UNet 的 fp16，再送进 fp32 的 VAE。
现在改为用 VAE 精度自行编码成 4 通道 latent 交给管线，并在解码入口兜一次 dtype 转换；
sd15 / SDXL 均实测通过（输出均值约 130，非黑图）。

#### 长提示词的真相（实测）

那份「水彩上下卡片」提示词 4942 字符，实测 **835 个 token**，而 CLIP 单次只吃 **77 个**：
**约 91% 被静默截断**（transformers 会打印 `truncated because CLIP can only handle sequences up to 77 tokens`）。
截断从「the bottom half occupies...」开始——也就是说「下半部分画插画」这条指令根本没进模型。

结论：**「上半照片 + 下半插画 + 各占 50%」这种版式，单次 SD 生成做不出来**，改提示词也没用。
可行做法只有两条：

1. **两段式**：上半用照片（必要时先做背景延展），下半单独生成插画，最后按 50/50 拼版 —— 版式可控、可复现；
2. 换支持图像编辑/版式控制的模型（如 Qwen-Image-Edit），见 `docs/comparison-vs-comfyui.md`。

本项目目前只做到「画幅与参考图正确」，两段式拼版**未内置**。

### 6.9 两段式海报：上半照片 + 下半插画（严格 50/50）

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
python tools\make_poster.py `
  --image "C:\Users\PPXW\Desktop\1.jpg" "C:\Users\PPXW\Desktop\3.jpg" `
  --prompt-file "C:\Users\PPXW\Desktop\水彩上下crad提示词.txt" `
  --model sdxl --gear balanced --size 3:4 --strength 0.72 --art-scale 0.62 `
  --out results\poster_two_pass
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

#### 6.9.1 对齐外部平台效果：做了什么、差在哪

对照图：`results/ext_ref/a1.jpg`、`a2.jpg`（外部平台生成）。它们的共同点是
**水彩质感 + 纸纹底 + 衬线小字排版 + 主体只占约 40–50% 宽、大片留白**。
据此对 `poster.py` 做了四项改进：

| 改进 | 做法 | 效果 |
|---|---|---|
| 融合方式 | 插画用 **正片叠底** 压在纸纹上 | 白色留白自动变回纸色，不再是"贴了一张矩形图" |
| 纸纹底 | 程序生成暖白底 + 6% 高斯噪点 | 有了纸张颗粒感，不再是纯色块 |
| 排版 | Georgia 衬线 + 手动字距 + 细分隔线 | 「标题 / 分隔线 / 副标题 / 年份」四段式，接近参考图 |
| 边缘羽化 | 蒙版高斯模糊 + 降饱和至 0.88 | 插画边缘融进纸面 |

**试过但失败的路线（已记录，避免重复踩）**

1. 先把照片缩成小图放白底，再整体 img2img —— 模型收到的视觉信号太弱，
   熊猫被画成哥特窗、人被画成鸟（实测 2 次，强度 0.70 / 0.75 都失败）。
2. 纯水彩提示词 + 满幅输入 —— 主体保住了，但出来是高对比版画/墨线，不是水彩。
   原因是 SDXL 对 "watercolor" 的理解偏向"高对比插画"，且负向里的
   `photorealistic` 会进一步推向平面化。

**结论：剩下的差距是模型能力，不是参数**

外部那两张的关键特征——柔和晕染的水彩、纸面纹理、**可读的排版文字**——来自
指令跟随型模型（Qwen-Image-Edit / GPT-Image 一类）。本地 SDXL 是 2022 年的
UNet + CLIP：77 token 上限、不会写字、对"画成水彩"只能做风格化改写。
在 768×512 上继续调参，收益已经很小。

要真正对齐，正确做法是**换模型**而不是继续调提示词：本机另一套
`D:\ai\comfyui-gen-img-workflows-win` 里的 **Qwen-Image-2.1 / Qwen-Image-Edit**
正是干这个的，它的中文与文字能力远超 SDXL。本项目可以作为"快速草稿 + 版式拼装"的
前置环节，把成片交给那套出。

#### 6.9.2 为什么"昨天能看、今天很抽象"：强度断崖

现象：下半插画与参考图差别过大，熊猫被洗成色块。

原因不在随机种子，而在**我上一轮把画风换成水彩、却没同时下调强度**。
水彩提示词（`soft translucent washes`）本身就在"洗细节"，再叠加 0.72 的重绘力度，
主体必然被冲掉。

**强度扫描实测**（同一张熊猫照片、同一提示词、同一种子，768×512）

| 强度 | 与参考图差异（越小越像） | 观感 |
|---|---|---|
| 0.35 | 0.091 | 几乎还是照片，只有轻微笔触 |
| 0.45 | 0.161 | 照片感仍强 |
| **0.55** | **0.187** | **既像参考图，又有水彩色块** |
| 0.65 | 0.183 | 风格更明显，主体仍可辨 |
| 0.72 | **0.296** | **断崖：主体被洗成抽象色块** |

差异指标是灰度直方图的 L1 距离，0–1；0.18 → 0.30 是肉眼能立刻看出的跳变。

**现在的默认值**：`--strength 0.58`（落在 0.55–0.65 的安全区内）。
命令里显式给 ≥ 0.68 时，输出会明显偏抽象——这是有意的取舍，不是 bug。

> 附带损失：参考图里原有的水印/字幕（如熊猫照片底部的新闻条）会被一起"水彩化"，
> 在 0.55–0.65 区间无法完全去掉。要干净底图，应在生成前先裁掉这些区域。

## 7. 实测数据（512×512）

### 7.1 全量挡位矩阵

实测环境：RTX 3060（sm_86，12GB）· torch 2.14.0+cu126 / CUDA 12.6 · 512×512 ·
每配置预热 2 次 + 计时 3 次（取中位数）· 提示词 `a red panda astronaut floating in space, cinematic lighting, highly detailed`

| 模型 | 精度 | 画质 | 步数 | 中位耗时 | 去噪 | 编解码 | 显存峰值 | 放置 | ≤30s |
|---|---|---|---|---|---|---|---|---|---|
| sd15 | fp32 | balanced | 25 | **8.82s** | 8.52s | 0.30s | 4.76GB | gpu | ✅ |
| sd15 | tf32 | balanced | 25 | **8.24s** | 7.94s | 0.32s | 4.95GB | gpu | ✅ |
| sd15 | bf16 | balanced | 25 | **3.38s** | 3.21s | 0.16s | 2.61GB | gpu | ✅ |
| sd15 | fp16 | balanced | 25 | **3.34s** | 3.05s | 0.29s | 2.92GB | gpu | ✅ |
| sd15 | fp16 | quality | 35 | **4.83s** | 4.53s | 0.31s | 2.92GB | gpu | ✅ |
| sdxl | fp32 | balanced | 25 | **60.76s** | 60.25s | 0.51s | 0.76GB | sequential_offload | ❌ |
| sdxl | tf32 | balanced | 25 | **62.84s** | 62.25s | 0.58s | 0.95GB | sequential_offload | ❌ |
| sdxl | bf16 | balanced | 25 | **5.18s** | 5.01s | 0.17s | 7.17GB | gpu | ✅ |
| sdxl | fp16 | balanced | 25 | **5.11s** | 4.82s | 0.29s | 7.48GB | gpu | ✅ |
| sdxl | fp16 | draft | 12 | **2.70s** | 2.41s | 0.29s | 7.48GB | gpu | ✅ |
| sdxl | fp16 | quality | 35 | **7.57s** | 7.25s | 0.32s | 7.48GB | gpu | ✅ |
| sdxl | fp16 | lightning | 4 | **0.82s** | 0.53s | 0.29s | 7.85GB | gpu | ✅ |
| sdxl | fp16 | lightning8 | 8 | **1.96s** | 1.66s | 0.30s | 7.85GB | gpu | ✅ |
| sdxl | fp8 | balanced | – | 硬件不支持 | – | – | – | – | – |
| sd-turbo | fp32 | turbo | 2 | **0.64s** | 0.35s | 0.28s | 5.59GB | gpu | ✅ |
| sd-turbo | tf32 | turbo | 2 | **0.59s** | 0.30s | 0.29s | 5.78GB | gpu | ✅ |
| sd-turbo | bf16 | turbo | 2 | **0.34s** | 0.17s | 0.16s | 3.03GB | gpu | ✅ |
| sd-turbo | fp16 | turbo | 2 | **0.45s** | 0.16s | 0.29s | 3.33GB | gpu | ✅ |

**17/18 个配置可跑通，其中 15 个满足 30 秒预算**；最快 0.34s（sd-turbo/bf16），
最慢可跑通项 62.84s（sdxl/tf32，已超预算）。原始数据：`results/bench/bench.csv`、`bench.json`。

### 7.2 从数据里得出的四条结论

1. **bf16 是本机最佳的默认精度**。sd-turbo 上 bf16（0.34s）比 fp16（0.45s）快 25%，
   显存 3.03 vs 3.33GB；sd15/sdxl 上两者持平（3.38 vs 3.34、5.18 vs 5.11）。
   原因：fp16 挡位为了保证画质，把 VAE 单独提到 fp32 解码，因此多出约 0.13s 的编解码开销；
   bf16 的动态范围与 fp32 一致，VAE 不需要升精度。→ **默认已改为 bf16**。
2. **tf32 在这台机器上不值得用**。sd15 从 8.82s 降到 8.24s（仅快 7%），
   sdxl 反而更慢（60.76s → 62.84s，两者都已 offload）。512 分辨率下 25 步的 UNet
   并不是纯 GEMM 瓶颈，TF32 的张量核优势发挥不出来。
3. **SDXL 的 fp32 / tf32 是显存硬上限**。fp32 权重共 13.9GB，超过 12GB 显存，
   自动退化为 `sequential_offload`（峰值显存只有 0.76GB，代价是每步都搬运权重），
   单张 60s+。这不是配置问题，加多少参数都绕不过去。
4. **SDXL + Lightning LoRA 是 SDXL 侧最有性价比的挡位**：4 步 0.82s、8 步 1.96s，
   比 balanced（5.11s）快 6 倍以上。代价是 CFG 被关闭，负向提示词不生效。

### 7.3 512×512 到底该用哪个模型（含对照实验）

同提示词、同种子、512×512，横向比对（图在 `results/lowres_exp/`）：

| 配置 | 耗时 | 512 画质表现 |
|---|---|---|
| `sdxl` @512，默认条件化 | 5.44s | **三格拼贴式构图崩坏**（见 `A_sdxl512_default_cond/`）|
| `sdxl` @512，原生分辨率条件化（original_size=target_size=1024）| 5.13s | **仍然三格拼贴**，只是主体更完整（见 `B_sdxl512_native_cond/`）|
| `sdxl` + lightning @512 | 0.82s | 主体清楚，但仍有明显竖直拼接缝 |
| `sd15` @512（原生 512）| 3.33s | 构图正常、画面干净，但主体保真偏弱 |
| `sd-turbo` @512（原生 512）| 0.57s | **主体完整、构图干净，本轮对比中最好** |

> **结论**：SDXL 在 512×512 下的崩坏**不是 micro-conditioning 能修的**（A/B 对照已证明），
> 本质是原生 1024 的模型在分布外分辨率上工作。要求 512 时就用原生 512 模型：
> 默认 `sd-turbo`（最快且主体最完整），需要负向提示词/CFG 时用 `sd15`，
> 要 SDXL 的画质就把它出到 1024。

### 7.4 修复“精度挡位失效”前后的逐组件对比

这也是为什么上表里的 sd15/sdxl 数字比上周的初版快了约 20 倍：

| 组件（SDXL fp16 @512）| 修复前 | 修复后 |
|---|---|---|
| UNet 单步前向 | 2428.44 ms | **119.16 ms** |
| VAE 解码 | 5070.74 ms | **278.62 ms** |
| text_encoder_2 前向 | 116.81 ms | **20.87 ms** |
| text_encoder 前向 | 8.62 ms | 8.03 ms |

---

## 8. 目录结构

```
sd-gen-img-win/
├─ genimg.py                  # 入口
├─ genimg/
│  ├─ presets.py              # 模型档案 / 精度挡位 / 画质挡位 / 硬件能力判定
│  ├─ directives.py           # 提示词内嵌指令解析
│  ├─ runtime.py              # 代理探测、HF 配置、精度落地、显存放置、计时
│  ├─ engine.py               # 管线装配、缓存、推理、元数据
│  └─ cli.py                  # 命令行
├─ bench.py                   # 挡位基准测试
├─ web.bat                    # 双击启动 Web 界面（推荐）
├─ web.ps1                    # 启动脚本本体（-Port / -Open / -Preload:$false）
├─ setup.ps1 / run.ps1        # 环境准备 / 命令行入口（UTF-8 BOM，兼容 PowerShell 5.1）
├─ scripts/common.ps1         # conda 与环境变量处理（含 AutoClaw 变量兜底）
├─ webapp/
│  ├─ server.py               # FastAPI 后端：任务队列 + 上传 + 接口
│  └─ static/                 # 前端（原生 HTML/CSS/JS，无 CDN）
│     ├─ index.html
│     ├─ app.js               # 目录拉取、模式切换、上传、轮询进度、结果渲染
│     └─ style.css
├─ tools/
│  ├─ fetch_models.py         # 按需下载权重（allow_patterns 只拉需要的文件）
│  ├─ make_contact_sheet.py   # 把样图拼成带标注的对照网格
│  ├─ exp_lowres.py           # 实验：SDXL 在 512 下的条件化对比
│  ├─ diag_profile.py         # 逐组件性能剖析（定位慢在哪）
│  ├─ diag_dtype.py           # 精度参数名有效性验证
│  ├─ check_frontend.py       # 前端 ID 交叉检查（浏览器不能访 localhost 时兜底）
│  └─ selfcheck.py            # 语法/导入自检
├─ models/                    # 权重缓存（HF 格式）
├─ outputs/                   # 生成结果
└─ results/                   # 基准与实验数据
```

---

## 9. 已知限制

* **SDXL + fp32** 权重 13.9GB 超出 12GB 显存，会走 CPU offload，单张耗时远超 30 秒预算 —— 这是显存硬约束，不是配置问题。
* **fp8 挡位**在本机不可用（sm_86 无 fp8 张量核），脚本会明确拒绝而非假装支持。
* `turbo` / `lightning` 挡位关闭了 CFG，因此**负向提示词不生效**（脚本会按 CFG>1 才传负向提示词）。
* 模型仓库 `sdxl-turbo` 默认未下载，需要时用 `python tools\fetch_models.py --with-sdxl-turbo`。
* 权重下载受网络影响：huggingface.co 需代理，hf-mirror.com 更快但偶有限流（429），下载脚本已带重试与续传。
* **模型已缓存时是完全离线加载的**（`local_files_only=True`）：即使 hf-mirror 抽风（实测遇到过 403）也不影响出图；
  反过来，模型未缓存且网络不可用时，会直接提示“请先运行 tools\fetch_models.py”，而不是丢一堆堆栈。
* `turbo` / `lightning` 档的 CFG=0，**图生图的风格迁移会明显偏弱**（提示词控制力下降）；
  想认真做风格改造，用 `sd15` 或带 CFG 的 `sdxl` 常规档位。
* Web 服务是**单任务串行**的：同时点两次只会排队，不会并行占显存。Port 7860 被占用时会自动往后找（实测自动切到 7861）。

---

## 10. 踩坑记录（本轮真实遇到的 5 个缺陷）

这一节保留下来，因为其中前两个是**症状隐蔽、后果严重**的类型：不报错、只是"变慢"，
很容易被误当成"显卡不行"。

### 10.1 精度挡位静默失效（最严重）

* **症状**：选了 `--precision fp16`，但 SDXL 单张要 120 秒还跑不完 25 步（约 10 秒/步）。
* **根因**：`Img2ImgPipeline.from_pipe(pipe)` 在未显式传 dtype 时，
  diffusers 内部写的是 `dtype = _resolve_dtype(...) or torch.float32` —— **默认把整条管线转成 float32**。
  于是 fp16 权重全被抹成 fp32（SDXL 共 13.9GB），超出 12GB 物理显存。
  Windows 的 WDDM 不允许直接 OOM，而是允许超卖、走 PCIe 换页，于是表现为"只是慢"。
* **证据**：显存已分配 13.10GB（fp16 时只有 6.6GB）；UNet 单步 2428ms vs 正确值 119ms。
* **修法**：① `from_pipe` 显式传 dtype；② 装配结束后**回读各组件实际 dtype 并纠正**
  （`ensure_dtypes`）。教训：diffusers 有多条会静默改写精度的路径，传参成功 ≠ 生效。

### 10.2 管线缓存没释放显存，污染放置决策

* **症状**：同一进程里先跑 fp32 再跑 fp16，fp16 竟被判定成"需要逐层 offload"，显存峰值 10.42GB。
* **根因**：引擎缓存了上一条管线且未释放，新管线做显存估算时看到的是**被上一条占掉**的显存。
* **修法**：换精度挡位前 `_evict_all()`（clear + `gc.collect()` + `torch.cuda.empty_cache()`）。

### 10.3 fp16 的 latent 撞上 fp32 的 VAE

* **症状**：`RuntimeError: Input type (struct c10::Half) and bias type (float) should be the same`。
* **根因**：把 VAE 单独加载成 fp32 后，管线内部"需要时把 latent 转成 VAE 精度"的逻辑不再触发
  （**只有 SDXL 系有这段逻辑，SD1.5 没有**），于是 fp16 latent 直接送进 fp32 VAE。
* **修法**：权重整体按主精度加载，装配后再把 VAE 调到目标精度，并在**采样最后一步**
  把 latent 转成 VAE 精度（中途转会把 fp32 latent 送回 fp16 权重的 UNet）。

### 10.4 环境变量陷阱：pip 把包装到了别的目录

* **症状**：`pip install` 显示成功，但目标环境里没有包。
* **根因**：AutoClaw 向会话注入 `PIP_USER=1` 与 `PYTHONUSERBASE=...\AutoClaw-official\toolchain\python`，
  pip 因此装进 `...\Python312\site-packages`（实测被装入 4.09GB，且该目录对**任何** Python 3.12 都可见）。
* **修法**：安装/运行前显式设 `PIP_USER=0`、`PYTHONNOUSERSITE=1` 并清空 `PYTHONUSERBASE`（已写进 `scripts\common.ps1`）。

### 10.5 PowerShell 5.1 的两个坑

* **编码**：PS 5.1 只把带 **UTF-8 BOM** 的 `.ps1` 当 UTF-8 解析，否则脚本里的中文会乱码
  （脚本已统一保存为 UTF-8 with BOM）。
* **空环境变量**：本机 `$env:LOCALAPPDATA` 为空，而 `Join-Path` 写进数组字面量会对**每个**元素求值，
  一句失败整个数组构造就报 `Cannot bind argument to parameter 'Path' because it is null`。
  修法：每个候选路径先判空。同时改用 `$PSCommandPath` 推导脚本根目录（`$PSScriptRoot` 在被点源的文件里可能为空）。
* **同一个坑的另一面**：`$env:APPDATA` 在受限/最小化的会话里也会为空，而 `gh` 靠
  `%APPDATA%\GitHub CLI\hosts.yml` 找凭据，于是 `gh auth status` 报“未登录”——其实登录态一直在。
  补上 `$env:APPDATA="C:\Users\<用户名>\AppData\Roaming"` 后 `gh` 与 `git push` 立即恢复正常，不必重新登录。

---

## 11. 后续可选（非阻塞）

本期目标（硬件评估 / 选型 / 512×512 30 秒内出图 / 多精度挡位 / 可上传的 Web 界面）已完成，
以下是可继续深挖的方向，按性价比排序：

1. **1024 出图调优**：SDXL 原生 1024 才是它的主场；目前只测了 512，可补一组 1024 的挡位矩阵
   （显存占用会明显上升，fp16 下预计 9–11GB，可能需要 VAE tiling）。
2. **SD1.5 微调模型/中文提示词适配**：默认模型对中文提示词支持一般，可挂中文 CLIP 或换社区精调模型。
3. **图生图的局部重绘（inpaint）**：当前 img2img 是整图重绘，加 mask 可做局部修改。
4. **量化挡位**：Windows 下 `torchao` / `bitsandbytes` 若可用，可再压显存或提速（当前未验证）。
5. **提示词模板库**：把常用风格（海报 / 封面 / 产品图）沉淀成预设，界面上一键切换。
6. **两段式海报拼版**：上半照片、下半插画、50/50 拼合（§6.8 已给出可行性结论，尚未实现）。
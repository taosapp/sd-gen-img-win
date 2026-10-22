# 孔融让梨 · 工笔分镜（10 格）

- **画风**：工笔（铁线描 + 矿物颜料 + 绢本）
- **时代**：东汉，交领右衽、束发、庭院席地
- **画幅**：9:16（768×1344）
- **实测档位**：sdxl / fp16 / **balanced**（25 步）——每张 20–35 秒

## 为什么这么写（实测教训）

| 版本 | 写法 | 结果 |
|---|---|---|
| v1 | 风格块在前，人物在句尾 | 工笔对了，但「取梨」画成成年女子，纯物件镜头退化成花纹 |
| v2 | 人物在前、风格在尾 | 人物对了，**风格整个丢了**（提示词 77 词元被截断，风格块正好在尾部） |
| **v3** | **风格块前置且压缩到 10 词元**，正文 ≤28 词元 | 两者都保住 ✅ |

> 结论：CLIP 只有 77 词元。风格块必须放最前、且要短；详细描写放后面。
> 每条提示词都用 `tools/count_tokens.py` 量过，本组实测 **29–38 词元**。

## 固定块

**画风块（每条都放最前）**

```
gongbi painting, fine ink outlines, mineral pigments, silk scroll
```

**负向（每条相同）**

```
anime, manga, cartoon, photo, photorealistic, 3d render, oil painting,
perspective shading, japanese kimono, ukiyo-e, modern clothing, text, watermark,
adult woman, female portrait, textile pattern, extra limbs, deformed hands
```

## 十格提示词

| # | 画面 | 提示词正文（接在画风块之后） |
|---|---|---|
| 01 | 梨树 | `a pear tree full of yellow pears over a Han dynasty courtyard, two servants beneath it` |
| 02 | 摘梨 | `a servant lifting a basket of pears, a father in dark robe and three brothers watching` |
| 03 | 盛梨 | `a small boy in blue hanfu looks at a lacquer tray of pears of different sizes` |
| 04 | 环坐 | `a father and three brothers sit on mats in age order, the youngest boy sits last` |
| 05 | 父命 | `the father gestures to the pear tray, asking his youngest son to choose first` |
| 06 | 取小 | `a four-year-old boy in blue hanfu holds the smallest pear, his brother holds a big one` |
| 07 | 父问 | `the father leans down to question his small son, brothers listening quietly` |
| 08 | 作答 | `the small boy bows with folded hands, answering his father, brothers watching` |
| 09 | 兄惭 | `the three elder brothers look ashamed at the little boy, the father nods approval` |
| 10 | 分梨 | `the family sits together sharing pears, empty silk space above for a colophon` |

## 怎么跑

**方式一：网页故事板**（推荐）

1. 打开 `http://127.0.0.1:7860/?mode=storyboard`；
2. 画质挡位选 **balanced**（lightning 只有 4 步、CFG=1，风格与主体容易跑偏——实测把小孩画成成年女子）；
3. 画幅选 9:16；
4. 点「＋ 添加镜头」凑到 10 个；每条**开头先写画风块**（`gongbi painting, fine ink outlines, mineral pigments, silk scroll,`），再接上表正文。

**方式二：脚本**（自动带上画风块与负向）

```powershell
$env:GONGBI_GEAR="balanced"
python tools\exp_gongbi3.py
```

产物：`results/gongbi_v3/shots/shot01..10.png` 与 `storyboard.png` 拼图。

## 已知局限

1. **人物不连戏**：txt2img 每次重抽，父亲和孔融的脸/衣服每格会变。要真正连戏，需
   ①先用一版定角色三视图，②把该图作为 img2img 初始图（强度 0.45 左右）逐格生成；
2. **孩童识别不稳定**：写 `four-year-old boy` 仍可能画成少年/成人，负向里的
   `adult woman, female portrait` 只能压住一部分；
3. **题跋**：画面顶部会出现类文字装饰（符合立轴观感），但不是真汉字。

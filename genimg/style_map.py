"""风格语义展开：把"抽象风格名"翻译成 CLIP 能听懂的"具体视觉特征"。

为什么需要这一层
----------------
实测（sd15 / balanced / 同种子）：

    "..., American comic style."                 → 照片（风格被完全无视）
    "... + 负向 photo, photorealistic"            → 还是照片
    "American comic style, ..."（前置）           → 还是照片
    "American comic book illustration,
     bold black ink outlines, flat colors,
     halftone dots"                              → 漫画 ✅

CLIP 只认训练时见过的"词—画面"对应。`watercolor`、`cyberpunk` 有稳定视觉原型，
一写就出；`American comic` 没有，`comic` 还兼指"喜剧"，语义被稀释，模型退回照片。

所以这一层做的不是"中译英"（那是 zh_prompt.py 的事），而是
**抽象概念 → 可见的线条/配色/材质/光**。

触发规则
--------
只有当提示词里出现该风格名、且**没有**配套的具体描述时才展开，
避免把用户已经写好的细节再堆一遍（长提示词会被 CLIP 截断）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StyleEntry:
    key: str
    label: str
    pattern: re.Pattern
    expansion: str
    negative: str = ""
    # 提示词里已经出现这些词，就认为用户已写具体细节，不再展开
    already: tuple = ()


# 已实测生效的展开（comic 前后对照见 README §6.11）
STYLES: tuple[StyleEntry, ...] = (
    StyleEntry(
        "comic",
        "美式漫画 / American comic",
        re.compile(r"american\s+comic|comic\s+style|美式漫画|美漫", re.IGNORECASE),
        "American comic book illustration, bold black ink outlines, flat colors, halftone dots, dynamic pose",
        negative="photo, photorealistic, soft shading, gradient",
        already=("ink outline", "halftone", "flat color"),
    ),
    StyleEntry(
        "anime",
        "日式动漫 / anime",
        re.compile(r"\banime\b|日漫|日式动漫|二次元", re.IGNORECASE),
        "anime illustration, clean line art, cel shading, expressive eyes, pastel palette",
        negative="photo, photorealistic, 3d render",
        already=("cel shading", "line art"),
    ),
    StyleEntry(
        "watercolor",
        "水彩 / watercolor",
        re.compile(r"watercolou?r|水彩", re.IGNORECASE),
        "watercolor painting, soft translucent washes, wet-on-wet bleeding, visible paper texture, muted palette",
        negative="photo, 3d render, harsh outlines",
        already=("wash", "bleeding", "paper texture"),
    ),
    StyleEntry(
        "oil",
        "油画 / oil painting",
        re.compile(r"oil\s+paint|油画", re.IGNORECASE),
        "oil painting, thick impasto brushstrokes, canvas texture, rich saturated color, chiaroscuro lighting",
        negative="photo, flat color, vector",
        already=("impasto", "brushstroke"),
    ),
    StyleEntry(
        "sketch",
        "素描 / pencil sketch",
        re.compile(r"pencil\s+sketch|sketch|素描|铅笔", re.IGNORECASE),
        "graphite pencil drawing, cross hatching, monochrome, visible paper grain, loose construction lines",
        negative="color, photo, 3d render",
        already=("hatching", "graphite"),
    ),
    StyleEntry(
        "pixel",
        "像素画 / pixel art",
        re.compile(r"pixel\s+art|像素", re.IGNORECASE),
        "8-bit pixel art, limited palette, hard edges, dithering, sprite design",
        negative="photo, smooth gradient, 3d render",
        already=("8-bit", "dithering"),
    ),
    StyleEntry(
        "render3d",
        "3D 渲染 / 3d render",
        re.compile(r"3d\s+render|三维渲染|3d渲染", re.IGNORECASE),
        "3d render, subsurface scattering, ray traced reflections, studio lighting, glossy materials, octane render",
        negative="2d, sketch, flat illustration",
        already=("ray trac", "subsurface", "octane"),
    ),
    StyleEntry(
        "cyberpunk",
        "赛博朋克 / cyberpunk",
        re.compile(r"cyberpunk|赛博朋克", re.IGNORECASE),
        "cyberpunk, neon signs, wet asphalt reflections, magenta and cyan rim light, dense urban haze",
        negative="daylight, pastoral, vintage",
        already=("neon", "rim light"),
    ),
    StyleEntry(
        "steampunk",
        "蒸汽朋克 / steampunk",
        re.compile(r"steampunk|蒸汽朋克", re.IGNORECASE),
        "steampunk, brass and copper machinery, exposed gears, victorian clothing, warm sepia tones, steam",
        negative="modern, neon, plastic",
        already=("brass", "gear"),
    ),
    StyleEntry(
        "inkwash",
        "国风水墨 / chinese ink wash",
        re.compile(r"ink\s+wash|水墨|国画", re.IGNORECASE),
        "chinese ink wash painting, rice paper texture, minimal expressive brush strokes, large negative space, muted ink tones",
        negative="photo, neon, saturated color",
        already=("rice paper", "brush stroke"),
    ),
    StyleEntry(
        "flat",
        "扁平插画 / flat illustration",
        re.compile(r"flat\s+illustration|扁平", re.IGNORECASE),
        "flat vector illustration, solid color blocks, no gradients, geometric shapes, bold silhouettes",
        negative="photo, texture, gradient, 3d render",
        already=("vector", "color block"),
    ),
    StyleEntry(
        "pop",
        "波普 / pop art",
        re.compile(r"pop\s+art|波普", re.IGNORECASE),
        "pop art, bold primary colors, ben-day dots, thick outlines, high contrast graphic layout",
        negative="photo, muted palette, realistic shading",
        already=("ben-day", "primary color"),
    ),
    StyleEntry(
        "ghibli",
        "吉卜力 / studio ghibli",
        re.compile(r"ghibli|吉卜力", re.IGNORECASE),
        "studio ghibli style, hand-painted background, soft natural light, detailed foliage, warm nostalgic palette",
        negative="photo, 3d render, dark horror",
        already=("hand-painted", "foliage"),
    ),
    StyleEntry(
        "pixar",
        "皮克斯 / pixar 3d",
        re.compile(r"pixar|皮克斯", re.IGNORECASE),
        "3d animated film still, soft global illumination, rounded stylized shapes, expressive face, cinematic bokeh",
        negative="sketch, flat vector, photo",
        already=("global illumination", "bokeh"),
    ),
)


@dataclass
class ExpandResult:
    prompt: str
    applied: list  # [(label, expansion)]
    negatives: list  # 建议追加的负向

    @property
    def changed(self) -> bool:
        return bool(self.applied)


def expand_styles(text: str) -> ExpandResult:
    """把抽象风格名展开成具体视觉特征；已写细节的不动。"""
    prompt = text or ""
    applied: list[tuple[str, str]] = []
    negatives: list[str] = []
    low = prompt.lower()

    for st in STYLES:
        if not st.pattern.search(prompt):
            continue
        if any(token in low for token in st.already):
            continue
        # 追加到句末：CLIP 对开头权重更高，但用户的主体通常在前，保留其语序
        prompt = f"{prompt.rstrip(' ,.')}, {st.expansion}"
        applied.append((st.label, st.expansion))
        if st.negative:
            negatives.append(st.negative)
        low = prompt.lower()

    return ExpandResult(prompt=prompt, applied=applied, negatives=negatives)

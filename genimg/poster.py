"""两段式海报拼版：上半照片 + 下半水彩/手绘插画 + 居中排版。

为什么不用单次生成
------------------
实测那份 4942 字符的提示词有 835 个 token，而 CLIP 单次只吃 77 个，91% 被截断，
截断点正好落在「the bottom half occupies exactly 50%」之后——版式指令根本进不了模型。
所以把版式交给代码，把「画什么」交给模型：

1. 上半：照片按目标区域等比裁切（cover），一个像素都不拉伸；
2. 下半：同一张照片裁切后做 img2img（高强度），只让模型负责"把照片变成水彩画"；
3. 融合：用**正片叠底**把插画压到纸纹底上——白色留白自动变回纸色，不会出现贴图矩形；
4. 排版：衬线字体 + 字距拉开 + 细分隔线，位置由代码算。

提示词怎么压缩
--------------
- `watercolor` 预设：直接用一组实测有效的水彩描述（不依赖长文本）；
- `paper` 预设：从长提示词里抽「BOTTOM HALF」小节首句与「Use:」列表，「AVOID」转负向。
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# 英文里约 1.3 词 = 1 token；留余量，50 词 ≈ 65 token
STYLE_MAX_WORDS = 50

# 纸色（暖白）
PAPER_RGB = (246, 242, 233)

WATERCOLOR_STYLE = (
    "watercolor painting of the scene in the photograph, same composition and subjects, "
    "soft translucent washes, muted natural colors, loose feathered edges, minimal detail, "
    "on plain white paper, generous white margins, hand painted"
)

# 水彩描述必须带上「保住主体」的约束，否则高强度重绘会把熊猫画成窗子、把人画成鸟（已踩过）
WATERCOLOR_PREFIX = "watercolor painting, soft translucent washes, muted palette, "
WATERCOLOR_NEGATIVE = (
    "photo, photorealistic, 3d render, harsh outlines, neon colors, busy composition, "
    "text, watermark, frame, border, drop shadow"
)

_SERIF = r"C:\Windows\Fonts\georgia.ttf"


def _slice_between(text: str, start_pat: str, end_pat: str) -> str:
    """按**行首的大写小节标题**切段（必须区分大小写）。"""
    m = re.search(start_pat, text, re.MULTILINE)
    if not m:
        return ""
    rest = text[m.end() :]
    m2 = re.search(end_pat, rest, re.MULTILINE)
    return rest[: m2.start()] if m2 else rest


def condense_prompt(text: str) -> tuple[str, str]:
    """把长提示词压成 (画风正向, 负向)。抽不到就用通用兜底。"""
    body = _slice_between(
        text,
        r"^BOTTOM HALF\b[^\n]*$",
        r"^\s*(COLOR PALETTE|TYPOGRAPHY|VISUAL LANGUAGE)\b[^\n]*$",
    )
    # 先「首句」（点明风格），再「Use:」列表（具体质感）；Preserve 清单对画风贡献最小
    head = ""
    m = re.search(r"[^.\n]+\.[^\n]*", body)
    if m:
        head = m.group(0)
    use_block = _slice_between(body, r"^Use:\s*$", r"^The main illustrated subject\b")
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", f"{head} {use_block}")
    if len(words) < 10:
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", body)
    style = " ".join(words[:STYLE_MAX_WORDS]).strip()

    avoid = _slice_between(text, r"^AVOID\s*$", r"^\s*(FINAL ART DIRECTION)\b[^\n]*$")
    neg_lines = [
        ln.strip(" .")
        for ln in avoid.splitlines()
        if ln.strip() and not ln.strip().lower().startswith("do not")
    ]
    negative = ", ".join(neg_lines[:25])

    if not style:
        style = "minimalist hand-drawn paper illustration, flat color shapes, rough paper texture"
    if not negative:
        negative = "photograph, 3d render, glossy, busy composition"
    return style, negative


def cover_crop(im: Image.Image, width: int, height: int) -> Image.Image:
    """等比放大到铺满，居中裁切——只裁不拉。"""
    scale = max(width / im.width, height / im.height)
    new = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    left = (new.width - width) // 2
    top = (new.height - height) // 2
    return new.crop((left, top, left + width, top + height))


def paper_texture(width: int, height: int, seed: int = 7) -> Image.Image:
    """程序生成的纸纹：暖白底 + 细噪点 + 极淡的明暗起伏。"""
    random.seed(seed)
    base = Image.new("RGB", (width, height), PAPER_RGB)
    noise = Image.effect_noise((width, height), 14).convert("L").filter(ImageFilter.GaussianBlur(0.6))
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    # 噪点只占 6%，保持干净，只求"不是纯色块"
    return Image.blend(base, noise_rgb, 0.06)


def _letterspaced(draw: ImageDraw.ImageDraw, cx: int, y: int, text: str, font, fill, tracking: float):
    """居中 + 拉开字距（参考图那种编辑感排版）。"""
    gap = font.size * tracking
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + gap * (len(text) - 1)
    x = cx - total / 2
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += w + gap


def _fit(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def compose_bottom(
    art: Image.Image,
    *,
    width: int,
    height: int,
    art_scale: float,
    title: str,
    subtitle: str,
    year: str,
) -> Image.Image:
    """插画缩到 art_scale 后用正片叠底压在纸纹上（白区≈无色，无矩形边），再排版。"""
    paper = paper_texture(width, height)

    tw = max(32, int(width * art_scale))
    th = max(32, int(art.height * tw / art.width))
    small = art.resize((tw, th), Image.LANCZOS)
    # 降一点饱和，貼近水彩的克制配色
    small = ImageEnhance.Color(small).enhance(0.88)

    left = (width - tw) // 2
    top = int(height * 0.13)
    box = (left, top, left + tw, top + th)

    # 羽化蒙版：否则就是一张贴上去的矩形图，与纸面无关
    mask = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(mask).rectangle([0, 0, tw - 1, th - 1], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(4, int(tw * 0.035))))
    paper.paste(ImageChops.multiply(paper.crop(box), small), box, mask)

    draw = ImageDraw.Draw(paper)
    ink = (58, 54, 48)

    if title:
        f_title = _fit(_SERIF, max(16, int(width * 0.042)))
        f_sub = _fit(_SERIF, max(10, int(width * 0.019)))
        f_year = _fit(_SERIF, max(9, int(width * 0.017)))
        y = top + th + int(height * 0.06)
        _letterspaced(draw, width // 2, y, title.upper(), f_title, ink, 0.34)
        y += int(f_title.size * 1.9)
        rule_w = int(width * 0.06)
        draw.line([(width // 2 - rule_w // 2, y), (width // 2 + rule_w // 2, y)], fill=(150, 143, 130), width=1)
        y += int(f_title.size * 0.9)
        if subtitle:
            _letterspaced(draw, width // 2, y, subtitle.upper(), f_sub, (96, 90, 80), 0.32)
            y += int(f_sub.size * 2.1)
        if year:
            _letterspaced(draw, width // 2, y, year, f_year, (120, 113, 100), 0.36)
    return paper


@dataclass
class PosterResult:
    path: str
    width: int
    height: int
    style_prompt: str
    negative_prompt: str
    seconds: float
    illustration_path: str


def render(
    engine,
    *,
    photo_path: str,
    prompt_text: str,
    model,
    precision,
    quality,
    width: int,
    height: int,
    strength: float = 0.58,
    seed: int | None = None,
    style_override: str = "",
    preset: str = "watercolor",
    art_scale: float = 0.52,
    title: str = "",
    subtitle: str = "",
    year: str = "",
    out_dir: str,
) -> PosterResult:
    """生成一张海报。width × height 任意比例，上下各占一半。"""
    from .engine import GenRequest  # 延迟导入，避免工具脚本单独 import 时拉起 torch

    os.makedirs(out_dir, exist_ok=True)
    half_h = height // 2

    if style_override.strip():
        style, negative = style_override.strip(), WATERCOLOR_NEGATIVE
    elif preset == "watercolor":
        # 抽取的段落里带「Preserve the recognizable subject / silhouette / pose」这类约束，
        # 正好用来锁住主体；前面再加一句水彩定调。
        base, negative = condense_prompt(prompt_text)
        words = (WATERCOLOR_PREFIX + base).split()
        style = " ".join(words[:60])
        negative = f"{negative}, {WATERCOLOR_NEGATIVE}" if negative else WATERCOLOR_NEGATIVE
    else:
        style, negative = condense_prompt(prompt_text)

    photo = Image.open(photo_path).convert("RGB")
    top = cover_crop(photo, width, half_h)

    # 关键：用**满幅**照片做 img2img。把照片缩成小白底小图再生成，模型收到的视觉信号太弱，
    # 会照着提示词凭空画（熊猫会变成窗户、人会变成鸟，已实测两次）——缩小改到生成之后做。
    bottom_src = cover_crop(photo, width, half_h)
    src_path = os.path.join(out_dir, "_bottom_src.png")
    bottom_src.save(src_path)

    req = GenRequest(
        mode="img2img",
        prompt=style,
        negative_prompt=negative,
        model=model.key,
        precision=precision.key,
        gear=quality.key,
        width=width,
        height=half_h,
        steps=None,
        guidance=None,
        seed=seed,
        strength=strength,
        init_image=src_path,
        out_dir=out_dir,
        size_source="海报下半",
    )
    t0 = time.perf_counter()
    res = engine.run(req, model, precision, quality)
    if not res.ok:
        raise RuntimeError(f"下半生成失败：{res.error}")
    elapsed = time.perf_counter() - t0

    art = Image.open(res.images[0]).convert("RGB")
    bottom = compose_bottom(
        art,
        width=width,
        height=half_h,
        art_scale=art_scale,
        title=title,
        subtitle=subtitle,
        year=year,
    )

    canvas = Image.new("RGB", (width, height), PAPER_RGB)
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, half_h))

    out_path = os.path.join(
        out_dir, f"poster_{model.key}_{width}x{height}_s{seed if seed is not None else 'rand'}.png"
    )
    canvas.save(out_path)
    return PosterResult(
        path=out_path,
        width=width,
        height=height,
        style_prompt=style,
        negative_prompt=negative,
        seconds=elapsed,
        illustration_path=res.images[0],
    )

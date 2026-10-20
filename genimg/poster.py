"""两段式海报拼版：上半照片 + 下半插画，严格 50/50。

为什么不用单次生成
------------------
实测那份 4942 字符的提示词有 835 个 token，而 CLIP 单次只吃 77 个，91% 被截断，
截断点正好落在「the bottom half occupies exactly 50%」之后——版式指令根本进不了模型。
所以把版式交给代码，把「画什么」交给模型：

1. 上半：照片按目标区域等比裁切（cover），一个像素都不拉伸；
2. 下半：同一张照片裁切后做 img2img（高强度 0.85+），只让模型负责"把照片变成手绘插画"；
3. 拼接：两段等高，直接拼，缝在正中间。

提示词怎么压缩
--------------
从长提示词里自动抽出「BOTTOM HALF …」到「COLOR PALETTE」之间的段落作为画风描述，
再按词数截到 CLIP 能吃的长度；「AVOID」清单转成负向提示词。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from PIL import Image

# 英文里约 1.3 词 = 1 token；留点余量，50 词 ≈ 65 token
STYLE_MAX_WORDS = 50


def _slice_between(text: str, start_pat: str, end_pat: str) -> str:
    """按**行首的大写小节标题**切段。

    注意：必须区分大小写。提示词正文里有 "the bottom half occupies exactly 50%"
    这类小写句子，忽略大小写会把它们误当成标题（已踩过）。
    """
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
    # 先用「首句」（点明这是什么风格的插画），再用「Use:」列表（具体质感），
    # 「Preserve:」清单是主体约束，对画风贡献最小，放最后。
    head = ""
    m = re.search(r"[^.\n]+\.[^\n]*", body)
    if m:
        head = m.group(0)
    use_block = _slice_between(body, r"^Use:\s*$", r"^The main illustrated subject\b")
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", f"{head} {use_block}")
    if len(words) < 10:  # 抽不到就用整段
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", body)
    style = " ".join(words[:STYLE_MAX_WORDS]).strip()

    avoid = _slice_between(
        text,
        r"^AVOID\s*$",
        r"^\s*(FINAL ART DIRECTION)\b[^\n]*$",
    )
    neg_lines = [
        ln.strip(" .").strip()
        for ln in avoid.splitlines()
        if ln.strip() and not ln.strip().lower().startswith("do not")
    ]
    negative = ", ".join(neg_lines[:25]) if neg_lines else ""

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
    strength: float = 0.72,
    seed: int | None = None,
    style_override: str = "",
    art_scale: float = 0.62,
    paper_color: tuple[int, int, int] = (246, 243, 236),
    out_dir: str,
) -> PosterResult:
    """生成一张海报。width × height 必须是 3:4（或任意比例），上下各占一半。"""
    from .engine import GenRequest  # 延迟导入，避免工具脚本单独 import 时拉起 torch

    os.makedirs(out_dir, exist_ok=True)
    half_h = height // 2
    style, negative = condense_prompt(prompt_text)
    if style_override.strip():
        style = style_override.strip()

    photo = Image.open(photo_path).convert("RGB")
    top = cover_crop(photo, width, half_h)
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

    bottom = Image.open(res.images[0]).convert("RGB")
    if bottom.size != (width, half_h):
        bottom = bottom.resize((width, half_h), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(top, (0, 0))
    # 「主体小、大量留白」这条 img2img 做不到（它会保留原构图），用确定性的缩放置入补上：
    # 插入到纸色底版上，四周自然形成留白，版式不依赖模型自觉。
    if art_scale < 0.999:
        paper = Image.new("RGB", (width, half_h), paper_color)
        tw, th = max(32, int(width * art_scale)), max(32, int(half_h * art_scale))
        art = bottom.resize((tw, th), Image.LANCZOS)
        paper.paste(art, ((width - tw) // 2, (half_h - th) // 2))
        bottom = paper
    canvas.paste(bottom, (0, half_h))

    out_path = os.path.join(out_dir, f"poster_{model.key}_{width}x{height}_s{seed if seed is not None else 'rand'}.png")
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

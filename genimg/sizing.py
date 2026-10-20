"""目标尺寸解析：固定尺寸 / 比例（3:4）/ 跟随参考图，保证图生图不被拉伸变形。

为什么需要
----------
把一张 1024×683 的横图 ``resize((512, 512))`` 会横向压扁 1.5 倍——人脸、建筑全部变形。
正确做法是先决定**目标画布的宽高比**，再把参考图按比例装进去（等比缩放 + 留白），
而不是把参考图硬塞进固定方框。

三种输入
--------
- ``512x768``  固定像素，原样使用（会提示是否偏离原生面积）
- ``3:4`` / ``2:3`` / ``16:9``  比例，按模型原生面积自动选最接近的 64 倍数尺寸
- ``match`` / ``auto``  跟随参考图宽高比（图生图默认就是它）

尺寸桶怎么选
------------
在 64 的倍数里搜索，同时最小化"比例误差"和"面积偏离原生面积"，比例误差权重更高：
宁可比原生小一点，也不要出一个 0.78 的"假 3:4"。SD1.5 原生 512²，SDXL 原生 1024²。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from PIL import Image, ImageFilter

# 各模型的"原生训练分辨率"（短边基准）
NATIVE_BASE = {
    "sdxl": 1024,
    "sdxl-turbo": 1024,
    "sd15": 512,
    "sd-turbo": 512,
}

# 关键词 -> 比例
_RATIO_WORDS = {
    "竖版": (3, 4),
    "竖屏": (3, 4),
    "竖构图": (3, 4),
    "portrait": (3, 4),
    "vertical": (3, 4),
    "横版": (4, 3),
    "横屏": (4, 3),
    "横构图": (4, 3),
    "landscape": (4, 3),
    "horizontal": (4, 3),
    "方形": (1, 1),
    "正方形": (1, 1),
    "square": (1, 1),
    "宽屏": (16, 9),
    "超宽": (21, 9),
    "宽银幕": (21, 9),
    "cinematic": (16, 9),
}

# 显式比例：3:4 / 3：4 / 3 比 4（前后不能紧跟数字，避免命中 "2023:4" 之类）
_RATIO_RE = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*[:：比]\s*(\d{1,2}(?:\.\d+)?)(?!\d)")
# 固定像素：512x512 / 512×512 / 512*512
_WH_RE = re.compile(r"^\s*(\d{2,4})\s*[x×*]\s*(\d{2,4})\s*$")
# 比例字符串本身
_RATIO_SPEC_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d+)?)\s*[:：/]\s*(\d{1,2}(?:\.\d+)?)\s*$")


@dataclass
class SizeDecision:
    """一次尺寸决策的完整说明，用于回显与写入元数据。"""

    width: int
    height: int
    ratio: str = "1:1"
    source: str = "default"  # cli | prompt | 参考图 | default
    note: str = ""

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def describe(self) -> str:
        return f"{self.width}x{self.height}（{self.ratio}，来自{self.source}）"


def native_base_of(model_key: str, fallback: int = 512) -> int:
    return NATIVE_BASE.get(model_key, fallback)


def image_size(path: str) -> tuple[int, int] | None:
    """只读图片头拿尺寸，读失败返回 None（不抛异常）。"""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return None


def _fmt_ratio(rw: float, rh: float) -> str:
    """把 1.3333 这类比例还原成 4:3 这种可读写法。"""
    best, best_err = (1, 1), float("inf")
    for a, b in ((1, 1), (3, 4), (4, 3), (2, 3), (3, 2), (9, 16), (16, 9), (4, 5), (5, 4), (21, 9)):
        err = abs(math.log((a / b) / (rw / rh)))
        if err < best_err:
            best, best_err = (a, b), err
    return f"{best[0]}:{best[1]}"


def parse_spec(spec: str | None):
    """解析尺寸描述：('wh', w, h) / ('ratio', rw, rh) / ('match',) / None。"""
    if spec is None:
        return None
    raw = str(spec).strip()
    if not raw:
        return None
    m = _WH_RE.match(raw)
    if m:
        return ("wh", int(m.group(1)), int(m.group(2)))
    if raw.lower() in ("match", "auto", "参考图", "原图"):
        return ("match",)
    m = _RATIO_SPEC_RE.match(raw)
    if m:
        return ("ratio", float(m.group(1)), float(m.group(2)))
    return None


def size_for_ratio(rw: float, rh: float, base: int) -> tuple[int, int]:
    """给一个比例和基准分辨率，选最合适的 64 倍数尺寸。"""
    target_area = base * base
    wanted = rw / rh
    best: tuple[float, int, int] | None = None
    for w in range(256, 1537, 64):
        h = int(round((w * rh / rw) / 64)) * 64
        if h < 256 or h > 1536:
            continue
        ratio_err = abs(math.log((w / h) / wanted))
        area_err = abs(math.log((w * h) / target_area))
        # 超过原生面积惩罚更重：SD 系在面积大于训练分辨率时容易出重复主体
        if w * h > target_area:
            area_err *= 2.5
        score = ratio_err * 20.0 + area_err  # 比例压倒一切：宁可小，也不要"假 3:4"
        if best is None or score < best[0]:
            best = (score, w, h)
    if best is None:  # 极端比例（如 1:20）兜底
        return base, base
    return best[1], best[2]


def ratio_from_prompt(text: str) -> tuple[tuple[int, int], str] | None:
    """从提示词里找明确的画幅要求。先找 a:b 写法（按出现顺序取第一个），再找关键词。"""
    if not text:
        return None
    m = _RATIO_RE.search(text)
    if m:
        rw, rh = float(m.group(1)), float(m.group(2))
        if 0.2 <= rw / rh <= 5.0:
            return (rw, rh), m.group(0).strip()
    low = text.lower()
    for word, ratio in _RATIO_WORDS.items():
        if word in low:
            return (float(ratio[0]), float(ratio[1])), word
    return None


def ratio_from_size(size: tuple[int, int]) -> tuple[float, float]:
    w, h = size
    return float(w), float(h)


def _snap8(value: int) -> int:
    return max(8, int(round(value / 8)) * 8)


def resolve(
    *,
    width: int | None = None,
    height: int | None = None,
    spec: str | None = None,
    prompt: str = "",
    model_key: str = "sd-turbo",
    init_size: tuple[int, int] | None = None,
    model_native: int | None = None,
) -> SizeDecision:
    """按优先级决定最终尺寸。

    优先级：显式宽高 > spec（WxH 或比例）> 提示词里的画幅要求 > 参考图比例 > 模型原生 1:1。
    """
    base = model_native or native_base_of(model_key)
    notes: list[str] = []

    # 只给了一边（例如只有 --width）：无法确定比例，退回后面的规则
    if (width and not height) or (height and not width):
        notes.append("只指定了宽或高，无法确定比例，已忽略该值")
        width = height = None

    # 1. 显式宽高（--width/--height 或界面里直接填）
    if width and height:
        w, h = _snap8(int(width)), _snap8(int(height))
        if (w, h) != (int(width), int(height)):
            notes.append(f"已对齐到 8 的倍数：{int(width)}x{int(height)} → {w}x{h}")
        area_ratio = (w * h) / (base * base)
        if area_ratio > 1.45:
            notes.append(f"面积是原生 {area_ratio:.2f} 倍，容易出现重复主体，建议降到 {int(base * 0.9)}x 级别")
        return SizeDecision(w, h, _fmt_ratio(w, h), "cli", "；".join(notes))

    # 2. spec：WxH / 比例 / match
    parsed = parse_spec(spec)
    if parsed is not None:
        kind = parsed[0]
        if kind == "wh":
            w, h = _snap8(parsed[1]), _snap8(parsed[2])
            return SizeDecision(w, h, _fmt_ratio(w, h), "cli", "")
        if kind == "match":
            if init_size:
                rw, rh = ratio_from_size(init_size)
                w, h = size_for_ratio(rw, rh, base)
                return SizeDecision(w, h, _fmt_ratio(rw, rh), "参考图", f"参考图 {init_size[0]}x{init_size[1]}")
            notes.append("指定了 match 但没有参考图，退回模型原生比例")
        else:  # ratio
            rw, rh = parsed[1], parsed[2]
            w, h = size_for_ratio(rw, rh, base)
            if abs((w / h) - (rw / rh)) > 1e-6:
                notes.append(f"64 倍数里没有完全精确的 {rw:g}:{rh:g}，实际 {w}x{h}={w / h:.3f}")
            return SizeDecision(w, h, _fmt_ratio(rw, rh), "cli", "；".join(notes))

    # 3. 提示词里写明的画幅（例如 "Strict 3:4 vertical composition"）
    hit = ratio_from_prompt(prompt)
    if hit:
        (rw, rh), matched = hit
        w, h = size_for_ratio(rw, rh, base)
        return SizeDecision(w, h, _fmt_ratio(rw, rh), f"提示词「{matched}」", "")

    # 4. 图生图：跟随参考图
    if init_size:
        rw, rh = ratio_from_size(init_size)
        w, h = size_for_ratio(rw, rh, base)
        return SizeDecision(w, h, _fmt_ratio(rw, rh), "参考图", f"参考图 {init_size[0]}x{init_size[1]}")

    # 5. 默认 1:1 原生
    return SizeDecision(base, base, "1:1", "default", "")


def fit_image(im: Image.Image, width: int, height: int, mode: str = "pad") -> tuple[Image.Image, str]:
    """把参考图装进目标画布：等比缩放 + 留白填充（默认），绝不拉伸。

    mode:
      - ``pad``   等比缩放到完整可见，四周用自身放大模糊图填满（相当于"延展背景"的起点）
      - ``crop``  等比放大到铺满，居中裁掉多余部分
    """
    if im.size == (width, height):
        return im, "原尺寸，无需调整"

    src_w, src_h = im.size
    if mode == "crop":
        scale = max(width / src_w, height / src_h)
        new = im.resize((max(1, round(src_w * scale)), max(1, round(src_h * scale))), Image.LANCZOS)
        left = (new.width - width) // 2
        top = (new.height - height) // 2
        return new.crop((left, top, left + width, top + height)), f"等比裁切（{scale:.2f}×，裁掉边缘）"

    # 背景：整图铺满后高斯模糊，避免出现黑边/白边
    bg = im.resize((width, height), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=max(8, max(width, height) // 40)))
    scale = min(width / src_w, height / src_h)
    fg = im.resize((max(1, round(src_w * scale)), max(1, round(src_h * scale))), Image.LANCZOS)
    canvas = bg.copy()
    canvas.paste(fg, ((width - fg.width) // 2, (height - fg.height) // 2))
    return canvas, f"等比缩放 {scale:.2f}× + 模糊留白（不拉伸）"

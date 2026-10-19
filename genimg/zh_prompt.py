"""中文提示词支持：把中文（尤其风格/画质词）转换成 SD 模型能听懂的英文。

为什么需要
----------
sd15 / sdxl / sd-turbo 的文本编码器都是 CLIP（英文语料为主），中文提示词经过分词后
大多是无效 token。实测：同参数同种子，"把图片改为水彩风格，细节丰富" 得到的是糊在一起的
色块；换成 `watercolor painting style, soft washes, paper texture` 才出纸质感。

为什么只做离线词典
------------------
本项目只内置**离线词典翻译**：不联网、不需要任何外部服务、不涉及任何凭据，
启动零延迟、结果可预期。词典覆盖风格、画质、光影、构图、颜色与常见场景（当前 116 条），
足以应付"把图片改成水彩/油画/赛博朋克风，细节丰富"这类常见用法。

词典覆盖不到的自由描述（如"一只戴墨镜的熊猫在月球上喝咖啡"）会被**原样保留**并在结果里
提示出来 —— 宁可少翻，也不静默丢主体。真要覆盖这类描述，正确做法是换中文原生底模
（Taiyi / AltDiffusion 等），而不是在提示词层做机器翻译。

安全约定
--------
词典没覆盖到的中文**不会丢弃**，而是原样保留在提示词里并在返回值里列出，
否则"水彩风格的熊猫"会被翻成只剩 `watercolor painting style`，主体就丢了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SPLIT_RE = re.compile(r"[，,。；;、\n]+")
_PUNCT_STRIP = "，,。；;、 \t\n\r\"'“”‘’"

# 指令性废话：对 CLIP 没意义，直接剔除
_INSTRUCTIONS = [
    "把这张图片改为", "把这张图改为", "把这张图片变成", "把这张图变成",
    "把图片改为", "把图片改成", "把图片变成", "把图改为", "把图改成",
    "将图片改为", "将图片改成", "将图片变成", "将这张图改为", "将这张图变成",
    "风格转换为", "转换为", "改成", "改为", "变成", "调整为",
    "请帮我", "帮我", "请", "生成一张", "画一张", "来一张", "给我一张",
    "要求", "需要", "希望", "图片", "这张图", "使其", "让它",
]

# 中文 -> 英文（按长度倒序匹配，长词优先）
_GLOSSARY_RAW = {
    # 风格 / 媒介
    "水彩画": "watercolor painting",
    "水彩风格": "watercolor painting style",
    "水彩": "watercolor painting",
    "水墨画": "chinese ink wash painting",
    "水墨风格": "chinese ink wash painting style",
    "水墨": "ink wash painting",
    "国画": "traditional chinese painting",
    "工笔": "gongbi fine brushwork painting",
    "写意": "freehand brushwork painting",
    "油画": "oil painting",
    "厚涂": "thick impasto painting",
    "丙烯": "acrylic painting",
    "粉彩": "pastel drawing",
    "素描": "pencil sketch",
    "速写": "quick sketch",
    "线稿": "line art",
    "插画": "illustration",
    "儿童插画": "children's book illustration",
    "绘本": "picture book illustration",
    "概念艺术": "concept art",
    "概念设计": "concept design art",
    "卡通": "cartoon style",
    "动漫": "anime style",
    "二次元": "anime style",
    "漫画": "comic book style",
    "像素风": "pixel art",
    "低多边形": "low poly",
    "3d渲染": "3d render",
    "三维渲染": "3d render",
    "赛博朋克": "cyberpunk",
    "蒸汽朋克": "steampunk",
    "蒸汽波": "vaporwave",
    "波普": "pop art",
    "涂鸦": "graffiti art",
    "版画": "printmaking",
    "壁画": "mural",
    "印象派": "impressionism",
    "超现实": "surrealism",
    "极简": "minimalist",
    "复古": "retro",
    "未来感": "futuristic",
    "手绘": "hand-drawn",
    "摄影": "photography",
    "胶片": "film photography",
    "电影感": "cinematic",
    "黑白": "black and white",
    "单色": "monochrome",
    "彩色": "colorful",
    # 画质 / 细节
    "细节丰富": "rich details",
    "细节": "detailed",
    "高清": "high resolution",
    "高分辨率": "high resolution",
    "超清": "ultra detailed",
    "精致": "exquisite",
    "清晰": "sharp focus",
    "质感": "texture",
    "逼真": "photorealistic",
    "写实": "realistic",
    "大师作品": "masterpiece",
    "获奖作品": "award winning",
    "最佳画质": "best quality",
    "杰作": "masterpiece",
    "8k": "8k",
    # 光影
    "柔和光线": "soft lighting",
    "柔光": "soft lighting",
    "逆光": "backlight",
    "侧光": "side lighting",
    "顶光": "top lighting",
    "工作室灯光": "studio lighting",
    "电影级灯光": "cinematic lighting",
    "霓虹灯": "neon lights",
    "体积光": "volumetric lighting",
    "光影": "lighting and shadow",
    # 构图 / 镜头
    "特写": "close-up",
    "半身": "half body",
    "全身": "full body",
    "正面": "front view",
    "侧面": "side view",
    "俯视": "top view",
    "仰视": "low angle view",
    "广角": "wide angle",
    "微距": "macro shot",
    "景深": "depth of field",
    "背景虚化": "bokeh",
    "留白": "negative space",
    "对称构图": "symmetrical composition",
    # 色彩
    "暖色调": "warm color tones",
    "冷色调": "cool color tones",
    "高对比": "high contrast",
    "莫兰迪色": "morandi color palette",
    "柔和色彩": "soft colors",
    "鲜艳": "vibrant colors",
    # 通用
    "风格": "style",
    "背景": "background",
    "纹理": "texture",
    # 常见主体 / 场景（能覆盖一点是一点，避免主体词全被归为“未识别”）
    "山水画": "landscape painting",
    "山水": "landscape",
    "风景": "landscape",
    "远山": "distant mountains",
    "淡墨": "light ink wash",
    "近景": "foreground",
    "肖像": "portrait",
    "人物": "character",
    "静物": "still life",
    "建筑": "architecture",
    "动物": "animal",
    "花卉": "flowers",
    "星空": "starry sky",
    "夜景": "night scene",
    "日出": "sunrise",
    "日落": "sunset",
    "森林": "forest",
    "城市": "city",
    "街景": "street scene",
    "室内": "indoor",
    "室外": "outdoor",
}

_GLOSSARY = sorted(_GLOSSARY_RAW.items(), key=lambda kv: len(kv[0]), reverse=True)
_INSTRUCTIONS_SORTED = sorted(_INSTRUCTIONS, key=len, reverse=True)


@dataclass
class TranslationResult:
    """一次提示词翻译的结果。"""

    original: str
    effective: str
    method: str = "none"  # none | dict
    replaced: list = field(default_factory=list)  # [(中文, english)]
    unknown: list = field(default_factory=list)  # 词典没覆盖、原样保留的片段
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.method != "none" and self.effective.strip() != self.original.strip()

    def summary(self) -> str:
        if self.method == "none":
            return "（无需翻译）"
        parts = [f"方式={self.method}", f"命中 {len(self.replaced)} 个词"]
        if self.unknown:
            parts.append(f"未识别 {len(self.unknown)} 段（已原样保留）")
        return "，".join(parts)


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _strip_instructions(segment: str) -> tuple[str, bool]:
    """剔除指令性措辞，返回 (剩余内容, 是否整句都是指令)。"""
    out = segment
    for phrase in _INSTRUCTIONS_SORTED:
        out = out.replace(phrase, " ")
    out = out.strip(_PUNCT_STRIP)
    return out, (out == "" and segment.strip() != "")


def translate_with_dict(text: str) -> TranslationResult:
    """离线词典翻译：只改它认识的词，不认识的段原样保留。"""
    original = text or ""
    if not contains_cjk(original):
        return TranslationResult(original=original, effective=original)

    pieces: list[str] = []
    replaced: list[tuple[str, str]] = []
    unknown: list[str] = []

    for raw_seg in _SPLIT_RE.split(original):
        seg = raw_seg.strip(_PUNCT_STRIP)
        if not seg:
            continue
        rest, only_instruction = _strip_instructions(seg)
        if only_instruction or not rest:
            continue

        translated_here = False
        for zh, en in _GLOSSARY:
            if zh in rest:
                # 孤立的 style/texture 合并到前一个词：
                # "oil painting" + "style" -> "oil painting style"（比 "oil painting, style" 更地道）
                if en in ("style", "texture") and pieces:
                    pieces[-1] = f"{pieces[-1]} {en}"
                else:
                    pieces.append(en)
                replaced.append((zh, en))
                rest = rest.replace(zh, " ")
                translated_here = True
        rest = rest.strip(_PUNCT_STRIP).strip()
        if rest and _CJK_RE.search(rest):
            # 没翻完的中文原样塞回去，避免"主体描述"被静默丢掉
            pieces.append(rest)
            unknown.append(rest)
        elif rest and translated_here:
            # 残留的英文/数字（例如 8k）保留
            pieces.append(rest)

    # 去重但保持顺序
    seen: set[str] = set()
    ordered: list[str] = []
    for p in pieces:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(p)

    effective = ", ".join(ordered).strip()
    if not effective:
        return TranslationResult(
            original=original,
            effective=original,
            method="none",
            note="词典没有识别到可翻译的词，已保留原文（可改用英文，或换中文原生底模）",
        )

    note = ""
    if unknown:
        note = "以下中文片段词典未覆盖，已原样保留：" + "、".join(unknown[:5])
    return TranslationResult(
        original=original,
        effective=effective,
        method="dict",
        replaced=replaced,
        unknown=unknown,
        note=note,
    )


def translate(text: str, mode: str = "dict") -> TranslationResult:
    """统一入口。mode: dict（默认，离线词典） | off（不翻译，原样使用）。"""
    original = text or ""
    if mode == "off" or not contains_cjk(original):
        return TranslationResult(original=original, effective=original)
    return translate_with_dict(original)

"""提示词内嵌指令解析。

让"在提示词里标明挡位"成为一等公民：提示词中任何形如 ``[精度=fp16]`` / ``[gear=max]`` /
``[size=640x640]`` 的片段都会被识别为指令并从提示词正文中剔除。

安全性约定
----------
**只剔除能识别的键**。形如 ``[blurry]``、``[cat:dog:0.5]`` 这类 A1111 风格的原生提示词语法
不在识别范围内，会被原样保留，避免解析器擅自改写用户提示词。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 值里允许出现除了方括号以外的任意字符（负向提示词常含逗号与空格）
_TOKEN_RE = re.compile(r"\[\s*([^\[\]=:]+?)\s*[=:]\s*([^\[\]]*?)\s*\]")

# 键别名 -> 规范键
KEY_ALIASES: dict[str, str] = {}


def _register(canonical: str, *names: str) -> None:
    for n in (canonical, *names):
        KEY_ALIASES[n.strip().lower()] = canonical


_register("precision", "精度", "精度挡位", "档位")
_register("gear", "quality", "画质", "画质挡位", "质量")
_register("size", "尺寸", "分辨率", "res")
_register("model", "模型", "底模")
_register("steps", "步数", "采样步数")
_register("guidance", "cfg", "引导", "引导强度")
_register("seed", "种子", "随机种子")
_register("strength", "强度", "改动强度")
_register("negative", "negative_prompt", "负向", "负面", "反向提示词")
_register("batch", "批次", "数量", "n")
_register("variant", "变体", "权重变体")
_register("placement", "放置", "显存策略", "offload")
_register("vae_dtype", "vae")
_register("hf_source", "源", "下载源")


# 需要转成整数的键
_INT_KEYS = {"steps", "seed", "batch"}
# 需要转成浮点的键
_FLOAT_KEYS = {"guidance", "strength"}


@dataclass
class PromptDirectives:
    """解析结果。"""

    prompt: str  # 剔除指令后的提示词正文
    raw_prompt: str  # 原始输入
    values: dict = field(default_factory=dict)  # 规范键 -> 解析后的值
    tokens: list = field(default_factory=list)  # 命中的原始片段，便于回显
    warnings: list = field(default_factory=list)  # 解析告警（值非法等）

    def as_cli_overrides(self) -> dict:
        """转换成 CLI 参数名 -> 值（与 argparse dest 对齐）。"""
        return dict(self.values)


def _parse_size(value: str):
    m = re.fullmatch(r"\s*(\d{2,4})\s*[x×*]\s*(\d{2,4})\s*", value)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_directives(text: str) -> PromptDirectives:
    """从提示词里抽取指令；返回清洗后的提示词与指令字典。"""
    result = PromptDirectives(prompt=text, raw_prompt=text)
    spans: list[tuple[int, int]] = []

    for m in _TOKEN_RE.finditer(text):
        key_raw, value_raw = m.group(1), m.group(2)
        canonical = KEY_ALIASES.get(key_raw.strip().lower())
        if canonical is None:
            # 不是已知指令：大概率是 A1111 原生语法，原样保留
            continue
        if value_raw == "":
            result.warnings.append(f"指令 [{m.group(0)[1:-1]}] 缺少取值，已忽略")
            continue

        value: object = value_raw
        try:
            if canonical == "size":
                value = _parse_size(value_raw)
                if value is None:
                    result.warnings.append(f"尺寸格式无法识别：{value_raw!r}（应形如 512x512），已忽略")
                    continue
            elif canonical in _INT_KEYS:
                value = int(float(value_raw))
            elif canonical in _FLOAT_KEYS:
                value = float(value_raw)
        except ValueError:
            result.warnings.append(f"指令 {canonical} 的取值 {value_raw!r} 无法解析为数字，已忽略")
            continue

        if canonical in result.values and result.values[canonical] != value:
            result.warnings.append(f"指令 {canonical} 重复出现，以最后一次 {value_raw!r} 为准")

        result.values[canonical] = value
        result.tokens.append(m.group(0))
        spans.append(m.span())

    # 从后往前删除，避免位移
    cleaned = text
    for start, end in sorted(spans, reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]

    result.prompt = re.sub(r"\s{2,}", " ", cleaned).strip().strip(",").strip()
    return result


def describe(directives: PromptDirectives) -> str:
    """人类可读的指令摘要。"""
    if not directives.tokens:
        return "（提示词中未内嵌指令）"

    def render(key: str, value: object) -> str:
        if key == "size" and isinstance(value, tuple):
            return f"size={value[0]}x{value[1]}"
        return f"{key}={value}"

    parts = [render(k, v) for k, v in directives.values.items()]
    return "提示词内嵌指令：" + "，".join(parts)

"""生成前的"体检"：把容易踩的坑用一句人话提示出来。

这些规则都来自本机实测，不是通用教条：

1. **图生图的实际采样步数是 int(步数 × 强度)**，不是步数本身。
   sd-turbo 的 turbo 档只有 2 步，强度 0.6 → 实际 1 步，风格改造基本无效（实测只像磨皮）。
2. **CFG ≤ 1 的挡位（turbo / lightning）提示词约束很弱**，负向提示词直接失效。
3. **中文提示词效果明显弱于英文**：SD 系列的 CLIP 文本编码器以英文语料训练，
   同一组参数下中文写"水彩风格"出来的只是色块涂抹，英文写 watercolor 才出纸质感（实测对照）。
4. **强度 0.5 能保住主体，0.6 起主体开始明显变化，0.7 以上基本等于重画**（实测对照，见 README §6.6）。
"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# CLIP 文本编码器单次最多 77 个 token（SDXL 的两个编码器各自也是 77）
CLIP_TOKEN_LIMIT = 77


# 实测（sd15 / balanced / 同种子）：
#   watercolor style  → 生效
#   cyberpunk style   → 生效
#   American comic style → **完全被无视**，出来的还是照片
# 区不在“风格名”本身，而在这个名字有没有对应的具体视觉特征。
# 把模糊的风格名换成具体画面元素，立刻出效果。
WEAK_STYLES = [
    (
        "American comic style",
        re.compile(r"american\s+comic|comic\s+style", re.IGNORECASE),
        "American comic book illustration, bold black ink outlines, flat colors, halftone dots",
    ),
]


def style_tips(prompt: str) -> list[str]:
    """风格名太模糊时给出实测有效的替代写法。"""
    text = prompt or ""
    low = text.lower()
    tips: list[str] = []
    for label, pattern, expanded in WEAK_STYLES:
        if pattern.search(text) and "ink outline" not in low and "halftone" not in low:
            tips.append(
                f"「{label}」在 CLIP 里权重很弱：实测同一句提示词里它会被完全无视（出来的还是照片），"
                f"加负向提示词也压不住。改成具体视觉特征立刻生效，例如：{expanded}"
            )
    return tips


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 1 字 1 token，英文约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    latin = len(text) - cjk
    return int(cjk + latin / 4.0)


def effective_steps(steps: int, strength: float) -> int:
    """diffusers 图生图实际执行的采样步数。"""
    return int(steps * strength)


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def check_request(
    *,
    mode: str,
    prompt: str,
    negative_prompt: str = "",
    steps: int,
    strength: float = 0.5,
    guidance: float = 7.0,
    model_key: str = "",
    skip_cjk_tip: bool = False,
) -> list[str]:
    """返回需要提醒用户的问题列表（空列表 = 没发现问题）。

    ``skip_cjk_tip``：调用方已确认中文会被自动翻译时置 True，避免“已转英文”与
    “建议改英文”两条提示同时出现。
    """
    tips: list[str] = []

    if mode == "img2img":
        eff = effective_steps(steps, strength)
        if eff < 1:
            tips.append(
                f"图生图的实际采样步数是 0（{steps} 步 × 强度 {strength}）。"
                f"请提高步数或强度，否则无法出图。"
            )
        elif eff < 3:
            tips.append(
                f"图生图实际只采样 {eff} 步（{steps} 步 × 强度 {strength}）。"
                f"步数太少时模型只是把原图重涂一层，做不出明确风格。"
                f"建议用 balanced / quality 档，强度 0.5–0.65。"
            )
        if strength >= 0.65:
            tips.append(
                f"改动强度 {strength} 偏大：实测 0.5 能保住原图主体，0.6 起主体开始明显变化，"
                f"0.7 以上基本等于按提示词重画。想要强风格化可以忽略这条。"
            )

    if guidance <= 1.0:
        tips.append(
            f"当前挡位 CFG={guidance}，等于关闭引导：提示词和负向提示词几乎不起作用。"
            f"做风格改造请选带 CFG 的挡位（balanced / quality / max）。"
        )

    if has_cjk(prompt) and not skip_cjk_tip:
        tips.append(
            "提示词含中文：SD 系列（sd15 / sdxl / sd-turbo）的文本编码器以英文语料为主，"
            "中文提示词的效果明显弱于英文。建议改成英文，例如「水彩风格」→ "
            "watercolor painting style, soft washes, paper texture, rich details。"
        )

    n_tok = estimate_tokens(prompt)
    if n_tok > CLIP_TOKEN_LIMIT:
        kept_pct = int(CLIP_TOKEN_LIMIT / n_tok * 100)
        tips.append(
            f"提示词约 {n_tok} 个 token，而 CLIP 单次只吃 {CLIP_TOKEN_LIMIT} 个："
            f"约 {100 - kept_pct}% 的内容会被静默截断（后面的“AVOID/风格细节”这类尾巴几乎全丢）。"
            f"对策：把最关键的 10–15 个词放最前面；长描述拆成多次生成再挑。"
        )

    tips.extend(style_tips(prompt))

    return tips

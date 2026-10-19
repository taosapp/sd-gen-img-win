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

    return tips

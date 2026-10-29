"""生成前的"体检"：把容易踩的坑用一句人话提示出来。

这些规则来自本机实测与公开规范借鉴，不是通用教条：

1. **图生图的实际采样步数是 int(步数 × 强度)**，不是步数本身。
   sd-turbo 的 turbo 档只有 2 步，强度 0.6 → 实际 1 步，风格改造基本无效（实测只像磨皮）。
2. **CFG ≤ 1 的挡位（turbo / lightning）提示词约束很弱**，负向提示词直接失效。
3. **中文提示词效果明显弱于英文**：SD 系列的 CLIP 文本编码器以英文语料训练，
   同一组参数下中文写"水彩风格"出来的只是色块涂抹，英文写 watercolor 才出纸质感（实测对照）。
4. **强度 0.5 能保住主体，0.6 起主体开始明显变化，0.7 以上基本等于重画**（实测对照，见 README §6.6）。
5. **要素完整度 / 空泛画质词 / 图内文字**：借鉴 Qwen-Image-2.1 官方提示词重写规范
   （经 iamyoki/qwen-image-2.1-skill 转述，Apache-2.0）。原规范面向长上下文文本编码器
   （400–500 词观察者长文），这里压缩成 CLIP 77-token 预算下的要素清单与 booster 检查，
   取舍见 README §3.7。
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


_CN_RE = re.compile(r"\bchinese\b|中国人|中式|华裔|亚洲人像", re.IGNORECASE)
_PORTRAIT_RE = re.compile(r"\b(girl|woman|man|portrait|face|headshot)\b|人像|女孩|女人|男人|肖像", re.IGNORECASE)
# 已经写了这些的，就不再啰嗝
_AGE_OR_PHOTO_RE = re.compile(
    r"\b\d{2}\s*(?:year|yr|years\s+old)\b|documentary|film grain|natural skin|skin texture|纪实|街拍",
    re.IGNORECASE,
)


def ethnicity_tips(prompt: str, negative_prompt: str = "") -> list[str]:
    """「Chinese + 人像」会被 SDXL 的少女偶像先验带跑，给出实测有效的纠正写法。"""
    text = prompt or ""
    if not (_CN_RE.search(text) and _PORTRAIT_RE.search(text)):
        return []
    if _AGE_OR_PHOTO_RE.search(text) or "idol" in (negative_prompt or "").lower():
        return []
    return [
        "「Chinese + 人像」在 SDXL 里有一个很强的「少女偶像」先验（大眼 / 小脸 / 无瑕皮肤），"
        "看起来就会偏日韩。实测有效的三步：① 加年龄（如 45 year old）——最有效；"
        "② 加纪实约束（documentary photograph, natural skin texture, minimal makeup）；"
        "③ 负向压偶像审美（k-pop idol, korean beauty standard, anime, plastic surgery, big eyes, slim jaw, airbrushed skin）。"
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


# ---------------------------------------------------------------------------
# 提示词结构体检（借鉴 Qwen-Image-2.1 官方提示词重写规范，压缩适配 CLIP 77 token）
#
# 原规范的“八步法”产出 400–500 词长文，Qwen 的文本编码器吃得下，CLIP 吃不下
# （77 token，超出部分静默丢弃）。这里只保留它的要素清单思想：
#   媒介/风格 + 主体(用户自己写) + 光影 + 镜头/构图 + 色调/氛围，每项 3–8 token。
# 正则是中英双语的：命令行路径拿到的是翻译+风格展开后的英文，网页 /api/advice
# 拿到的是用户原文，两边都要能识别。
# ---------------------------------------------------------------------------

# 空泛画质词（booster）：不描述任何画面内容，只消耗 token 预算。原规范明令禁用。
_BOOSTER_RE = re.compile(
    r"\b(?:masterpiece|best\s+quality|top\s+quality|award[\s-]*winning|8k|16k|4k|"
    r"ultra\s+detailed|highly\s+detailed|high\s+resolution|trending\s+on\s+artstation)\b"
    r"|大师作品|杰作|获奖作品|最佳画质",
    re.IGNORECASE,
)

# 媒介/风格：一句话里最好有一个“这是哪类画/照片”
_MEDIUM_RE = re.compile(
    r"\b(?:photo|photograph|photography|painting|illustration|sketch|drawing|render|"
    r"anime|cartoon|cinematic|poster|watercolou?r|oil\s+paint|ink\s+wash|pixel|vector|"
    r"photorealistic|realistic|surreal|minimalist|concept\s+art|artwork|art|cyberpunk|"
    r"steampunk)\b"
    r"|赛博朋克|蒸汽朋克|照片|摄影|胶片|插画|素描|线稿|油画|水彩|水墨|国画|卡通|动漫|二次元"
    r"|漫画|像素|写意|工笔|版画|壁画|风格|写实|逼真|电影感|海报|概念艺术|极简|复古|手绘|渲染",
    re.IGNORECASE,
)

# 光影
_LIGHT_RE = re.compile(
    r"\b(?:light|lighting|lit|backlit|backlight|neon|glow|luminous|shadow|sunlight|"
    r"sunlit|sunset|sunrise|golden\s+hour|dusk|dawn|moonlight|moonlit|candlelight|"
    r"overcast|volumetric|god\s+rays?|spotlight|rim\s+light|silhouette|softbox|studio|"
    r"twilight|chiaroscuro)\b"
    r"|柔光|暖光|冷光|逆光|侧光|顶光|灯光|光线|光影|光照|霓虹|烛光|月光|晨光|黄昏|剪影"
    r"|轮廓光|体积光|丁达尔|日出|日落|夜景|星空|反光|舞台光|光斑",
    re.IGNORECASE,
)

# 镜头/构图（含“主体在画面哪里”的方位词）
_CAMERA_RE = re.compile(
    r"\b(?:close[\s-]*up|closeup|wide\s+angle|macro|fisheye|aerial|drone|top[\s-]*down|"
    r"bird'?s?\s+eye|low\s+angle|high\s+angle|full\s+body|half\s+body|long\s+shot|"
    r"medium\s+shot|headshot|pov|from\s+above|from\s+below|symmetr\w*|negative\s+space|"
    r"depth\s+of\s+field|bokeh|foreground|background|vanishing\s+point|panorama|"
    r"panoramic|tilt[\s-]*shift|dutch\s+angle)\b"
    r"|特写|半身|全身|正面|侧面|俯视|仰视|俯拍|仰拍|鸟瞰|航拍|广角|超广角|微距|鱼眼|景深"
    r"|背景虚化|留白|对称构图|近景|远景|全景",
    re.IGNORECASE,
)

# 色调/氛围
_PALETTE_RE = re.compile(
    r"\b(?:warm|warmth|cool|vibrant|muted|pastel|monochrome|greyscale|grayscale|"
    r"black\s+and\s+white|high\s+contrast|low\s+contrast|morandi|palette|saturated|"
    r"desaturated|sepia|teal|colorful|colou?rs?|tones?|moody|serene|melancholic|cozy|"
    r"earthy|fluorescent)\b"
    r"|暖色调|冷色调|暖色|冷色|高对比|莫兰迪|柔和色彩|鲜艳|黑白|单色|彩色|大地色|马卡龙"
    r"|荧光色|色调|配色",
    re.IGNORECASE,
)

# 要素清单：名称 / 识别正则 / 缺失时的示例（全部是词典已收录的词，中文写会自动转英文）
_STRUCT_ELEMENTS = (
    ("媒介/风格", _MEDIUM_RE, "「水彩风格」「电影感」或 photograph, flat illustration"),
    ("光影", _LIGHT_RE, "「暖光」「轮廓光」「电影级灯光」或 golden-hour light, rim lighting"),
    ("镜头/构图", _CAMERA_RE, "「特写」「广角」「俯视」或 close-up, wide angle"),
    ("色调/氛围", _PALETTE_RE, "「暖色调」「莫兰迪色」「高对比」或 warm tones, high contrast"),
)

# 图内文字意图：成对引号 / reads "..." / 招牌标语类词。
# 只认直引号与中文弯引号成对出现，英文撇号（panda's）不算。
_TEXT_IN_IMAGE_RE = re.compile(
    r"\"[^\"]{1,60}\"|“[^”]{1,60}”|reads?\s+[\"“]"
    r"|招牌|字样|标语|写着|刻着|标牌|广告牌|霓虹字"
    r"|\b(?:signage|billboard|lettering|slogan|logo)\b|\bsign\b|\btext\b",
    re.IGNORECASE,
)


def booster_tips(prompt: str) -> list[str]:
    """空泛画质词占着 77-token 预算却不描述画面（Qwen-Image-2.1 官方规范禁用 booster）。

    只在提示词总 token 超过约 40（预算过半）时提醒，短提示词里它们无害。
    """
    text = prompt or ""
    found = sorted({m.group(0).lower() for m in _BOOSTER_RE.finditer(text)})
    if not found:
        return []
    n_tok = estimate_tokens(text)
    if n_tok < 40:
        return []
    cost = sum(estimate_tokens(", " + w) for w in found)
    return [
        f"「{'、'.join(found)}」属于空泛画质词：不描述任何画面内容（Qwen-Image-2.1 官方"
        f"提示词规范直接禁用）。当前提示词约 {n_tok} token（CLIP 上限 77），"
        f"删掉它们可省约 {cost} token 给主体与光影细节。"
    ]


def structure_tips(prompt: str) -> list[str]:
    """要素完整度：把 Qwen 八步法压缩成 77-token 预算下的“缺啥补啥”清单。

    至少缺两项才提示一次，避免每条短提示词都被唠叨；可识别的风格名也算媒介
    （style_map 会把抽象名展开成具体视觉特征）。
    """
    text = prompt or ""
    if estimate_tokens(text) < 3:
        return []

    medium_covered = bool(_MEDIUM_RE.search(text))
    if not medium_covered:
        try:
            from genimg.style_map import STYLES  # 局部导入，避免潜在环形依赖

            medium_covered = any(st.pattern.search(text) for st in STYLES)
        except Exception:  # noqa: BLE001
            pass

    missing: list[str] = []
    for name, pattern, example in _STRUCT_ELEMENTS:
        if name == "媒介/风格" and medium_covered:
            continue
        if not pattern.search(text):
            missing.append(f"缺「{name}」（可补 {example}）")
    if len(missing) < 2:
        return []
    return [
        "提示词要素检查（可选）：" + "；".join(missing) + "。"
        "这些词词典已收录，用「，」分段的短词写出后会自动转英文。"
    ]


def text_in_image_tips(prompt: str) -> list[str]:
    """图内文字：SD 系渲染长文字不可靠，控制字数并优先考虑两段式拼版。"""
    text = prompt or ""
    if not _TEXT_IN_IMAGE_RE.search(text):
        return []
    return [
        "提示词要求在图里出现文字：SD 系渲染长文字很不可靠。引号里的内容建议控制在 "
        "1–3 个短词，并写明位置与材质（例：a neon sign that reads \"OPEN\"）；"
        "整段版式文字请用两段式拼版（genimg/poster.py），写在提示词里只会被 77 token "
        "截断或画成乱码。"
    ]


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
    tips.extend(ethnicity_tips(prompt, negative_prompt))
    tips.extend(booster_tips(prompt))
    tips.extend(structure_tips(prompt))
    tips.extend(text_in_image_tips(prompt))

    return tips

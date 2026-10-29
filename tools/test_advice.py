"""提示词体检（advice）与 Qwen 规范借鉴改动（词典/画幅）的自检（纯离线，无外部依赖）。

对应 README §3.7：要素完整度 / 空泛画质词 / 图内文字三类体检，
以及 zh_prompt 词典增补、sizing 画幅关键词增补。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from genimg.advice import (  # noqa: E402
    booster_tips,
    check_request,
    estimate_tokens,
    structure_tips,
    text_in_image_tips,
)
from genimg.sizing import resolve  # noqa: E402
from genimg.zh_prompt import _GLOSSARY_RAW, translate  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "✅" if cond else "❌"
    line = f"  {mark} {name}"
    if not cond and detail:
        line += f"  [{detail}]"
    print(line)
    if not cond:
        FAILS.append(name)


print("== advice.booster_tips：空泛画质词 ==")
long_prompt = (
    "masterpiece, best quality, 8k, "
    + "a small wooden cabin beside a frozen lake in early winter, " * 2
    + "snow drifting over the pine trees, warm light in the windows"
)
tok = estimate_tokens(long_prompt)
check(f"长提示词（约 {tok} token）触发 booster 提示", len(booster_tips(long_prompt)) == 1)
check("短提示词不触发（预算充裕时不唠叨）", booster_tips("masterpiece, a panda") == [])
zh_long = "大师作品，" + "一只戴墨镜的熊猫坐在木屋前，" * 8
check("中文画质词（大师作品）也能识别", len(booster_tips(zh_long)) == 1)

print("== advice.structure_tips：要素完整度 ==")
tips = structure_tips("a panda on the grassland")
check("裸主体提示词给出一条要素提示", len(tips) == 1)
check(
    "提示里点名缺失要素（媒介/光影/镜头/色调）",
    bool(tips) and all(k in tips[0] for k in ("媒介", "光影", "镜头", "色调")),
    tips[0] if tips else "",
)
check(
    "要素齐全的提示词不再提示",
    structure_tips("watercolor painting, close-up, soft lighting, warm tones") == [],
)
tips = structure_tips("cyberpunk panda")
check(
    "风格名算媒介（style_map 会展开），只提示其余要素",
    bool(tips) and "媒介" not in tips[0] and "光影" in tips[0],
    tips[0] if tips else "",
)
check("过短提示词保持安静", structure_tips("a cat") == [])
check("中文提示词同样识别", len(structure_tips("一只可爱的熊猫")) == 1)

print("== advice.text_in_image_tips：图内文字 ==")
check('reads "OPEN" 触发', len(text_in_image_tips('a neon sign that reads "OPEN"')) == 1)
check("中文「写着/招牌」触发", len(text_in_image_tips("写着欢迎光临的招牌")) == 1)
check("texture / design 不误报（词边界）", text_in_image_tips("nice texture, graphic design") == [])
check("普通提示词不触发", text_in_image_tips("a panda") == [])

print("== advice.check_request：接入主通道 ==")
tips = check_request(mode="txt2img", prompt="a panda on the grassland", steps=25, guidance=7.0)
check("check_request 包含要素检查", any("要素检查" in t for t in tips))
check("空提示词安静", check_request(mode="txt2img", prompt="", steps=25, guidance=7.0) == [])

print("== zh_prompt：新增词条 ==")
check(f"词典条目数 = {len(_GLOSSARY_RAW)}", len(_GLOSSARY_RAW) == 163)
r = translate("暖光，轮廓光，磨砂玻璃，浮世绘，马卡龙色")
check(
    "新增词整句翻译且无残留",
    r.effective == "warm lighting, rim lighting, frosted glass, ukiyo-e, pastel colors" and not r.unknown,
    r.effective,
)
check("长词优先（霓虹招牌）", translate("霓虹招牌").effective == "neon signage")
check("长词优先（胶片颗粒 ≠ film photography grain）", translate("胶片颗粒").effective == "film grain")

print("== sizing：新增画幅关键词 ==")
d = resolve(prompt="一张手机壁纸，熊猫", model_key="sd-turbo")
check(f"手机壁纸 → {d.width}x{d.height}（{d.ratio}）", d.ratio == "9:16" and "手机壁纸" in d.source, d.describe())
d = resolve(prompt="证件照", model_key="sd-turbo")
check(f"证件照 → {d.ratio}", d.ratio == "3:4", d.describe())
d = resolve(prompt="a panorama of grassland", model_key="sd15")
check(f"panorama → {d.ratio}", d.ratio == "2:1", d.describe())
d = resolve(prompt="城市全景照片", model_key="sdxl")
check(f"全景照片 → {d.ratio}", d.ratio == "2:1", d.describe())
d = resolve(prompt="熊猫全景", model_key="sd-turbo")
check("模糊的「全景」不触发画幅（仍是默认 1:1）", d.source == "default", d.describe())
d = resolve(prompt="一张 A4 海报设计", model_key="sdxl")
check(f"A4 → {d.ratio}", d.ratio == "5:7", d.describe())

print()
if FAILS:
    print(f"共 {len(FAILS)} 项失败：{'、'.join(FAILS)}")
    sys.exit(1)
print("全部通过 ✅")

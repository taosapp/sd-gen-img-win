"""量一下候选提示词的 CLIP 词元数，确保风格块不被截断。"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import CLIPTokenizer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
snaps = glob.glob(
    os.path.join(ROOT, "models", "hub", "models--stabilityai--stable-diffusion-xl-base-1.0", "snapshots", "*")
)
tok = CLIPTokenizer.from_pretrained(snaps[0], subfolder="tokenizer")

STYLE = "gongbi painting, fine ink outlines, mineral pigments, silk scroll"

SHOTS = [
    "a pear tree full of yellow pears over a Han dynasty courtyard, two servants beneath it",
    "a servant lifting a basket of pears, a father in dark robe and three brothers watching",
    "a small boy in blue hanfu looks at a lacquer tray of pears of different sizes",
    "a father and three brothers sit on mats in age order, the youngest boy sits last",
    "the father gestures to the pear tray, asking his youngest son to choose first",
    "a four-year-old boy in blue hanfu holds the smallest pear, his brother holds a big one",
    "the father leans down to question his small son, brothers listening quietly",
    "the small boy bows with folded hands, answering his father, brothers watching",
    "the three elder brothers look ashamed at the little boy, the father nods approval",
    "the family sits together sharing pears, empty silk space above for a colophon",
]

print(f"{'镜头':<6}{'词元数':>8}  状态")
bad = 0
for i, s in enumerate(SHOTS, 1):
    full = f"{STYLE}, {s}"
    n = len(tok(full)["input_ids"]) - 2
    ok = n <= 77
    bad += 0 if ok else 1
    print(f"{i:<6}{n:>8}  {'OK' if ok else '截断!'}")
print(f"\n共 {len(SHOTS)} 条，超限 {bad} 条")

"""修复 webapp/server.py 中 generate() 的校验块（之前误删了 mode 校验的条件行）。"""

import re

PATH = "webapp/server.py"
src = open(PATH, encoding="utf-8").read()

# 目标：把"孤立在 if not prompt 里的第二行 raise"还原成独立的 mode 校验
pattern = re.compile(
    r'(    if not prompt:\n        raise HTTPException\(status_code=400, detail="[^"]*"\)\n)'
    r'        raise HTTPException\(status_code=400, detail="mode[^"]*"\)\n'
)
replacement = (
    r"\1"
    '    if mode not in ("txt2img", "img2img"):\n'
    '        raise HTTPException(status_code=400, detail="mode \u53ea\u80fd\u662f txt2img \u6216 img2img")\n'
)

new, count = pattern.subn(replacement, src)
if count != 1:
    raise SystemExit(f"未匹配到目标片段（匹配数 {count}），未做修改")

open(PATH, "w", encoding="utf-8").write(new)
print(f"已修复 1 处；文件行数 {len(new.splitlines())}")

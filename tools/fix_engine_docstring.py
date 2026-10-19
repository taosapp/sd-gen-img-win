"""修复 engine.py 中被误删的 docstring 首行（导致语法错误）。"""

import io
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "genimg", "engine.py")
lines = io.open(PATH, encoding="utf-8", newline="").read().split("\n")

# 定位：def _from_pretrained_compat 之后的两行裸文本（第 188、189 行，1-based）
idx = next(i for i, ln in enumerate(lines) if ln.startswith("def _from_pretrained_compat"))
assert lines[idx + 1].strip() == "", lines[idx + 1]
bad_start = idx + 2  # 0-based：裸文本第一行
assert "diffusers 0.41" in lines[bad_start], lines[bad_start]
bad_end = bad_start
while bad_end < len(lines) and not lines[bad_end].rstrip().endswith('"""'):
    bad_end += 1
assert bad_end < len(lines), "没有找到 docstring 结束标记"

doc = [
    '    """调用 from_pretrained，兼容 ``dtype`` 与旧的 ``torch_dtype`` 参数名。',
    "",
    "    diffusers 0.41 已把 ``torch_dtype`` 标为弃用（将在 1.0 移除），但旧版本只认",
    "    ``torch_dtype``，因此优先用新名、失败则回退。",
    '    """',
]
lines[bad_start : bad_end + 1] = doc

io.open(PATH, "w", encoding="utf-8", newline="").write("\n".join(lines))
print(f"已修复，文件行数 {len(lines)}")

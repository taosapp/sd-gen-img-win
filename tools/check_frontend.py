"""前端一致性检查：app.js 里引用的元素 ID 是否都能在 index.html 里找到。

浏览器工具不允许访问 localhost，无法截图验证，因此用静态交叉检查兜底：
ID 对不上是这类原生 JS 界面最常见的"点了没反应"原因。
"""

from __future__ import annotations

import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(PROJECT, "webapp", "static", "index.html")
JS = os.path.join(PROJECT, "webapp", "static", "app.js")

html = open(HTML, encoding="utf-8").read()
js = open(JS, encoding="utf-8").read()

html_ids = set(re.findall(r'\bid="([^"]+)"', html))
js_ids = set(re.findall(r"\$\('([^']+)'\)", js))
js_ids |= set(re.findall(r'getElementById\("([^"]+)"\)', js))

# 动态创建的元素（不在 HTML 里）
dynamic = {"badge"}

missing = sorted(js_ids - html_ids - dynamic)
unused = sorted(html_ids - js_ids - dynamic)

print(f"HTML 中的 id 数量：{len(html_ids)}")
print(f"JS 引用的 id 数量：{len(js_ids)}")
print()
if missing:
    print("[FAIL] JS 引用了但 HTML 里不存在的元素：")
    for m in missing:
        print("   -", m)
else:
    print("[ OK ] JS 引用的所有元素 ID 都能在 HTML 里找到")

if unused:
    print(f"\n[INFO] HTML 里有但 JS 未引用的 id（{len(unused)}）：{', '.join(unused)}")

# 括号配平的粗检查，避免明显的语法半截
for name, src in (("index.html", html), ("app.js", js)):
    for open_ch, close_ch in (("{", "}"), ("(", ")"), ("[", "]")):
        if src.count(open_ch) != src.count(close_ch):
            print(f"[WARN] {name} 里 {open_ch}{close_ch} 数量不等（可能只是字符串里的括号）")

print()
print("结论：" + ("通过" if not missing else "需要修复"))
sys.exit(0 if not missing else 1)

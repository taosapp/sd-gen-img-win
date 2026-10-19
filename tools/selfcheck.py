"""开发期自检：检查各模块能否导入、语法是否正常、关键结构是否完整。"""

import ast
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    "genimg.py",
    "bench.py",
    "genimg/__init__.py",
    "genimg/presets.py",
    "genimg/directives.py",
    "genimg/runtime.py",
    "genimg/engine.py",
    "genimg/cli.py",
    "tools/fetch_models.py",
    "tools/make_contact_sheet.py",
]

ok = True
for rel in FILES:
    path = os.path.join(PROJECT, rel)
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        print(f"[FAIL] {rel}: 语法错误 {exc}")
        ok = False
        continue
    # 统计顶层定义，便于确认结构完整
    kinds = {}
    for node in tree.body:
        kinds[type(node).__name__] = kinds.get(type(node).__name__, 0) + 1
    funcs = sum(1 for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    classes = sum(1 for n in tree.body if isinstance(n, ast.ClassDef))
    print(f"[ OK ] {rel:28s} 行数 {len(src.splitlines()):4d}  顶层函数 {funcs:2d}  顶层类 {classes:2d}  非 ASCII {sum(1 for c in src if ord(c) > 127)}")

print()
print("导入检查：")
sys.path.insert(0, PROJECT)
for mod in ("genimg.presets", "genimg.directives", "genimg.runtime", "genimg.engine", "genimg.cli"):
    try:
        __import__(mod)
        print(f"[ OK ] import {mod}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] import {mod}: {type(exc).__name__}: {exc}")
        ok = False

sys.exit(0 if ok else 1)

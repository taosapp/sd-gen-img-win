"""汇总矩阵耗时，标记超 1 分钟的组合，并输出候选黑名单。

用法：
  python tools/screen_matrix.py                 # 只看耗时
  python tools/screen_matrix.py --fails a.json  # 再合并人工判定的内容不符项
"""

from __future__ import annotations

import csv
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
PROJECT = _ROOT
CSV_PATH = os.path.join(PROJECT, "results", "ref_matrix", "timing.csv")
LIMIT = 60.0


def main() -> int:
    fails_path = None
    if "--fails" in sys.argv:
        fails_path = sys.argv[sys.argv.index("--fails") + 1]

    if not os.path.exists(CSV_PATH):
        print("还没有 timing.csv")
        return 1

    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    print(f"共 {len(rows)} 条记录")

    slow = [r for r in rows if r["status"] == "ok" and float(r["hot_seconds"] or 0) > LIMIT]
    errors = [r for r in rows if r["status"] != "ok"]

    # 按 (模型, 精度) 汇总热态耗时区间
    agg: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        key = (r["model"], r["precision"])
        agg.setdefault(key, []).append(float(r["hot_seconds"]))

    print(f"\n{'模型':<10}{'精度':<6}{'张数':>4}{'最快':>8}{'最慢':>8}   判定")
    print("-" * 56)
    for (model, prec), vals in sorted(agg.items()):
        lo, hi = min(vals), max(vals)
        verdict = "超时" if hi > LIMIT else "OK"
        print(f"{model:<10}{prec:<6}{len(vals):>4}{lo:>7.2f}s{hi:>7.2f}s   {verdict}")

    if slow:
        print(f"\n热态 > {LIMIT:.0f}s 的组合 {len(slow)} 个：")
        for r in slow:
            print(f"  {r['model']}/{r['precision']}/{r['gear']}  {float(r['hot_seconds']):.1f}s")

    if errors:
        print(f"\n生成失败 {len(errors)} 个：")
        for r in errors:
            print(f"  {r['model']}/{r['precision']}/{r['gear']}: {r['error'][:60]}")

    content = {}
    if fails_path and os.path.exists(fails_path):
        content = json.load(open(fails_path, encoding="utf-8"))
        print(f"\n人工判定内容不符：{len(content.get('combos', []))} 条")
        for c in content.get("combos", []):
            print("  ", c)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

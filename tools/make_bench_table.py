"""把 bench.json 转成 Markdown 表格，避免手工抄写出错。

用法：
    python tools\\make_bench_table.py                    # 打印到控制台
    python tools\\make_bench_table.py --write README_SECTION.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATUS_ZH = {
    "unsupported": "硬件不支持",
    "invalid": "组合非法",
    "failed": "失败",
    "timeout": "超时中止",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join("results", "bench", "bench.json"))
    ap.add_argument("--write", default=None, help="写入指定文件（否则打印到控制台）")
    args = ap.parse_args()

    path = args.json if os.path.isabs(args.json) else os.path.join(PROJECT, args.json)
    data = json.load(open(path, encoding="utf-8"))
    hw = data["hardware"]
    cfg = data["config"]

    lines: list[str] = []
    lines.append(
        f"实测环境：{hw['gpu']}（{hw['cc']}，{hw['vram_gb']}GB）· torch {hw['torch']} / CUDA {hw['cuda']} · "
        f"{cfg['size']} · 每配置预热 {cfg['warmup']} 次 + 计时 {cfg['reps']} 次（取中位数）"
    )
    lines.append("")
    lines.append("| 模型 | 精度 | 画质 | 步数 | 中位耗时 | 去噪 | 编解码 | 显存峰值 | 放置 | ≤30s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in data["rows"]:
        if r.get("status") != "ok":
            tag = STATUS_ZH.get(r.get("status"), r.get("status"))
            lines.append(f"| {r['model']} | {r['precision']} | {r['gear']} | – | {tag} | – | – | – | – | – |")
            continue
        lines.append(
            f"| {r['model']} | {r['precision']} | {r['gear']} | {r['steps']} | "
            f"**{r['median_infer_s']:.2f}s** | {r['median_denoise_s']:.2f}s | {r['decode_s']:.2f}s | "
            f"{r['peak_vram_gb']:.2f}GB | {r['placement']} | {'✅' if r['within_30s'] else '❌'} |"
        )

    ok = [r for r in data["rows"] if r.get("status") == "ok"]
    if ok:
        fast = min(ok, key=lambda r: r["median_infer_s"])
        slow = max(ok, key=lambda r: r["median_infer_s"])
        passed = sum(1 for r in ok if r["within_30s"])
        lines.append("")
        lines.append(
            f"小结：可跑通 {len(ok)}/{len(data['rows'])} 个配置，其中 {passed} 个满足 30 秒预算；"
            f"最快 {fast['model']}/{fast['precision']}/{fast['gear']} {fast['median_infer_s']:.2f}s，"
            f"最慢（可跑通）{slow['model']}/{slow['precision']}/{slow['gear']} {slow['median_infer_s']:.2f}s。"
        )

    text = "\n".join(lines)
    if args.write:
        out = args.write if os.path.isabs(args.write) else os.path.join(PROJECT, args.write)
        open(out, "w", encoding="utf-8").write(text + "\n")
        print(f"已写入 {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

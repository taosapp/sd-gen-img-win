"""按需下载 diffusers 格式的模型权重。

只用 ``allow_patterns`` 拉取真正需要的文件，避免把整仓（含 openvino / 单文件 ckpt /
非 ema 副本 / 安全检查器）全部拖下来——SDXL 系整仓约 49GB，本脚本只需约 7GB。

用法（在 genimg 环境里执行）：
    python tools\\fetch_models.py                 # 下载默认组合
    python tools\\fetch_models.py --only sdxl     # 只下某一个
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genimg.runtime import configure_hf  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 所有仓库都要的小文件（配置、分词器、调度器）
COMMON = [
    "model_index.json",
    "*/config.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "feature_extractor/*",
    "*.txt",
    "*.json",
]


def _weights(prefix: str, names: list[str], fp16: bool) -> list[str]:
    suffix = ".fp16.safetensors" if fp16 else ".safetensors"
    return [f"{prefix}/{n}{suffix}" for n in names]


def unet_repo_patterns(fp16: bool) -> list[str]:
    """SD / SDXL diffusers 仓库：按精度变体挑权重文件。"""
    pats = list(COMMON)
    pats += _weights("unet", ["diffusion_pytorch_model"], fp16)
    pats += _weights("text_encoder", ["model"], fp16)
    pats += _weights("vae", ["diffusion_pytorch_model"], fp16)
    pats += _weights("text_encoder_2", ["model"], fp16)  # SDXL 专用；单塔模型无所谓
    return pats


DOWNLOADS: dict[str, dict] = {
    "sd15": {
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "patterns": unet_repo_patterns(False) + unet_repo_patterns(True),
        "about": "SD1.5：同时拉 fp32 与 fp16 两套权重，用于精度对照（约 6.4GB）",
        "size_gb": 6.4,
    },
    "sdxl": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "patterns": unet_repo_patterns(True),
        "about": "SDXL base：只拉 fp16 变体（约 6.9GB；fp32 整仓另有 13.9GB）",
        "size_gb": 6.9,
    },
    "sd-turbo": {
        "repo": "stabilityai/sd-turbo",
        "patterns": unet_repo_patterns(True),
        "about": "SD-Turbo：512 原生少步模型（约 2.2GB）",
        "size_gb": 2.2,
    },
    "lightning": {
        "repo": "ByteDance/SDXL-Lightning",
        "patterns": ["sdxl_lightning_4step_lora.safetensors", "sdxl_lightning_8step_lora.safetensors"],
        "about": "SDXL-Lightning LoRA：4 步 / 8 步加速（约 0.8GB）",
        "size_gb": 0.8,
    },
}

DEFAULT_SET = ["sd-turbo", "sdxl", "sd15", "lightning"]


def main() -> int:
    ap = argparse.ArgumentParser(description="下载所需模型权重")
    ap.add_argument("--only", action="append", choices=sorted(DOWNLOADS), help="只下载指定项，可重复")
    ap.add_argument("--hf-source", choices=["huggingface", "mirror"], default="mirror")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--retries", type=int, default=4, help="单项失败后的重试次数（会自动续传）")
    ap.add_argument("--max-workers", type=int, default=4, help="并发下载文件数")
    args = ap.parse_args()

    targets = args.only or list(DEFAULT_SET)

    # 单个连接卡住时尽快失败并重试，而不是无限等下去
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    configure_hf(PROJECT_ROOT, source=args.hf_source, proxy=args.proxy, verbose=True)
    from huggingface_hub import snapshot_download

    print(f"计划下载：{', '.join(targets)}")
    total = sum(DOWNLOADS[t]["size_gb"] for t in targets)
    print(f"预计体积：约 {total:.1f} GB\n")

    failed: list[str] = []
    for key in targets:
        spec = DOWNLOADS[key]
        print("=" * 96)
        print(f"[{key}] {spec['repo']}")
        print(f"        {spec['about']}")
        ok = False
        t0 = time.perf_counter()
        for attempt in range(1, args.retries + 1):
            try:
                path = snapshot_download(
                    repo_id=spec["repo"],
                    allow_patterns=spec["patterns"],
                    max_workers=args.max_workers,
                )
                dt = time.perf_counter() - t0
                print(f"        完成于 {dt:.0f}s（第 {attempt} 次尝试）-> {path}")
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"        第 {attempt}/{args.retries} 次失败：{type(exc).__name__}: {str(exc)[:120]}")
                time.sleep(2)
        if not ok:
            failed.append(key)

    print()
    if failed:
        print(f"以下项目下载失败，可重跑本脚本续传：{', '.join(failed)}")
        return 1
    print("全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

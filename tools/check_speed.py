"""Measure real download throughput for HF weights: hf-mirror (direct) vs huggingface (proxy)."""

import time
import urllib.request

PROXY = "http://10.10.100.199:3128"
FILE = "stabilityai/sd-turbo/resolve/main/unet/diffusion_pytorch_model.safetensors"  # ~1.7GB, we read only 40MB
CHUNK = 40 * 1024 * 1024
UA = {"User-Agent": "Mozilla/5.0"}


def measure(label, url, proxy=None):
    if proxy:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers=UA)
    try:
        t0 = time.perf_counter()
        with op.open(req, timeout=30) as r:
            total = 0
            while total < CHUNK:
                b = r.read(1024 * 256)
                if not b:
                    break
                total += len(b)
        dt = time.perf_counter() - t0
        mb = total / 1e6
        print(f"{label:34s} {mb:7.1f} MB in {dt:5.1f}s -> {mb/dt:6.2f} MB/s   (header: {r.status})")
    except Exception as e:  # noqa: BLE001
        print(f"{label:34s} FAIL {type(e).__name__}: {str(e)[:70]}")


measure("hf-mirror.com (direct)", f"https://hf-mirror.com/{FILE}")
measure("huggingface.co (proxy)", f"https://huggingface.co/{FILE}", PROXY)
print()
print("== 1.7GB unet extrapolated ==")

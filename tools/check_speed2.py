"""并排比较：HF 官方仓库经代理的原始吞吐 vs 现有下载器实际吞吐。"""

import time
import urllib.request

PROXY = "http://10.10.100.199:3128"
UA = {"User-Agent": "Mozilla/5.0"}

FILE = "stabilityai/sd-turbo/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors"
CHUNK = 30 * 1024 * 1024


def measure(label, url, proxy=None, chunk=CHUNK, read=1024 * 256):
    if proxy:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        t0 = time.perf_counter()
        with op.open(urllib.request.Request(url, headers=UA), timeout=30) as r:
            total = 0
            while total < chunk:
                b = r.read(read)
                if not b:
                    break
                total += len(b)
        dt = time.perf_counter() - t0
        print(f"{label:44s} {total/1e6:7.1f} MB / {dt:5.2f}s -> {total/1e6/dt:6.2f} MB/s")
    except Exception as e:  # noqa: BLE001
        print(f"{label:44s} FAIL {type(e).__name__}: {str(e)[:60]}")


print("== 相同文件、相同进程、不同源/不同读块 ==")
measure("huggingface.co via proxy (256KB reads)", f"https://huggingface.co/{FILE}", PROXY)
measure("huggingface.co via proxy (1MB reads)", f"https://huggingface.co/{FILE}", PROXY, read=1024 * 1024)
measure("hf-mirror.com direct (256KB reads)", f"https://hf-mirror.com/{FILE}", None)
measure("huggingface.co via proxy (中文UA)", f"https://huggingface.co/{FILE}", PROXY)

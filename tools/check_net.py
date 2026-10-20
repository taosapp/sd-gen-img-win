"""Connectivity probe: direct vs PAC proxy, for the hosts we need.

Runs the same small request through (a) direct connection and (b) the PAC proxy,
so we can decide how to configure pip / huggingface_hub.
"""

import json
import time
import urllib.error
import urllib.request

PROXY = "http://10.10.100.199:3128"
TARGETS = [
    ("hf_api", "https://huggingface.co/api/models/stabilityai/sd-turbo"),
    ("hf_mirror_api", "https://hf-mirror.com/api/models/stabilityai/sd-turbo"),
    ("hf_resolve", "https://huggingface.co/stabilityai/sd-turbo/resolve/main/model_index.json"),
    ("torch_idx", "https://download.pytorch.org/whl/cu126/torch/"),
    ("pypi", "https://pypi.org/pypi/diffusers/json"),
]


def probe(url, proxy=None, timeout=20):
    if proxy:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.perf_counter()
    try:
        with op.open(req, timeout=timeout) as r:
            data = r.read(2048)
        return f"OK {r.status} {len(data)}B {(time.perf_counter()-t0)*1000:.0f}ms"
    except Exception as e:  # noqa: BLE001
        return f"FAIL {type(e).__name__}: {str(e)[:70]}"


print(f"{'target':16s} {'DIRECT':52s} PROXY")
for name, url in TARGETS:
    d = probe(url)
    p = probe(url, PROXY) if "FAIL" in d else "(direct ok, skipped)"
    print(f"{name:16s} {d:52s} {p}")
    print()

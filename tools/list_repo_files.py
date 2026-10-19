"""List exact files+sizes for the candidate repos (via proxy) so we can budget the download."""

import json
import urllib.request

PROXY = "http://10.10.100.199:3128"
OP = urllib.request.build_opener(urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
UA = {"User-Agent": "Mozilla/5.0"}

REPOS = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "ByteDance/SDXL-Lightning",
]


def api(path):
    return json.load(OP.open(urllib.request.Request(f"https://huggingface.co/api/{path}", headers=UA), timeout=40))


for rid in REPOS:
    try:
        info = api(f"models/{rid}?blobs=true")
        print(f"\n### {rid}   gated={info.get('gated')}  private={info.get('private')}")
        rows = []
        for s in info.get("siblings", []):
            f = s["rfilename"]
            if f.endswith((".safetensors", ".bin", ".ckpt", ".json")) and s.get("size"):
                rows.append((s["size"], f))
        rows.sort(reverse=True)
        total = sum(r[0] for r in rows)
        for size, f in rows:
            print(f"   {size/1e6:9.1f} MB  {f}")
        print(f"   --- total listed: {total/1e9:.2f} GB")
    except Exception as e:  # noqa: BLE001
        print(f"\n### {rid} FAIL {type(e).__name__}: {str(e)[:90]}")

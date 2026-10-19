"""Probe PyPI / PyTorch wheel availability for Python 3.13 on Windows (x64)."""
import json
import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")


print("=== PyPI latest + cp313/win_amd64 or pure-python wheels ===")
for pkg in [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "torchao",
    "peft",
    "pillow",
    "huggingface-hub",
    "sentencepiece",
]:
    try:
        d = json.loads(get(f"https://pypi.org/pypi/{pkg}/json"))
        ver = d["info"]["version"]
        c13 = [f["filename"] for f in d["urls"] if "cp313" in f["filename"] and "win_amd64" in f["filename"]]
        pure = [f["filename"] for f in d["urls"] if f["filename"].endswith("py3-none-any.whl")]
        sizes = {f["filename"]: round(f["size"] / 1e6) for f in d["urls"] if f["filename"] in c13}
        print(f"{pkg}: latest={ver} cp313win={len(c13)} pure={len(pure)} sizes_MB={sizes}")
    except Exception as e:  # noqa: BLE001
        print(f"{pkg}: ERROR {e}")

print()
print("=== download.pytorch.org/whl indexes: cp313 win_amd64 torch wheels ===")
for tag in ["cu121", "cu124", "cu126", "cu128", "cu129", "cu130"]:
    try:
        h = get(f"https://download.pytorch.org/whl/{tag}/torch/")
        hits = re.findall(r"torch-(\d[^\"<>]*?)cp313[^\"<>]*?win_amd64\.whl", h)
        uniq = sorted(set(hits))[-3:]
        print(f"{tag}: {len(set(hits))} distinct -> {uniq}")
    except Exception as e:  # noqa: BLE001
        print(f"{tag}: ERROR {e}")

print()
print("=== index page (no revision suffix) ===")
try:
    h = get("https://download.pytorch.org/whl/index.html")
    print("index ok, len", len(h))
except Exception as e:  # noqa: BLE001
    print("index ERROR", e)

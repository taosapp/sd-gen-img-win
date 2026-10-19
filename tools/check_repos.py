"""Check HuggingFace model repo ids exist and enumerate the files we need."""

import json
import urllib.request

REPOS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-turbo": "stabilityai/sd-turbo",
    "lightning": "ByteDance/SDXL-Lightning",
    "flux-schnell-gguf": "city96/FLUX.1-schnell-gguf",
}

UA = {"User-Agent": "Mozilla/5.0"}


def api(path):
    req = urllib.request.Request(f"https://huggingface.co/api/{path}", headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=30))


for key, rid in REPOS.items():
    try:
        d = api(f"models/{rid}")
        files = [s["rfilename"] for s in d.get("siblings", [])]
        heavy = [f for f in files if f.endswith((".safetensors", ".bin", ".ckpt", ".gguf"))]
        print(f"[OK]   {key:18s} {rid}  gated={d.get('gated')}  files={len(files)}")
        wanted = [
            f
            for f in heavy
            if any(k in f for k in ("fp16", "unet", "vae", "text_encoder", "lora", "Q4_K_S", "diffusion_pytorch_model.safetensors"))
        ]
        for f in sorted(wanted)[:14]:
            print(f"          - {f}")
        other = [f for f in heavy if f not in wanted][:4]
        for f in sorted(other):
            print(f"          (other) {f}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {key:18s} {rid}  -> {type(e).__name__}: {e}")

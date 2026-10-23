"""决定性实验：diffusers 0.41 到底认哪个 dtype 参数名？"""

from __future__ import annotations

import gc
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 本脚本位于 tools/archive/
sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from genimg.runtime import configure_hf  # noqa: E402

PROJECT = _ROOT
configure_hf(PROJECT, source="mirror", verbose=False)

import diffusers  # noqa: E402
from diffusers import AutoencoderKL, StableDiffusionXLPipeline  # noqa: E402

print("diffusers", diffusers.__version__, "| torch", torch.__version__)
REPO = "stabilityai/stable-diffusion-xl-base-1.0"


def probe(label, **kwargs):
    gc.collect()
    torch.cuda.empty_cache()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                REPO, use_safetensors=True, add_watermarker=False, variant="fp16", **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:38s} 异常 {type(exc).__name__}: {str(exc)[:90]}")
            return
    u = next(pipe.unet.parameters()).dtype
    t = next(pipe.text_encoder.parameters()).dtype
    t2 = next(pipe.text_encoder_2.parameters()).dtype
    v = next(pipe.vae.parameters()).dtype
    print(f"  {label:38s} unet={str(u):16s} te={str(t):16s} te2={str(t2):16s} vae={str(v)}")
    msgs = [str(w.message) for w in caught if "dtype" in str(w.message).lower()]
    for m in msgs[:2]:
        print(f"       警告: {m[:110]}")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


print()
print("== A) 只传 dtype= ==")
probe("dtype=torch.float16", dtype=torch.float16)

print("== B) 只传 torch_dtype= ==")
probe("torch_dtype=torch.float16", torch_dtype=torch.float16)

print("== C) 都不传（基线） ==")
probe("(无 dtype 参数)")

print()
print("== D) 单独加载 VAE 组件，看 dtype 是否生效 ==")
for key in ("dtype", "torch_dtype"):
    kwargs = {key: torch.float32}
    try:
        vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae", variant="fp16", **kwargs)
        print(f"  VAE via {key:12s} -> {next(vae.parameters()).dtype}")
        del vae
    except Exception as exc:  # noqa: BLE001
        print(f"  VAE via {key:12s} -> 异常 {type(exc).__name__}: {str(exc)[:80]}")

"""逐组件定位性能瓶颈：分别计时文本编码器、UNet 单步、VAE 解码。"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from genimg.engine import Engine  # noqa: E402
from genimg.presets import resolve_model, resolve_precision, resolve_quality  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "sdxl"
PREC = sys.argv[2] if len(sys.argv) > 2 else "fp16"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
engine = Engine(project_root, verbose=True)
profile = resolve_model(MODEL)
precision = resolve_precision(PREC)
quality = resolve_quality(profile.default_gear, profile)

loaded = engine.load(profile, precision, quality, resolution=512)
pipe = loaded.txt2img

print()
print("=" * 90)
print("管线状态")
print("=" * 90)
for attr in ("unet", "vae", "text_encoder", "text_encoder_2"):
    m = getattr(pipe, attr, None)
    if m is None:
        continue
    p = next(m.parameters())
    n = sum(x.numel() for x in m.parameters())
    print(f"  {attr:16s} dtype={str(p.dtype):16s} device={str(p.device):10s} params={n/1e6:8.1f}M")
print(f"  torch.backends.cuda.matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")
print(f"  torch.backends.cudnn.allow_tf32       = {torch.backends.cudnn.allow_tf32}")
print(f"  unet attn processor = {type(pipe.unet.attn_processors[[k for k in pipe.unet.attn_processors][0]]).__name__}")
print(f"  显存已分配 {torch.cuda.memory_allocated()/1024**3:.2f} GB / 保留 {torch.cuda.memory_reserved()/1024**3:.2f} GB")


def timeit(label, fn, warmup=2, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    print(f"  {label:28s} {dt*1000:9.2f} ms")
    return dt


dev, dt = next(pipe.unet.parameters()).device, next(pipe.unet.parameters()).dtype
print()
print("=" * 90)
print("逐组件计时 @ 512x512")
print("=" * 90)

with torch.no_grad():
    # 文本编码器
    ids = torch.randint(0, 49000, (1, 77), device=dev)
    timeit("text_encoder 前向", lambda: pipe.text_encoder(ids))
    if getattr(pipe, "text_encoder_2", None) is not None:
        timeit("text_encoder_2 前向", lambda: pipe.text_encoder_2(ids))

    # UNet 单步（SDXL 需要 added_cond_kwargs）
    lat = torch.randn(1, 4, 64, 64, device=dev, dtype=dt)
    ts = torch.tensor([500.0], device=dev, dtype=dt)
    if profile.family == "sdxl":
        ehs = torch.randn(1, 77, 2048, device=dev, dtype=dt)
        add = {
            "text_embeds": torch.randn(1, 1280, device=dev, dtype=dt),
            "time_ids": torch.randn(1, 6, device=dev, dtype=dt),
        }
    else:
        ehs = torch.randn(1, 77, 1024, device=dev, dtype=dt)
        add = None
    timeit("UNet 单步前向", lambda: pipe.unet(lat, ts, encoder_hidden_states=ehs, added_cond_kwargs=add).sample)

    # VAE 解码
    vlat = torch.randn(1, 4, 64, 64, device=dev, dtype=next(pipe.vae.parameters()).dtype)
    timeit("VAE 解码", lambda: pipe.vae.decode(vlat / pipe.vae.config.scaling_factor).sample)

"""验证「比例 + 像素档」组合，实际出图尺寸是否符合预期。"""

import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7866"

# 期望值来自后端 size_for_ratio（64 倍数、面积贴近 base²）
CASES = [
    ("1:1", 512, (512, 512)),
    ("1:1", 768, (768, 768)),
    ("1:1", 1024, (1024, 1024)),
    ("3:4", 512, (384, 512)),
    ("3:4", 1024, (768, 1024)),
    ("4:3", 768, (768, 576)),
    ("16:9", 512, (576, 320)),
    ("16:9", 1024, (1344, 768)),
    ("9:16", 1024, (768, 1344)),
]


def post(form):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in form.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        body.write(str(v).encode("utf-8"))
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{BASE}/api/generate",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def wait(job_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{BASE}/api/jobs/{job_id}", timeout=10) as r:
            job = json.loads(r.read().decode())
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(0.3)
    raise TimeoutError(job_id)


bad = 0
print(f"{'比例':<6}{'像素档':<8}{'期望':<12}{'实际':<12}{'耗时':>7}  结果")
print("-" * 60)
for ratio, px, expect in CASES:
    data = post(
        {
            "mode": "txt2img",
            "prompt": "a panda running on a green grassland, sunny day",
            "model": "sd-turbo",
            "precision": "bf16",
            "gear": "turbo",
            "ratio": ratio,
            "pixels": px,
        }
    )
    job = wait(data["id"])
    if job.get("status") != "done":
        print(f"{ratio:<6}{px:<8}{'-':<12}{'-':<12}{'-':>7}  失败 {job.get('error')[:40]}")
        bad += 1
        continue
    m = job.get("meta") or {}
    got = (m.get("width"), m.get("height"))
    ok = got == expect
    bad += 0 if ok else 1
    print(f"{ratio:<6}{px:<8}{f'{expect[0]}x{expect[1]}':<12}{f'{got[0]}x{got[1]}':<12}{m.get('seconds', 0):>6.2f}s  {'OK' if ok else '不符'}")

print("\n结论：" + ("全部符合" if bad == 0 else f"{bad} 组不符"))
sys.exit(0 if bad == 0 else 1)

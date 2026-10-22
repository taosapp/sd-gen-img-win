"""验证 9:16 画幅：界面传 size=9:16 时实际出图尺寸。"""

import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:7860"


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
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print("  HTTP", e.code, e.read().decode()[:300])
        raise


def wait(job_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{BASE}/api/jobs/{job_id}", timeout=10) as r:
            job = json.loads(r.read().decode())
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(0.4)
    raise TimeoutError(job_id)


CASES = [("sd15", "balanced", (320, 576)), ("sdxl", "lightning", (768, 1344))]
bad = 0
for model, gear, expect in CASES:
    data = post(
        {
            "mode": "txt2img",
            "prompt": "a lighthouse at dusk",
            "model": model,
            "gear": gear,
            "size": "9:16",
        }
    )
    job = wait(data["id"])
    if job.get("status") != "done":
        print(f"  {model}: 失败 {job.get('error')}")
        bad += 1
        continue
    m = job.get("meta") or {}
    got = (m.get("width"), m.get("height"))
    flag = "OK " if got == expect else "!! "
    if got != expect:
        bad += 1
    print(f"  {flag}{model:<6} size=9:16 → {got[0]}x{got[1]}（期望 {expect[0]}x{expect[1]}） {m.get('seconds', 0):.2f}s")

print("结论：" + ("符合预期" if bad == 0 else "有不符合项"))
sys.exit(0 if bad == 0 else 1)

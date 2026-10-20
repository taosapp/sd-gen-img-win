"""验证 Web API 的画幅参数：auto / 3:4 / custom 是否落到正确的像素尺寸。"""

from __future__ import annotations

import io
import json
import mimetypes
import sys
import time
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7865"
PHOTO = r"C:\Users\PPXW\Desktop\1.jpg"


def post(form: dict, image_path: str | None = None):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()

    def write_field(name, value):
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(str(value).encode("utf-8"))
        body.write(b"\r\n")

    for k, v in form.items():
        write_field(k, v)

    if image_path:
        ctype = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="image"; filename="photo.jpg"\r\n'.encode()
        )
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.write(open(image_path, "rb").read())
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{BASE}/api/generate",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def wait(job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{BASE}/api/jobs/{job_id}", timeout=10) as r:
            job = json.loads(r.read().decode())
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(0.4)
    raise TimeoutError(job_id)


CASES = [
    ("auto + 提示词里的 3:4", {"size": "auto", "prompt": "Strict 3:4 vertical poster, watercolor", "model": "sd15", "gear": "balanced", "strength": "0.5"}, None, (384, 512)),
    ("size=3:4", {"size": "3:4", "prompt": "watercolor illustration", "model": "sd15", "gear": "balanced", "strength": "0.5"}, None, (384, 512)),
    ("size=custom 640x480", {"size": "custom", "width": "640", "height": "480", "prompt": "watercolor illustration", "model": "sd15", "gear": "balanced", "strength": "0.5"}, None, (640, 480)),
    ("size=auto + 参考图 1024x683", {"size": "auto", "prompt": "watercolor illustration", "model": "sd15", "gear": "balanced", "strength": "0.5"}, PHOTO, (576, 384)),
]

print(f"目标服务: {BASE}")
ok = True
for label, form, img, expect in CASES:
    form = dict(form)
    form.setdefault("mode", "img2img" if img else "txt2img")
    data = post(form, img)
    job = wait(data["id"])
    if job.get("status") != "done":
        print(f"  {label:<26} 失败: {job.get('error')}")
        ok = False
        continue
    m = job.get("meta") or {}
    got = (m.get("width"), m.get("height"))
    flag = "OK " if got == expect else "!! "
    print(f"  {flag}{label:<26} 期望 {expect[0]}x{expect[1]}  实际 {got[0]}x{got[1]}  "
          f"来源={m.get('size_source')}  耗时 {m.get('seconds', 0):.2f}s")
    if got != expect:
        ok = False

print("\n结论：" + ("全部符合预期" if ok else "存在不符合预期的项"))
sys.exit(0 if ok else 1)

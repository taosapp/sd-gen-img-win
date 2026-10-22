"""验证故事板：提交 10 条提示词，轮询到全部完成，检查张数与尺寸。"""

import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7866"

SHOTS = [f"shot {i}: a lone traveler walking through an ancient street, cinematic" for i in range(1, 11)]


def post_storyboard(prompts, ratio="9:16"):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in {"prompts": json.dumps(prompts, ensure_ascii=False), "ratio": ratio}.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        body.write(str(v).encode("utf-8"))
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{BASE}/api/storyboard",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:300])
        raise


t0 = time.time()
meta = post_storyboard(SHOTS)
print(f"批次 {meta['batch_id']}  画幅 {meta['width']}x{meta['height']}  共 {meta['total']} 张")

done = 0
while True:
    with urllib.request.urlopen(f"{BASE}/api/storyboard/{meta['batch_id']}", timeout=15) as r:
        st = json.loads(r.read().decode())
    finished = st["done"] + st["failed"]
    if finished != done:
        done = finished
        print(f"  进度 {done}/{st['total']}  失败 {st['failed']}  已用 {time.time() - t0:.1f}s", flush=True)
    if finished >= st["total"]:
        break
    time.sleep(1.0)

urls = [i["url"] for i in st["items"] if i["url"]]
print(f"\n完成 {st['done']}/{st['total']}，失败 {st['failed']}，总耗时 {time.time() - t0:.1f}s")
print("前 3 个地址:", urls[:3])

# 校验尺寸
from PIL import Image  # noqa: E402

import os  # noqa: E402

sizes = set()
for u in urls:
    p = os.path.join(r"D:\ai\gen-img-agent", u.lstrip("/").replace("/", os.sep))
    if os.path.exists(p):
        sizes.add(Image.open(p).size)
print("出图尺寸:", sizes)
sys.exit(0 if st["done"] == st["total"] and sizes == {(meta["width"], meta["height"])} else 1)

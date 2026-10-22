"""验证故事板：指定张数与画幅，检查是否同一文件夹、文件名是否带 shotNN 前缀。"""

import glob
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7866"
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 3
RATIO = sys.argv[3] if len(sys.argv) > 3 else "16:9"
EXPECT = {"9:16": (768, 1344), "16:9": (1344, 768)}[RATIO]

SHOTS = [f"shot {i}: a traveler walking through an ancient street, cinematic" for i in range(1, COUNT + 1)]


def post(prompts, ratio):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in {"prompts": json.dumps(prompts), "ratio": ratio}.items():
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
meta = post(SHOTS, RATIO)
print(f"批次 {meta['batch_id']}  画幅 {meta['width']}x{meta['height']}  提交 {meta['total']} 张")

while True:
    with urllib.request.urlopen(f"{BASE}/api/storyboard/{meta['batch_id']}", timeout=15) as r:
        st = json.loads(r.read().decode())
    if st["done"] + st["failed"] >= st["total"]:
        break
    time.sleep(1.0)

urls = [i["url"] for i in st["items"] if i["url"]]
print(f"完成 {st['done']}/{st['total']}，失败 {st['failed']}，耗时 {time.time() - t0:.1f}s")

root = r"D:\ai\gen-img-agent"
paths = [os.path.join(root, u.lstrip("/").replace("/", os.sep)) for u in urls]
folders = {os.path.dirname(p) for p in paths}
names = [os.path.basename(p) for p in paths]

from PIL import Image  # noqa: E402

sizes = {Image.open(p).size for p in paths if os.path.exists(p)}

print("图片所在文件夹:", [os.path.relpath(f, root) for f in folders])
print("文件名:", names)
print("尺寸:", sizes)

ok = (
    len(paths) == COUNT
    and len(folders) == 1
    and sizes == {EXPECT}
    and all(n.startswith("shot") for n in names)
)
print("结论：" + ("符合预期（同一文件夹 + shotNN 前缀 + 尺寸一致）" if ok else "不符合预期"))
sys.exit(0 if ok else 1)

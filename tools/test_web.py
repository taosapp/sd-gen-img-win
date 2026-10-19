"""Web 界面端到端测试：文生图 + 上传图片做图生图，并校验返回的图片真的可用。

用法：
    python tools\\test_web.py                    # 默认 http://127.0.0.1:7861
    python tools\\test_web.py --port 7860
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import sys
import time
import urllib.request
import uuid

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post_form(base: str, path: str, fields: dict, file_field: str | None = None, file_path: str | None = None):
    boundary = "----genimg" + uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        body.write(str(v).encode("utf-8"))
        body.write(b"\r\n")
    if file_field and file_path:
        ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{file_field}"; filename="{os.path.basename(file_path)}"\r\n'.encode()
        )
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.write(open(file_path, "rb").read())
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        base + path,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_job(base: str, job_id: str, timeout: float = 180.0) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        job = get_json(f"{base}/api/jobs/{job_id}")
        if job["status"] in ("done", "error"):
            return job
        tag = f'{job["status"]} {job["step_now"]}/{job["step_total"]}'
        if tag != last:
            print(f"    {tag}  {job['elapsed']:.1f}s")
            last = tag
        time.sleep(0.8)
    raise TimeoutError(f"任务 {job_id} 超时未完成")


def check_image(base: str, url: str) -> tuple[bool, str]:
    with urllib.request.urlopen(base + url, timeout=30) as r:
        data = r.read()
        ctype = r.headers.get("Content-Type", "")
    head = data[:8]
    is_png = head == b"\x89PNG\r\n\x1a\n"
    return is_png, f"{len(data)/1024:.0f} KB, {ctype}, PNG签名={is_png}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    ok = True
    print(f"目标：{base}")
    print()

    # ---------------- 1. 文生图 ----------------
    print("[1] 文生图（sd-turbo / bf16 / turbo / 512x512）")
    status, data = post_form(
        base,
        "/api/generate",
        {
            "mode": "txt2img",
            "prompt": "a red panda astronaut floating in space, cinematic lighting",
            "negative": "blurry",
            "model": "sd-turbo",
            "precision": "bf16",
            "gear": "turbo",
            "width": 512,
            "height": 512,
            "seed": 4242,
            "batch": 1,
        },
    )
    print(f"    提交 HTTP {status} → job {data.get('id')}")
    if status != 200:
        print(f"    失败：{data}")
        return 1

    job = wait_job(base, data["id"])
    if job["status"] != "done":
        print(f"    任务失败：{job['error']}")
        ok = False
    else:
        st = job["meta"]["stage_seconds"]
        print(f"    完成：推理 {st['infer_total']:.2f}s（去噪 {st['denoise']:.2f}s）"
              f" 显存 {job['meta']['vram']['peak_allocated_gb']:.2f}GB")
        for u in job["images"]:
            good, info = check_image(base, u)
            print(f"    图片 {u}\n         {info}")
            ok = ok and good

    # ---------------- 2. 图生图 ----------------
    first_png = None
    if job["status"] == "done" and job["images"]:
        rel = job["images"][0].replace("/outputs/", "outputs/").replace("/", os.sep)
        candidate = os.path.join(PROJECT, rel)
        if os.path.exists(candidate):
            first_png = candidate

    print()
    print("[2] 图生图（上传上一步的图，strength 0.6）")
    if not first_png:
        print("    跳过：没有可用的初始图")
        ok = False
    else:
        status, data = post_form(
            base,
            "/api/generate",
            {
                "mode": "img2img",
                "prompt": "oil painting style, thick brush strokes",
                "negative": "blurry",
                "model": "sd-turbo",
                "precision": "bf16",
                "gear": "turbo",
                "width": 512,
                "height": 512,
                "seed": 4242,
                "batch": 1,
                "strength": 0.6,
            },
            file_field="image",
            file_path=first_png,
        )
        print(f"    提交 HTTP {status} → job {data.get('id')}")
        if status != 200:
            print(f"    失败：{data}")
            ok = False
        else:
            job2 = wait_job(base, data["id"])
            if job2["status"] != "done":
                print(f"    任务失败：{job2['error']}")
                ok = False
            else:
                print(f"    完成：{job2['meta']['stage_seconds']['infer_total']:.2f}s，"
                      f"mode={job2['meta']['mode']}，strength={job2['meta']['strength']}")
                for u in job2["images"]:
                    good, info = check_image(base, u)
                    print(f"    图片 {u}\n         {info}")
                    ok = ok and good

    # ---------------- 3. 历史接口 ----------------
    print()
    print("[3] /api/history")
    hist = get_json(f"{base}/api/history")
    print(f"    记录数 {len(hist['items'])}，最近一条 prompt：{hist['items'][0]['prompt'][:40] if hist['items'] else '-'}")

    print()
    print("结论：" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

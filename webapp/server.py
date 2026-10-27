"""本地 Web 界面：上传图片 / 文生图 / 图生图 + 挡位选择。

设计要点
--------
* **单工作线程串行执行**：只有一块 GPU，多个请求并发只会互相抢显存并拖慢彼此，
  因此用队列把生图任务排队，一次只跑一个。
* **管线常驻**：Engine 在进程内缓存管线，同精度下连续出图不必重新加载权重。
* **不依赖外网**：前端是本地静态文件，不用 CDN、不用打包工具。

启动：
    C:\\Users\\PPXW\\miniconda3\\envs\\genimg\\python.exe -m webapp.server
    或   powershell -File web.ps1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from genimg.advice import check_request, effective_steps
from genimg.engine import Engine, GenRequest, is_repo_cached
from genimg.sizing import resolve as resolve_size
from genimg.presets import (
    MODEL_PROFILES,
    PRECISION_GEARS,
    QUALITY_GEARS,
    check_precision_support,
    probe_hardware,
    quality_gears_for,
    resolve_model,
    resolve_precision,
    resolve_quality,
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
WEB_OUTPUT = os.path.join(PROJECT, "outputs", "web")
os.makedirs(WEB_OUTPUT, exist_ok=True)

# 必须最早设定 HF 缓存路径：huggingface_hub 在首次 import 时就固化常量，
# 之后再改 HF_HOME 无效（否则会出现"模型明明在本地却找不到"）。
from genimg.runtime import configure_hf  # noqa: E402

configure_hf(PROJECT, source="mirror", verbose=False)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_BATCH = 4
MIN_SIZE, MAX_SIZE = 256, 1536

app = FastAPI(title="genimg 本地生图", docs_url=None, redoc_url=None)
app.mount("/outputs", StaticFiles(directory=os.path.join(PROJECT, "outputs")), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------------------
# 任务队列
# --------------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    mode: str
    prompt: str
    params: dict
    status: str = "queued"  # queued | running | done | error
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    step_now: int = 0
    step_total: int = 0
    images: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    error: str = ""
    warning: str = ""

    def public(self) -> dict:
        elapsed = 0.0
        if self.started:
            elapsed = (self.finished or time.time()) - self.started
        return {
            "id": self.id,
            "mode": self.mode,
            "prompt": self.prompt,
            "params": self.params,
            "status": self.status,
            "created": self.created,
            "elapsed": round(elapsed, 2),
            "step_now": self.step_now,
            "step_total": self.step_total,
            "images": self.images,
            "meta": self.meta,
            "error": self.error,
            "warning": self.warning,
        }


JOBS: dict[str, Job] = {}
JOB_ORDER: list[str] = []
QUEUE: "queue.Queue[str]" = queue.Queue()

_engine: Engine | None = None
_engine_lock = threading.Lock()

# 预热状态：服务启动后就把默认挡位加载进显存，避免第一次点“生成”卡十几秒
PRELOAD: dict = {
    "enabled": False,
    "status": "off",  # off | pending | loading | ready | error
    "model": "",
    "precision": "",
    "gear": "",
    "seconds": None,
    "error": "",
}

_worker_started = threading.Event()


def get_engine() -> Engine:
    """延迟创建 Engine（在 worker 线程里初始化，保证 CUDA 上下文只用一条线程）。"""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Engine(PROJECT, hf_source="mirror", verbose=False)
        return _engine


def _run_preload() -> None:
    """在 worker 线程里预热默认挡位（保持 GPU 只在一个线程里使用）。"""
    PRELOAD["status"] = "loading"
    t0 = time.time()
    try:
        engine = get_engine()
        profile = resolve_model(PRELOAD["model"])
        prec = resolve_precision(PRELOAD["precision"])
        quality = resolve_quality(PRELOAD["gear"], profile)
        engine.load(profile, prec, quality, resolution=512)
        # 加载权重只是第一步：首次前向还要做 kernel 选择/图优化（实测约 0.5s）。
        # 这里空跑一步把它也预热掉，让用户第一次点「生成」就是稳态速度。
        warm_dir = os.path.join(WEB_OUTPUT, "_preload")
        os.makedirs(warm_dir, exist_ok=True)
        engine.run(
            GenRequest(
                mode="txt2img",
                prompt="warmup",
                model=profile.key,
                precision=prec.key,
                gear=quality.key,
                width=512,
                height=512,
                seed=0,
                steps=1,
                guidance=0.0,
                out_dir=warm_dir,
            ),
            profile,
            prec,
            quality,
        )
        for fname in os.listdir(warm_dir):  # 预热图没保留价值，顺手删掉
            try:
                os.remove(os.path.join(warm_dir, fname))
            except OSError:
                pass

        PRELOAD["seconds"] = round(time.time() - t0, 2)
        PRELOAD["status"] = "ready"
        print(
            f"[preload] 已预热 {profile.key} / {prec.key} / {quality.key}"
            f"（{PRELOAD['seconds']}s），现在打开页面即可直接出图"
        )
    except Exception as exc:  # noqa: BLE001
        PRELOAD["status"] = "error"
        PRELOAD["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[preload] 预热失败：{PRELOAD['error']}（不影响使用，首次生成时会重新加载）")


def ensure_worker() -> None:
    """启动唯一的工作线程（只启一次）。"""
    if not _worker_started.is_set():
        _worker_started.set()
        threading.Thread(target=_worker, name="genimg-worker", daemon=True).start()


def _worker() -> None:
    if PRELOAD["enabled"]:
        _run_preload()

    while True:
        job_id = QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            QUEUE.task_done()
            continue

        job.status = "running"
        job.started = time.time()
        try:
            engine = get_engine()
            profile = resolve_model(job.params["model"])
            precision = resolve_precision(job.params["precision"])
            quality = resolve_quality(job.params["gear"], profile)

            def _progress(done: int, total: int, elapsed: float) -> None:
                job.step_now = done
                job.step_total = total

            req = GenRequest(
                mode=job.mode,
                prompt=job.prompt,
                negative_prompt=job.params.get("negative", ""),
                model=profile.key,
                precision=precision.key,
                gear=quality.key,
                width=job.params["width"],
                height=job.params["height"],
                steps=job.params.get("steps"),
                guidance=job.params.get("guidance"),
                seed=job.params.get("seed"),
                batch=job.params["batch"],
                strength=job.params.get("strength", 0.6),
                init_image=job.params.get("init_image"),
                placement="auto",
                out_dir=job.params.get("job_dir"),
            )
            res = engine.run(req, profile, precision, quality, progress_cb=_progress)

            if not res.ok:
                job.status = "error"
                job.error = res.error
            else:
                job.status = "done"
                outputs = list(res.meta.get("outputs", []))
                job.meta = res.meta
                # meta.outputs 是相对项目根的路径（形如 outputs/web/<id>/x.png），转成 URL
                job.images = ["/" + p.replace("\\", "/") for p in outputs]
                job.step_now = len(res.meta.get("step_seconds", [])) or job.step_now
                job.step_total = job.step_now or job.step_total
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()
            QUEUE.task_done()


# 注：工作线程在 main() 里根据预热配置后再启动（ensure_worker），
# 保证 worker 启动时 PRELOAD 已经赋值。


# --------------------------------------------------------------------------------------
# 接口
# --------------------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/catalog")
def catalog() -> dict:
    hw = probe_hardware()
    models = []
    for key, prof in MODEL_PROFILES.items():
        models.append(
            {
                "key": key,
                "label": prof.label,
                "repo": prof.repo,
                "native": prof.native_size,
                "default_gear": prof.default_gear,
                "gears": quality_gears_for(prof),
                "note": prof.note,
                "cached": is_repo_cached(prof.repo, PROJECT),
                "blocked_precisions": list(getattr(prof, "blocked_precisions", ())),
                "blocked_reason": getattr(prof, "blocked_reason", ""),
                "supports_negative": prof.supports_negative,
            }
        )
    precisions = []
    for key, gear in PRECISION_GEARS.items():
        why = check_precision_support(gear, hw)
        precisions.append({"key": key, "label": gear.label, "note": gear.note, "ok": why is None, "reason": why or ""})
    gears = []
    for key, gear in QUALITY_GEARS.items():
        gears.append(
            {
                "key": key,
                "label": gear.label,
                "steps": gear.steps,
                "guidance": gear.guidance,
                "note": gear.note,
                "only_family": list(gear.only_family),
            }
        )
    return {
        "hardware": {
            "summary": hw.summary(),
            "name": hw.name,
            "cc": f"sm_{hw.cc[0]}{hw.cc[1]}",
            "vram_gb": round(hw.total_gb, 1),
            "driver": hw.driver,
            "torch": hw.torch_version,
            "cuda": hw.cuda_version,
            "fp8": hw.supports_fp8,
        },
        "models": models,
        "precisions": precisions,
        "gears": gears,
        "defaults": {"model": "sd-turbo", "precision": "bf16", "gear": "turbo", "width": 512, "height": 512},
        "preload": dict(PRELOAD),
    }


@app.post("/api/generate")
def generate(
    mode: str = Form("txt2img"),
    prompt: str = Form(""),
    negative: str = Form(""),
    model: str = Form("sd-turbo"),
    precision: str = Form("fp16"),
    gear: str = Form(""),
    ratio: str = Form("auto"),
    pixels: int = Form(512),
    size: str = Form("auto"),
    width: int = Form(512),
    height: int = Form(512),
    steps: str = Form(""),
    guidance: str = Form(""),
    seed: str = Form(""),
    batch: int = Form(1),
    strength: float = Form(0.6),
    image: UploadFile | None = File(None),
) -> JSONResponse:
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="提示词不能为空")
    if mode not in ("txt2img", "img2img"):
        raise HTTPException(status_code=400, detail="mode 只能是 txt2img 或 img2img")
    if mode == "img2img" and image is None:
        raise HTTPException(status_code=400, detail="图生图需要先上传一张图片")

    ratio_spec = (ratio or "auto").strip()
    # 旧客户端可能还在传 size（或 custom + width/height），保留兼容
    if ratio_spec == "auto" and (size or "").strip() not in ("", "auto"):
        ratio_spec = (size or "").strip()
    if ratio_spec == "custom":
        for name, val in (("width", width), ("height", height)):
            if not (MIN_SIZE <= val <= MAX_SIZE) or val % 8 != 0:
                raise HTTPException(status_code=400, detail=f"{name} 需为 {MIN_SIZE}–{MAX_SIZE} 之间且能被 8 整除")
    if not (1 <= batch <= MAX_BATCH):
        raise HTTPException(status_code=400, detail=f"一次最多生成 {MAX_BATCH} 张")
    if not (0.0 <= strength <= 1.0):
        raise HTTPException(status_code=400, detail="改动强度需在 0–1 之间")

    try:
        profile = resolve_model(model)
        prec = resolve_precision(precision)
        gear_key = gear or profile.default_gear
        q = resolve_quality(gear_key, profile)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip('"'))

    hw = probe_hardware()
    why = check_precision_support(prec, hw, profile)
    if why:
        raise HTTPException(status_code=400, detail=f"精度挡位 {prec.key} 在本机不可用：{why}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(WEB_OUTPUT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    init_path = None
    init_size = None
    if mode == "img2img" and image is not None:
        raw = image.file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="图片超过 20MB")
        try:
            from PIL import Image

            probe = Image.open(io.BytesIO(raw))
            init_size = probe.size
            probe.verify()
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="上传的文件不是有效图片")
        from PIL import Image as _Image

        init_path = os.path.join(job_dir, "init.png")
        _Image.open(io.BytesIO(raw)).convert("RGB").save(init_path)

    # 尺寸优先级：显式宽高 > 显式比例 > 提示词画幅 > 参考图 > 模型原生
    # 「像素」档是所有分支共用的基准边长：自动只决定比例，不该把尺度也一并吞掉
    px = pixels if pixels in (512, 768, 1024) else 512
    if ratio_spec == "custom":
        decision = resolve_size(
            width=width,
            height=height,
            prompt=prompt,
            model_key=profile.key,
            init_size=init_size,
            model_native=profile.native_size,
        )
    elif ratio_spec == "auto":
        decision = resolve_size(
            base_override=px,
            prompt=prompt,
            model_key=profile.key,
            init_size=init_size,
            model_native=profile.native_size,
        )
    else:
        decision = resolve_size(
            spec=ratio_spec,
            base_override=px,
            prompt=prompt,
            model_key=profile.key,
            init_size=init_size,
            model_native=profile.native_size,
        )

    warning = ""
    if not is_repo_cached(profile.repo, PROJECT):
        warning = f"模型 {profile.key} 尚未下载，首次生成会先下载权重（可能数分钟）"

    job = Job(
        id=job_id,
        mode=mode,
        prompt=prompt,
        params={
            "negative": negative.strip(),
            "model": profile.key,
            "precision": prec.key,
            "gear": q.key,
            "width": decision.width,
            "height": decision.height,
            "size_source": decision.source,
            "size_note": decision.note,
            "size_ratio": decision.ratio,
            "steps": int(steps) if str(steps).strip() else None,
            "guidance": float(guidance) if str(guidance).strip() else None,
            "seed": int(seed) if str(seed).strip() else None,
            "batch": batch,
            "strength": strength,
            "init_image": init_path,
            "job_dir": job_dir,
        },
        warning=warning,
    )
    JOBS[job_id] = job
    JOB_ORDER.append(job_id)

    # 历史只保留最近 60 条，避免内存无限增长
    while len(JOB_ORDER) > 60:
        old = JOB_ORDER.pop(0)
        JOBS.pop(old, None)

    QUEUE.put(job_id)
    ensure_worker()
    return JSONResponse({"id": job_id, "status": "queued", "warning": warning})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return job.public()


@app.get("/api/history")
def history(limit: int = 12) -> dict:
    items = []
    for jid in reversed(JOB_ORDER):
        j = JOBS.get(jid)
        if j is None or j.status != "done":
            continue
        items.append(j.public())
        if len(items) >= limit:
            break
    return {"items": items}


@app.get("/api/advice")
def advice(
    mode: str = "txt2img",
    prompt: str = "",
    model: str = "sd-turbo",
    gear: str = "",
    strength: float = 0.6,
    steps: str = "",
    guidance: str = "",
) -> dict:
    """生成前体检：给界面实时提示（图生图步数换算 / CFG / 中文提示词）。"""
    try:
        profile = resolve_model(model)
        q = resolve_quality(gear or profile.default_gear, profile)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc).strip('"'))

    st = int(steps) if str(steps).strip().isdigit() else q.steps
    gd = float(guidance) if str(guidance).strip() else q.guidance
    tips = check_request(
        mode=mode, prompt=prompt, steps=st, strength=strength, guidance=gd, model_key=profile.key
    )

    # 风格语义展开：提前告诉用户"你写的风格名会被换成什么"，而不是生成完才知道
    expanded, expanded_applied = prompt, []
    try:
        from genimg.style_map import expand_styles

        ex = expand_styles(prompt)
        if ex.changed:
            expanded, expanded_applied = ex.prompt, [[a, b] for a, b in ex.applied]
            tips = tips + [
                f"风格已自动展开（{'、'.join(a[0] for a in ex.applied)}）：实际送入模型的是「{ex.prompt}」"
            ]
    except Exception:  # noqa: BLE001
        pass

    return {
        "advice": tips,
        "effective_steps": effective_steps(st, strength) if mode == "img2img" else st,
        "prompt_effective": expanded,
        "style_expanded": expanded_applied,
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "engine_loaded": _engine is not None,
        "queued": QUEUE.qsize(),
        "preload": dict(PRELOAD),
    }


def _port_free(host: str, port: int) -> bool:
    """检查端口是否可绑定。"""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="genimg 本地 Web 界面")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    ap.add_argument("--preload", action="store_true", help="启动后先把默认挡位加载进显存（首次生图不用等权重加载）")
    ap.add_argument("--preload-model", default="sd-turbo", help="预热的模型，默认 sd-turbo")
    ap.add_argument("--preload-precision", default="bf16", help="预热的精度挡位，默认 bf16")
    ap.add_argument("--preload-gear", default="turbo", help="预热的画质挡位，默认 turbo")
    args = ap.parse_args()

    import uvicorn

    # 端口被占用时自动往后找，避免上一次的进程还占着 7860 导致直接启动失败
    if not _port_free(args.host, args.port):
        for candidate in range(args.port + 1, args.port + 11):
            if _port_free(args.host, candidate):
                print(f"[warn] 端口 {args.port} 已被占用，自动改用 {candidate}")
                args.port = candidate
                break
        else:
            print(f"[错误] {args.port}–{args.port + 10} 均被占用，请用 --port 指定其他端口")
            return 1

    url = f"http://{args.host}:{args.port}"
    print("=" * 78)
    print("genimg 本地生图界面")
    print("=" * 78)
    print(f"  地址：{url}")
    print("  提示：首次生成会加载权重（几秒到十几秒），之后同精度下复用管线")
    print("  停止：在此窗口按 Ctrl+C")
    print("=" * 78)

    if args.preload:
        PRELOAD.update(
            enabled=True,
            status="pending",
            model=args.preload_model,
            precision=args.preload_precision,
            gear=args.preload_gear,
        )
        print(f"  预热：{args.preload_model} / {args.preload_precision} / {args.preload_gear}（后台加载）")
    else:
        print("  预热：未启用（第一次生成时才加载权重，会多等几秒）")

    if args.open:
        import threading as _t
        import webbrowser

        _t.Timer(1.5, lambda: webbrowser.open(url)).start()

    ensure_worker()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

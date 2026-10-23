"""生图引擎：管线装配、显存放置、文生图/图生图推理与计时。"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .advice import check_request, effective_steps
from .sizing import fit_image
from .style_map import expand_styles
from .zh_prompt import translate
from .presets import (
    LORA_REGISTRY,
    ModelProfile,
    PrecisionGear,
    QualityGear,
    check_precision_support,
    probe_hardware,
)
from .runtime import (
    PlacementPlan,
    StageTimer,
    VramMonitor,
    apply_placement,
    apply_precision,
    configure_hf,
    empty_cuda_cache,
    module_bytes,
    pipeline_weight_bytes,
    plan_placement,
    set_cudnn_benchmark,
    silence_known_warnings,
)

# 模型家族 -> (文生图类名, 图生图类名)
PIPELINE_CLASSES = {
    "sd15": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"),
    "sd-turbo": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"),
    "sdxl": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline"),
}


# --------------------------------------------------------------------------------------
# 请求描述
# --------------------------------------------------------------------------------------
@dataclass
class GenRequest:
    """一次生图请求的完整参数（已合并 CLI + 提示词指令 + 默认值）。"""

    mode: str = "txt2img"  # txt2img | img2img
    prompt: str = ""
    negative_prompt: str = ""
    model: str = "sdxl"
    precision: str = "fp16"
    gear: str = ""
    width: int = 512
    height: int = 512
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    batch: int = 1
    strength: float = 0.5
    init_image: str | None = None
    placement: str = "auto"
    out_dir: str | None = None  # 自定义输出目录（Web 界面按任务号归档时用）
    extra_call_kwargs: dict = field(default_factory=dict)  # 透传给 diffusers 管线的额外参数
    translate: str = "dict"  # 中文提示词处理：dict（离线词典） | off
    style_expand: str = "auto"  # 风格语义展开：auto（抽象风格名→具体视觉特征） | off
    size_source: str = ""  # 尺寸来源（cli / 提示词「3:4」 / 参考图 / default），仅用于回显
    size_note: str = ""  # 尺寸与参考图处理的说明（是否缩放、是否留白）
    def size_str(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class GenResult:
    """一次生图结果。"""

    ok: bool
    images: list[str] = field(default_factory=list)
    seconds: float = 0.0
    stage_seconds: dict = field(default_factory=dict)
    vram: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    error: str = ""


# --------------------------------------------------------------------------------------
# 管线装配辅助
# --------------------------------------------------------------------------------------
def _resolve_pipeline_classes(profile: ModelProfile):
    import diffusers

    t2i_name, i2i_name = PIPELINE_CLASSES[profile.family]
    return getattr(diffusers, t2i_name), getattr(diffusers, i2i_name)


def _build_scheduler(pipe, kind: str, verbose: bool = True):
    from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

    if kind == "default":
        return pipe.scheduler
    cfg = dict(pipe.scheduler.config)
    try:
        if kind == "euler":
            sched = EulerDiscreteScheduler.from_config(cfg)
        elif kind == "euler_trailing":
            sched = EulerDiscreteScheduler.from_config(cfg, timestep_spacing="trailing")
        elif kind == "dpmpp_2m":
            sched = DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="dpmsolver++")
        else:
            raise ValueError(kind)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[sched] 切换调度器 {kind} 失败（{exc}），沿用仓库默认调度器")
        return pipe.scheduler
    return sched


@dataclass
class LoadedPipeline:
    """一条已装配好的管线及其元信息。"""

    txt2img: Any
    img2img: Any
    profile: ModelProfile
    precision: PrecisionGear
    lora_key: str | None
    load_seconds: float
    placement: PlacementPlan
    dtype: str
    vae_dtype: str
    scheduler_kind: str = "default"
    variant: str | None = None

    @property
    def key(self) -> tuple:
        return (self.profile.key, self.precision.key, self.lora_key)


def describe_components(pipe) -> str:
    """各组件精度与体积摘要（排查"精度挡位没生效"类问题用）。"""
    parts = []
    for attr in ("unet", "transformer", "vae", "text_encoder", "text_encoder_2"):
        m = getattr(pipe, attr, None)
        if m is None:
            continue
        try:
            dt = next(m.parameters()).dtype
        except StopIteration:
            dt = "?"
        parts.append(f"{attr}={str(dt).replace('torch.', '')} {module_bytes(m)/1024**3:.2f}GB")
    total = pipeline_weight_bytes(pipe) / 1024**3
    return ", ".join(parts) + f" | 合计 {total:.2f}GB"


def ensure_dtypes(pipe, main_dtype, vae_dtype, verbose: bool = True) -> list[str]:
    """回读各组件实际精度，与目标不符则纠正。

    为什么需要这一步：diffusers 里能静默改变精度的路径不止一条（例如
    ``Img2ImgPipeline.from_pipe()`` 未传 dtype 时会把整条管线转成 float32）。
    既然传参不可信，就在装配完成后回读校验一次。
    """
    fixes: list[str] = []
    for attr in ("unet", "transformer", "text_encoder", "text_encoder_2", "text_encoder_3"):
        m = getattr(pipe, attr, None)
        if m is None:
            continue
        try:
            actual = next(m.parameters()).dtype
        except StopIteration:
            continue
        if actual != main_dtype:
            m.to(main_dtype)
            fixes.append(f"{attr}: {str(actual).replace('torch.', '')} -> {str(main_dtype).replace('torch.', '')}")

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        try:
            actual = next(vae.parameters()).dtype
            if actual != vae_dtype:
                vae.to(vae_dtype)
                fixes.append(
                    f"vae: {str(actual).replace('torch.', '')} -> {str(vae_dtype).replace('torch.', '')}"
                )
        except StopIteration:
            pass

    if fixes and verbose:
        print("[engine] 精度校正：" + "；".join(fixes))
    return fixes


def _from_pretrained_compat(fn, repo: str, kwargs: dict, verbose: bool = True):

    """调用 from_pretrained，兼容 ``dtype`` 与旧的 ``torch_dtype`` 参数名。

    diffusers 0.41 已把 ``torch_dtype`` 标为弃用（将在 1.0 移除），但旧版本只认
    ``torch_dtype``，因此优先用新名、失败则回退。
    """
    kwargs = dict(kwargs)
    if "torch_dtype" in kwargs:
        kwargs["dtype"] = kwargs.pop("torch_dtype")
    try:
        return fn(repo, **kwargs)
    except TypeError as exc:
        if "dtype" in kwargs and "dtype" in str(exc):
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            if verbose:
                print("[engine] 当前 diffusers 版本不认 dtype 参数，回退 torch_dtype")
            return fn(repo, **kwargs)
        raise


# --------------------------------------------------------------------------------------
# 引擎
# --------------------------------------------------------------------------------------
class Engine:
    """管线缓存 + 推理调度。

    同一次进程内，按 (模型, 精度, LoRA) 复用管线：切画质挡位不必重新加载权重。
    换精度挡位必须重新加载，因为权重 dtype 变了。
    """

    def __init__(
        self,
        project_root: str,
        hf_source: str = "huggingface",
        hf_proxy: str | None = None,
        verbose: bool = True,
        use_safety_checker: bool = False,
    ):
        self.project_root = project_root
        self.verbose = verbose
        self.use_safety_checker = use_safety_checker
        silence_known_warnings()
        self._cache: dict[tuple, LoadedPipeline] = {}
        self.hw = probe_hardware()
        self.hf_cfg = configure_hf(project_root, source=hf_source, proxy=hf_proxy, verbose=verbose)
        # 显式给 diffusers 传 cache_dir：huggingface_hub 的缓存路径在首次 import 时
        # 就固化了，之后再改 HF_HOME 对它无效（已踩过：Web 服务的 /api/catalog 先查了
        # 缓存，导致后续引擎看不到项目内的权重）。传参不依赖环境变量时序，最稳。
        self.hub_cache = os.path.join(self.hf_cfg.cache_dir, "hub")

    # ---------------------------------------------------------------- 装配
    def load(
        self,
        profile: ModelProfile,
        precision: PrecisionGear,
        quality: QualityGear,
        placement_override: str = "auto",
        resolution: int = 512,
        cudnn_benchmark: bool = False,
    ) -> LoadedPipeline:
        lora_key = quality.lora
        cache_key = (profile.key, precision.key, lora_key, resolution, placement_override, cudnn_benchmark)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # 画质挡位可能换了调度器，而管线是复用的（同精度下切挡位不重载）——这里补上
            if cached.scheduler_kind != quality.scheduler:
                cached.txt2img.scheduler = _build_scheduler(cached.txt2img, quality.scheduler, self.verbose)
                cached.img2img.scheduler = cached.txt2img.scheduler
                cached.scheduler_kind = quality.scheduler
                if self.verbose:
                    print(f"[engine] 复用管线并重设调度器 -> {quality.scheduler}")
            if self.verbose:
                print(
                    f"[engine] 复用已加载管线：{profile.key} / {precision.key}"
                    + (f" / {lora_key}" if lora_key else "")
                )
            return cached

        # 关键：不同精度挡位需要不同权重的管线。若把旧管线继续留在显存里，新管线的
        # 放置决策会看到被旧管线占掉的显存，从而被误判成需要 offload（甚至逐层 offload）。
        # 因此换挡时先释放旧管线。
        self._evict_all()

        unsupported = check_precision_support(precision, self.hw, profile)
        if unsupported:
            raise RuntimeError(f"精度挡位 {precision.key} 在本机不可用：{unsupported}")

        import torch

        t0 = time.perf_counter()
        dtype = apply_precision(precision, self.hw)
        set_cudnn_benchmark(cudnn_benchmark)

        cls_t2i, cls_i2i = _resolve_pipeline_classes(profile)
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        # add_watermarker 仅在 XL 系列管线支持，其他管线会打印无意义的警告
        if profile.family == "sdxl":
            kwargs["add_watermarker"] = False
        # 变体选择：优先用仓库发布的高性能变体；precision 为 fp16 时优先 fp16 变体（读盘量减半）
        variant = profile.variant
        if variant is None and precision.dtype == "float16":
            variant = "fp16"
        if variant and variant != "fp32":
            kwargs["variant"] = variant
        # 安全检查器：本地个人工具默认关闭（省下载、省显存）；需要时用 --safety-checker 打开
        if not self.use_safety_checker and profile.family in ("sd15", "sd-turbo"):
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

        if self.verbose:
            print(f"[engine] 加载 {profile.repo}（dtype={dtype}，variant={kwargs.get('variant', '默认')}）…")

        # 已完整缓存时必须完全离线加载：diffusers 默认会去 Hub 查元数据，
        # 而本机 huggingface.co 不可直连、hf-mirror 还会时不时的 403/429，
        # 结果就是"本地明明有模型却加载失败"（已实测复现）。
        cached_locally = is_repo_cached(profile.repo, self.project_root)
        kwargs["cache_dir"] = self.hub_cache
        if cached_locally:
            kwargs["local_files_only"] = True
            if self.verbose:
                print("[engine] 本地已缓存 → 离线加载（不访问网络）")
        # VAE 精度不在加载阶段单独指定。原因：把 VAE 提前加载成 fp32 后，管线内部的
        # "需要时把 latent 转成 VAE 精度" 逻辑不会触发（只有 SDXL 系有该逻辑，SD1.5 没有），
        # 于是 fp16 的 latent 撞上 fp32 的 VAE 权重会直接报 dtype 不匹配。
        # 正确做法：先按主精度整体加载，装配完成后再把 VAE 调到目标精度，
        # 并在推理最后一步把 latent 转成 VAE 精度（见 run() 里的 _cb 与 ensure_dtypes）。
        vae_dtype_name = precision.vae_dtype or precision.dtype
        try:
            pipe = _from_pretrained_compat(cls_t2i.from_pretrained, profile.repo, kwargs, self.verbose)
        except Exception as exc:  # noqa: BLE001
            # 文件缺失或网络受限时，给一句能落地的提示，而不是把原始堆栈丢给用户
            if not cached_locally:
                text = f"{type(exc).__name__}: {exc}"
                if any(k in text for k in ("Connection", "connect", "10061", "403", "429", "offline", "metadata")):
                    raise RuntimeError(
                        f"模型 {profile.key} 未完整缓存，且当前无法联网下载（{type(exc).__name__}）。"
                        f"请先运行 tools\\fetch_models.py 下载权重，或检查网络/代理设置。"
                    ) from exc
            # 变体缺失或参数不兼容时，退回不带 variant 的加载方式并说明
            if "variant" in kwargs:
                if self.verbose:
                    print(f"[engine] 指定变体加载失败（{type(exc).__name__}: {exc}），改用默认变体重试")
                kwargs.pop("variant")
                pipe = _from_pretrained_compat(cls_t2i.from_pretrained, profile.repo, kwargs, self.verbose)
            else:
                raise

        pipe.set_progress_bar_config(disable=True)

        if self.verbose:
            print(f"[engine] 权重分布：{describe_components(pipe)}")

        # 兜底：若上面没能按组件指定 VAE 精度，这里再纠正一次
        if vae_dtype_name != precision.dtype:
            try:
                current = next(pipe.vae.parameters()).dtype
                if str(current) != f"torch.{vae_dtype_name}":
                    pipe.vae.to(getattr(torch, vae_dtype_name))
            except StopIteration:
                pass

        # 加速 LoRA（Lightning 档）：官方发布的 394MB 小文件，融合进权重以省一次前向
        if lora_key:
            spec = LORA_REGISTRY[lora_key]
            if self.verbose:
                print(f"[engine] 挂载 LoRA {spec['repo']} / {spec['weight_name']}…")
            lora_kwargs = {
                "weight_name": spec["weight_name"],
                "adapter_name": lora_key,
                "cache_dir": self.hub_cache,
            }
            try:
                pipe.load_lora_weights(spec["repo"], **lora_kwargs)
            except TypeError:
                # 某些 diffusers 版本的 load_lora_weights 不接收 cache_dir
                lora_kwargs.pop("cache_dir")
                pipe.load_lora_weights(spec["repo"], **lora_kwargs)
            pipe.fuse_lora(lora_scale=1.0)
            if self.verbose:
                print("[engine] LoRA 已融合进 UNet")

        # 调度器
        pipe.scheduler = _build_scheduler(pipe, quality.scheduler, verbose=self.verbose)

        # 放置
        plan = plan_placement(pipe, resolution=resolution, override=placement_override, verbose=self.verbose)
        apply_placement(pipe, plan, verbose=self.verbose)

        load_seconds = time.perf_counter() - t0

        # 必须显式传 dtype：from_pipe 在未传 dtype 时会按 `or torch.float32` 把整条管线转成 fp32
        img2img = cls_i2i.from_pipe(pipe, dtype=dtype)
        img2img.set_progress_bar_config(disable=True)

        # 必须做的校验：from_pipe / to() 等路径可能静默改写精度，回读实际 dtype 并纠正
        ensure_dtypes(pipe, dtype, getattr(torch, vae_dtype_name), verbose=self.verbose)

        loaded = LoadedPipeline(
            txt2img=pipe,
            img2img=img2img,
            profile=profile,
            precision=precision,
            lora_key=lora_key,
            load_seconds=load_seconds,
            placement=plan,
            dtype=str(dtype).replace("torch.", ""),
            vae_dtype=vae_dtype_name,
            scheduler_kind=quality.scheduler,
            variant=kwargs.get("variant"),
        )
        self._cache[cache_key] = loaded
        return loaded

    def _evict_all(self) -> None:
        """清掉缓存管线并真正归还显存（换精度挡位前必须先做）。"""
        if not self._cache:
            return
        if self.verbose:
            print(f"[engine] 释放 {len(self._cache)} 条已缓存管线，归还显存")
        self._cache.clear()
        import gc

        gc.collect()
        empty_cuda_cache()

    def unload(self) -> None:
        self._evict_all()

    # ---------------------------------------------------------------- 推理
    def run(
        self,
        req: GenRequest,
        profile: ModelProfile,
        precision: PrecisionGear,
        quality: QualityGear,
        cudnn_benchmark: bool = False,
        timeout_seconds: float | None = None,
        progress_cb=None,
    ) -> GenResult:
        """执行一次生图（含 --batch 多张）。

        ``timeout_seconds``：通过步回调实现硬预算。超过即在下一个采样步抛出中止，
        用于保证不被某个不实用的挡位（如 12GB 显存上跑 fp32 SDXL）拖住。
        ``progress_cb(done, total, elapsed_s)``：每个采样步结束时回调，供 CLI / Web 显示进度。
        """
        import torch
        from PIL import Image, PngImagePlugin

        steps = req.steps if req.steps is not None else quality.steps
        guidance = req.guidance if req.guidance is not None else quality.guidance
        seed = req.seed

        # 中文提示词 → 英文（离线词典）。必须放在“生成前体检”之前：
        # 翻译后提示词变成英文，就不会再触发“中文提示词效果弱”的告警。
        tr = translate(req.prompt, mode=req.translate)
        neg_tr = translate(req.negative_prompt, mode=req.translate) if req.negative_prompt else None
        prompt_eff = tr.effective
        negative_eff = neg_tr.effective if neg_tr is not None else (req.negative_prompt or "")
        if self.verbose and tr.changed:
            print(f"[prompt] 中文提示词已转英文（{tr.method}）：{prompt_eff}")
            if tr.note:
                print(f"[prompt] {tr.note}")
        if self.verbose and tr.unknown:
            print(f"[prompt] 词典未覆盖的片段已原样保留：{'、'.join(tr.unknown[:5])}")

        # 风格语义展开：把"American comic style"这类 CLIP 听不懂的抽象名，
        # 换成具体视觉特征（粗黑描边/平涂/网点）。实测这是唯一有效的办法。
        style_expanded: list = []
        if req.style_expand != "off":
            ex = expand_styles(prompt_eff)
            if ex.changed:
                prompt_eff = ex.prompt
                style_expanded = [list(x) for x in ex.applied]
                if ex.negatives:
                    extra = ", ".join(ex.negatives)
                    negative_eff = f"{negative_eff}, {extra}" if negative_eff else extra
                if self.verbose:
                    for label, expansion in ex.applied:
                        print(f"[prompt] 风格展开（{label}）：{expansion}")

        # 生成前的"体检"：提前把踩坑点在命令行里说清楚（图生图步数换算 / CFG / 中文提示词）
        tips = check_request(
            mode=req.mode,
            prompt=prompt_eff,
            negative_prompt=negative_eff,
            steps=steps,
            strength=req.strength,
            guidance=guidance,
            model_key=profile.key,
        )
        eff_steps = effective_steps(steps, req.strength) if req.mode == "img2img" else steps
        if req.mode == "img2img" and eff_steps < 1:
            detail = tips[0] if tips else f"图生图实际采样步数为 0（{steps} 步 × 强度 {req.strength}）"
            return GenResult(ok=False, error=detail)
        if tips and self.verbose:
            for tip in tips:
                print(f"[提示] {tip}")

        try:
            loaded = self.load(
                profile,
                precision,
                quality,
                placement_override=req.placement,
                resolution=max(req.width, req.height),
                cudnn_benchmark=cudnn_benchmark,
            )
        except Exception as exc:  # noqa: BLE001
            return GenResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        pipe = loaded.txt2img if req.mode == "txt2img" else loaded.img2img
        timer = StageTimer(sync=True)

        init_image = None
        fit_note = ""
        if req.mode == "img2img":
            if not req.init_image or not os.path.exists(req.init_image):
                return GenResult(ok=False, error=f"图生图需要有效的初始图片：{req.init_image!r}")
            # 关键：不能 resize((w, h))，那会把横图压扁。等比缩放后模糊留白，
            # 既保住主体比例，又给模型一个可以"延展背景"的起点。
            _raw = Image.open(req.init_image).convert("RGB")
            init_image, fit_note = fit_image(_raw, req.width, req.height)

        init_latents = None

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator = generator.manual_seed(int(seed))

        step_times: list[float] = []
        _last = [time.perf_counter()]
        call_start = [0.0]

        # fp16 挡位下 VAE 保持 fp32（避开半精度解码的黑图）。此时管线内部的 latent 仍是
        # fp16，必须在**最后一步结束后**把它转成 VAE 精度再解码；中途转会把 fp32 的 latent
        # 送回 fp16 权重的 UNet，直接 dtype 不匹配。
        main_dtype = next(pipe.unet.parameters()).dtype
        vae_dtype = next(pipe.vae.parameters()).dtype
        need_latent_cast = vae_dtype != main_dtype
        cast_state = [False]

        # img2img 且 VAE 精度 != UNet 精度（fp16 挡位把 VAE 升到了 fp32）：diffusers 会把输入图
        # 先 cast 成 UNet 的 fp16，再挤进 fp32 VAE → "Input type (Half) and bias type (float)"。
        # 解法：自己用 VAE 精度编码出 4 通道 latent，管线见 shape[1]==4 就跳过 VAE 编码。
        if req.mode == "img2img" and need_latent_cast:
            with torch.no_grad():
                _t = pipe.image_processor.preprocess(
                    init_image, height=req.height, width=req.width
                ).to(device=pipe.vae.device, dtype=vae_dtype)
                _dist = pipe.vae.encode(_t).latent_dist
                _lat = _dist.sample(generator) if generator is not None else _dist.sample()
                init_latents = (_lat * pipe.vae.config.scaling_factor).to(main_dtype)

        def _cb(_pipe, step_index, timestep, cb_kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            step_times.append(now - _last[0])
            _last[0] = now
            # scheduler.timesteps 是 Tensor，不能写 `ts or []`（Tensor 的布尔求值会报错）
            ts = getattr(_pipe.scheduler, "timesteps", None)
            total_steps = len(ts) if ts is not None else 0
            if progress_cb is not None:
                try:
                    progress_cb(len(step_times), total_steps, now - (call_start[0] or now))
                except Exception:  # noqa: BLE001
                    pass
            if need_latent_cast and "latents" in cb_kwargs:
                # 注意：diffusers 传进来的 step_index 也可能是 tensor，同样不能直接用于布尔判断。
                # 这里用自己统计的步数：len(step_times) 即"当前是第几步"，从 1 起算。
                if total_steps and len(step_times) == total_steps:
                    cb_kwargs["latents"] = cb_kwargs["latents"].to(vae_dtype)
                    cast_state[0] = True
            if timeout_seconds and call_start[0] and (now - call_start[0]) > timeout_seconds:
                raise TimeoutError(
                    f"单次生图已超过 {timeout_seconds:.0f} 秒预算（已跑 {len(step_times)}/{steps} 步），主动中止"
                )
            return cb_kwargs

        call_kwargs: dict[str, Any] = {
            "prompt": prompt_eff,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "num_images_per_prompt": max(1, req.batch),
            "callback_on_step_end": _cb,
        }
        if need_latent_cast:
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        if negative_eff and guidance > 1.0:
            call_kwargs["negative_prompt"] = negative_eff
        if req.mode == "txt2img":
            call_kwargs["width"] = req.width
            call_kwargs["height"] = req.height
        else:
            call_kwargs["image"] = init_latents if init_latents is not None else init_image
            call_kwargs["strength"] = req.strength
        # 透传参数（例如 SDXL 的 original_size / target_size micro-conditioning）
        if req.extra_call_kwargs:
            call_kwargs.update(req.extra_call_kwargs)

        out_dir = req.out_dir or os.path.join(self.project_root, "outputs", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)

        with VramMonitor() as vram:
            # fp16 挡位：VAE 是 fp32，而管线交给 decode 的 latent 仍是 fp16。
            # 不依赖 callback 的时序，直接在解码入口转一次 dtype，结束再恢复。
            _orig_decode = None
            if need_latent_cast:
                _orig_decode = pipe.vae.decode

                def _decode_any_dtype(latents, *a, **kw):
                    return _orig_decode(latents.to(vae_dtype), *a, **kw)

                pipe.vae.decode = _decode_any_dtype
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _last[0] = time.perf_counter()
                call_start[0] = _last[0]
                with timer.stage("infer_total"):
                    output = pipe(**call_kwargs)
            except Exception as exc:  # noqa: BLE001
                import traceback

                tb = traceback.format_exc()
                if self.verbose or os.environ.get("GENIMG_TRACEBACK"):
                    print(tb)
                return GenResult(ok=False, error=f"{type(exc).__name__}: {exc}", meta={"traceback": tb})
            finally:
                if _orig_decode is not None:
                    pipe.vae.decode = _orig_decode

        images = output.images
        denoise = sum(step_times)
        timer.stages["denoise"] = denoise
        timer.stages["encode_decode"] = max(0.0, timer.get("infer_total") - denoise)

        saved: list[str] = []
        for idx, img in enumerate(images):
            one_seed = None
            if seed is not None:
                one_seed = int(seed) + idx
            meta = PngImagePlugin.PngInfo()
            meta.add_text("prompt", prompt_eff)
            if req.prompt != prompt_eff:
                meta.add_text("prompt_original", req.prompt)
            if negative_eff:
                meta.add_text("negative_prompt", negative_eff)
            meta.add_text("model", profile.repo)
            meta.add_text("precision_gear", precision.key)
            meta.add_text("quality_gear", quality.key)
            meta.add_text("steps", str(steps))
            meta.add_text("guidance", str(guidance))
            if req.size_source:
                meta.add_text("size_source", req.size_source)
            if fit_note:
                meta.add_text("init_fit", fit_note)
            if one_seed is not None:
                meta.add_text("seed", str(one_seed))
            fname = (
                f"{profile.key}_{precision.key}_{quality.key}"
                f"_s{one_seed if one_seed is not None else 'rand'}_{idx:02d}.png"
            )
            path = os.path.join(out_dir, fname)
            img.save(path, pnginfo=meta)
            saved.append(path)

        digest = hashlib.sha256()
        with open(saved[0], "rb") as fh:
            digest.update(fh.read())

        meta_out = {
            "mode": req.mode,
            "prompt": prompt_eff,
            "prompt_original": req.prompt,
            "prompt_translated": tr.changed,
            "style_expanded": style_expanded,
            "translation": {
                "method": tr.method,
                "replaced": [list(x) for x in tr.replaced],
                "unknown": tr.unknown,
                "note": tr.note,
            },
            "negative_prompt": negative_eff,
            "model_key": profile.key,
            "model_repo": profile.repo,
            "model_label": profile.label,
            "precision_gear": precision.key,
            "precision_label": precision.label,
            "quality_gear": quality.key,
            "quality_label": quality.label,
            "lora": loaded.lora_key,
            "scheduler": quality.scheduler,
            "width": req.width,
            "height": req.height,
            "size_source": req.size_source,
            "size_note": req.size_note,
            "init_fit": fit_note,
            "steps": steps,
            "effective_steps": eff_steps,
            "advice": tips,
            "guidance": guidance,
            "cfg_enabled": guidance > 1.0,
            "seed": seed,
            "batch": len(saved),
            "strength": req.strength if req.mode == "img2img" else None,
            "dtype": loaded.dtype,
            "vae_dtype": loaded.vae_dtype,
            "latent_cast_applied": cast_state[0] if need_latent_cast else None,
            "placement": loaded.placement.mode,
            "load_seconds": round(loaded.load_seconds, 3),
            "stage_seconds": {k: round(v, 4) for k, v in timer.stages.items()},
            "step_seconds": [round(s, 4) for s in step_times],
            "infer_seconds": round(timer.get("infer_total"), 3),
            "wall_seconds": round(loaded.load_seconds + timer.get("infer_total"), 3),
            "vram": dataclasses.asdict(vram.sample),
            "outputs": [os.path.relpath(p, self.project_root) for p in saved],
            "sha256_first": digest.hexdigest(),
            "gpu": self.hw.name,
            "gpu_cc": f"sm_{self.hw.cc[0]}{self.hw.cc[1]}",
            "torch": self.hw.torch_version,
            "cuda": self.hw.cuda_version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        return GenResult(
            ok=True,
            images=saved,
            seconds=timer.get("infer_total"),
            stage_seconds=dict(timer.stages),
            vram=dataclasses.asdict(vram.sample),
            meta=meta_out,
        )


# --------------------------------------------------------------------------------------
# 本地缓存查询
# --------------------------------------------------------------------------------------
def is_repo_cached(repo: str, project_root: str) -> bool:
    """判断仓库是否已在本地缓存里（用于提示"是否需要下载"）。

    优先用 huggingface_hub 自己的缓存查询（能正确处理符号链接与多快照），
    不可用时回退到目录名判断。
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo_id=repo, filename="model_index.json")
        if isinstance(hit, str) and os.path.exists(hit):
            return True
    except Exception:  # noqa: BLE001
        pass

    hub = os.path.join(project_root, "models", "hub")
    snaps = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots")
    if not os.path.isdir(snaps):
        return False
    for commit in os.listdir(snaps):
        if os.path.exists(os.path.join(snaps, commit, "model_index.json")):
            return True
    return False

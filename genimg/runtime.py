"""运行时环境：代理探测、HF 下载配置、精度落地、显存放置规划与计时工具。

本模块负责所有"跟本机环境打交道"的事，让 engine/cli 只关心生图逻辑。
"""

from __future__ import annotations

import contextlib
import os
import re
import time
import urllib.request
from dataclasses import dataclass

from .presets import HardwareInfo, PrecisionGear, probe_hardware

DEFAULT_PROXY_HINT = "http://10.10.100.199:3128"
_PAC_PROXY_RE = re.compile(r"PROXY\s+([0-9A-Za-z\.\-]+:\d+)", re.I)


# --------------------------------------------------------------------------------------
# 网络 / 下载配置
# --------------------------------------------------------------------------------------
def detect_system_proxy(verbose: bool = False) -> str | None:
    """读取 Windows 代理配置。

    本机是"PAC 自动配置"模式（GFWList 规则），WinINET 会自动分流，但 Python 的
    ``urllib`` / ``requests`` 不认 PAC，因此这里主动把 PAC 里的代理地址解析出来，
    再显式交给 Python 使用。
    """
    # 1) 环境变量优先（用户显式设置的尊重之）
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        if os.environ.get(var):
            return os.environ[var]

    # 2) 注册表里的固定代理 / PAC
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        )
        try:
            pac_url = winreg.QueryValueEx(key, "AutoConfigURL")[0]
        except FileNotFoundError:
            pac_url = None
        try:
            hostport = winreg.QueryValueEx(key, "ProxyServer")[0]
        except FileNotFoundError:
            hostport = None
        try:
            enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
        except FileNotFoundError:
            enabled = 0
        winreg.CloseKey(key)

        if enabled and hostport:
            return hostport if "://" in hostport else f"http://{hostport}"
        if pac_url:
            try:
                raw = urllib.request.urlopen(pac_url, timeout=8).read().decode("utf-8", "ignore")
                m = _PAC_PROXY_RE.search(raw)
                if m:
                    if verbose:
                        print(f"[net] 从 PAC 解析到代理：{m.group(1)}")
                    return f"http://{m.group(1)}"
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return None


@dataclass
class HfConfig:
    """一次运行使用的 HuggingFace 访问配置。"""

    endpoint: str
    proxy: str | None
    cache_dir: str
    source: str  # huggingface | mirror

    def describe(self) -> str:
        p = f"，代理 {self.proxy}" if self.proxy else "，直连"
        return f"HF 源 {self.source}（{self.endpoint}{p}），缓存 {self.cache_dir}"


def configure_hf(
    project_root: str,
    source: str = "huggingface",
    proxy: str | None = None,
    cache_dir: str | None = None,
    autodetect_proxy: bool = True,
    verbose: bool = True,
) -> HfConfig:
    """设置 HF_HOME / HF_ENDPOINT / 代理等环境变量。

    本机实测：``huggingface.co`` 直连会被拒（10061），必须走 PAC 代理；``hf-mirror.com``
    可直连且更快（6.7MB/s vs 代理 2.1MB/s），因此下载默认走 mirror。
    """
    cache = cache_dir or os.path.join(project_root, "models")
    os.makedirs(cache, exist_ok=True)

    if source == "mirror":
        endpoint, use_proxy = "https://hf-mirror.com", None
    else:
        endpoint, use_proxy = "https://huggingface.co", proxy
        if use_proxy is None and autodetect_proxy:
            use_proxy = detect_system_proxy(verbose=verbose)

    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = os.path.join(cache, "hub")
    os.environ["HF_ENDPOINT"] = endpoint
    if use_proxy:
        os.environ["HTTPS_PROXY"] = use_proxy
        os.environ["HTTP_PROXY"] = use_proxy
    else:
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("HTTP_PROXY", None)

    # xet 传输在高延迟/代理链路上容易卡住，统一走经典 HTTP 路径更可预期
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    cfg = HfConfig(endpoint=endpoint, proxy=use_proxy, cache_dir=cache, source=source)
    if verbose:
        print(f"[net] {cfg.describe()}")
    return cfg


# --------------------------------------------------------------------------------------
# 精度落地
# --------------------------------------------------------------------------------------
def apply_precision(gear: PrecisionGear, hw: HardwareInfo | None = None):
    """把精度挡位施加到 torch 全局开关上，返回主 dtype。"""
    import torch

    hw = hw or probe_hardware()

    # TF32 开关（只影响 fp32 路径的矩阵乘/卷积）
    torch.backends.cuda.matmul.allow_tf32 = bool(gear.tf32 and hw.supports_tf32)
    torch.backends.cudnn.allow_tf32 = bool(gear.tf32 and hw.supports_tf32)

    return getattr(torch, gear.dtype)


def set_cudnn_benchmark(enabled: bool) -> None:
    import torch

    torch.backends.cudnn.benchmark = bool(enabled)


def empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# 显存放置规划
# --------------------------------------------------------------------------------------
def module_bytes(module) -> int:
    """一个 nn.Module 在当前 dtype 下的常驻字节数。"""
    if module is None:
        return 0
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def pipeline_weight_bytes(pipe) -> int:
    """扩散管线常驻权重的总字节数（排除可选组件）。"""
    total = 0
    for attr in ("unet", "transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"):
        total += module_bytes(getattr(pipe, attr, None))
    return total


def largest_component_bytes(pipe) -> tuple[str, int]:
    """单个最大组件的字节数（offload 时驻留显存的其实是它，而不是权重总和）。"""
    best_name, best_bytes = "", 0
    for attr in ("unet", "transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
        b = module_bytes(getattr(pipe, attr, None))
        if b > best_bytes:
            best_name, best_bytes = attr, b
    return best_name, best_bytes


@dataclass
class PlacementPlan:
    """显存放置决策。"""

    mode: str  # gpu | model_offload | sequential_offload
    weights_gb: float
    free_gb: float
    required_gb: float
    reason: str = ""

    def describe(self) -> str:
        zh = {"gpu": "全量常驻显存", "model_offload": "模型级 CPU offload", "sequential_offload": "逐层 CPU offload"}
        return (
            f"放置策略 {self.mode}（{zh.get(self.mode, self.mode)}）：权重 {self.weights_gb:.2f} GB，"
            f"估算需求 {self.required_gb:.2f} GB，可用显存 {self.free_gb:.2f} GB"
        )


def plan_placement(
    pipe,
    resolution: int,
    override: str = "auto",
    safety: float = 0.92,
    verbose: bool = True,
) -> PlacementPlan:
    """决定把权重放在显存还是放 CPU 再按需搬运。

    估算方式：权重字节 × 1.15（临时缓冲）+ 激活预留。512 分辨率下激活开销很小，
    1024 则明显上升，因此按分辨率给不同预留。
    """
    import torch

    weights = pipeline_weight_bytes(pipe)
    weights_gb = weights / 1024**3
    component_name, component_bytes = largest_component_bytes(pipe)
    component_gb = component_bytes / 1024**3
    activation_gb = 1.2 if resolution <= 640 else 2.5
    required_gb = weights_gb * 1.15 + activation_gb
    # offload 模式下同一时刻只需一个组件驻留显存（通常就是 UNet）
    offload_need_gb = component_gb * 1.05 + activation_gb

    free_bytes, _total = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3

    if override == "gpu":
        plan = PlacementPlan("gpu", weights_gb, free_gb, required_gb, "用户指定 --placement gpu")
    elif override == "offload":
        plan = PlacementPlan("model_offload", weights_gb, free_gb, required_gb, "用户指定 --placement offload")
    elif required_gb <= free_gb * safety:
        plan = PlacementPlan("gpu", weights_gb, free_gb, required_gb, "权重可全量常驻")
    elif offload_need_gb <= free_gb * safety:
        plan = PlacementPlan(
            "model_offload",
            weights_gb,
            free_gb,
            required_gb,
            f"权重总量放不下，但最大组件 {component_name}（{component_gb:.2f} GB）可以，改用模型级 offload",
        )
    else:
        plan = PlacementPlan(
            "sequential_offload",
            weights_gb,
            free_gb,
            required_gb,
            f"连最大组件 {component_name}（{component_gb:.2f} GB）都放不下，退回逐层 offload（会明显变慢）",
        )

    if verbose:
        print(f"[vram] {plan.describe()}")
        if plan.reason:
            print(f"[vram] 依据：{plan.reason}")
    return plan


def apply_placement(pipe, plan: PlacementPlan, verbose: bool = True) -> None:
    """按放置计划把管线搬到设备上。"""
    if plan.mode == "gpu":
        pipe.to("cuda")
    elif plan.mode == "model_offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_sequential_cpu_offload()
    if verbose:
        print(f"[vram] 已应用：{plan.mode}")


# --------------------------------------------------------------------------------------
# 计时 / 显存观测
# --------------------------------------------------------------------------------------
class StageTimer:
    """分阶段计时器（自动 CUDA 同步，保证数字真实）。"""

    def __init__(self, sync: bool = True):
        self.sync = sync
        self.stages: dict[str, float] = {}
        self._marks: list[tuple[str, float]] = []

    def _now(self) -> float:
        if self.sync:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return time.perf_counter()

    @contextlib.contextmanager
    def stage(self, name: str):
        t0 = self._now()
        try:
            yield
        finally:
            dt = self._now() - t0
            self.stages[name] = self.stages.get(name, 0.0) + dt
            self._marks.append((name, dt))

    def get(self, name: str, default: float = 0.0) -> float:
        return self.stages.get(name, default)

    @property
    def total(self) -> float:
        return sum(self.stages.values())

    def table(self) -> str:
        rows = [f"  {k:<22}{v:7.3f} s" for k, v in self.stages.items()]
        return "\n".join(rows)


@dataclass
class VramSample:
    """一次推理的显存观测结果。"""

    peak_allocated_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    free_before_gb: float = 0.0
    total_gb: float = 0.0

    def describe(self) -> str:
        return (
            f"峰值已分配 {self.peak_allocated_gb:.2f} GB / 峰值保留 {self.peak_reserved_gb:.2f} GB "
            f"（设备总量 {self.total_gb:.1f} GB，推理前空闲 {self.free_before_gb:.2f} GB）"
        )


class VramMonitor:
    """峰值显存观测：``with VramMonitor() as m: ...`` 然后读 ``m.sample``。"""

    def __enter__(self) -> "VramMonitor":
        import torch

        self.sample = VramSample()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            self.sample.free_before_gb = free / 1024**3
            self.sample.total_gb = total / 1024**3
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> None:
        import torch

        if torch.cuda.is_available():
            self.sample.peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
            self.sample.peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

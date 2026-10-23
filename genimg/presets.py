"""挡位与模型注册表：模型档案、精度挡位、画质挡位，以及硬件能力判定。

设计原则
--------
* **精度挡位（precision gear）** 控制权重的存储 dtype 与前向计算精度，是"速度快慢 / 数值保真"的开关。
* **画质挡位（quality gear）** 控制采样步数、CFG 强度与调度器，是"画得快 / 画得细"的开关。
* **模型档案（model profile）** 描述底模本身，并声明它与哪些挡位兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------------------
# 精度挡位
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PrecisionGear:
    """一个精度挡位的完整定义。"""

    key: str
    dtype: str  # torch dtype 名称
    tf32: bool
    label: str
    note: str
    vae_dtype: str | None = None  # VAE 单独精度；None 表示与主精度一致
    min_cc: tuple[int, int] | None = None  # 硬件能力门槛 (major, minor)
    backend: str | None = None  # 需要额外推理后端（如 torchao）时的标记


PRECISION_GEARS: dict[str, PrecisionGear] = {
    "fp32": PrecisionGear(
        key="fp32",
        dtype="float32",
        tf32=False,
        label="fp32（全精度，画质基线）",
        note="权重与计算均为 fp32，关闭 TF32。数值最保真、显存占用最大、速度最慢，用作对照基线。",
    ),
    "tf32": PrecisionGear(
        key="tf32",
        dtype="float32",
        tf32=True,
        label="tf32（fp32 权重 + 张量核）",
        note="权重仍是 fp32，矩阵乘/卷积走 TF32 张量核（10 位尾数）。相对 fp32 通常有 2~3 倍提速，画质差异肉眼难辨。",
    ),
    "bf16": PrecisionGear(
        key="bf16",
        dtype="bfloat16",
        tf32=False,
        label="bf16（半精度，数值最稳）",
        note="8 位指数 + 7 位尾数，动态范围与 fp32 相同，几乎不会溢出/出黑图；Ampere 上速度与 fp16 接近。",
    ),
    "fp16": PrecisionGear(
        key="fp16",
        dtype="float16",
        tf32=False,
        label="fp16（半精度，默认最快）",
        note="10 位尾数 + 5 位指数，张量核吞吐最高、显存占用最小，是官方推荐的推理精度。VAE 解码单独用 fp32，避免半精度解码偶发的黑图。",
        vae_dtype="float32",
    ),
    "fp8": PrecisionGear(
        key="fp8",
        dtype="float8_e4m3fn",
        tf32=False,
        label="fp8（需 Ada/Hopper 硬件）",
        note="fp8 张量核需要计算能力 >= 8.9（Ada / Hopper）。Ampere（RTX 30 系，sm_86）没有 fp8 单元，本挡位在本机会被明确拒绝并给出原因。",
        min_cc=(8, 9),
        backend="torchao",
    ),
}


# --------------------------------------------------------------------------------------
# 画质挡位
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityGear:
    """一个画质挡位的完整定义。"""

    key: str
    steps: int
    guidance: float
    scheduler: str  # euler | euler_trailing | dpmpp_2m | default
    label: str
    note: str
    lora: str | None = None  # 需要挂载的加速 LoRA 档位键
    only_family: tuple[str, ...] = ()  # 仅对特定模型家族有效；空 = 全部


QUALITY_GEARS: dict[str, QualityGear] = {
    "turbo": QualityGear(
        key="turbo",
        steps=2,
        guidance=0.0,
        scheduler="euler_trailing",
        label="turbo（极速：2 步）",
        note="蒸馏模型的少步采样，2 步出图。guidance=0 表示不启用 CFG，因此负向提示词不起作用。",
        only_family=("sd-turbo",),
    ),
    "lightning": QualityGear(
        key="lightning",
        steps=4,
        guidance=1.0,
        scheduler="euler_trailing",
        label="lightning（快：4 步 + Lightning LoRA）",
        note="挂载 SDXL-Lightning 4 步 LoRA。guidance=1 即关闭 CFG，负向提示词不起作用，但速度约为同画质的 1/5 耗时。",
        lora="lightning-4step",
        only_family=("sdxl",),
    ),
    "lightning8": QualityGear(
        key="lightning8",
        steps=8,
        guidance=1.5,
        scheduler="euler_trailing",
        label="lightning8（较快：8 步 + Lightning LoRA）",
        note="同 Lightning LoRA 的 8 步档，细节比 4 步档更完整，仍远快于常规采样。",
        lora="lightning-8step",
        only_family=("sdxl",),
    ),
    "draft": QualityGear(
        key="draft",
        steps=12,
        guidance=5.0,
        scheduler="euler",
        label="draft（草稿：12 步）",
        note="少量步数快速预览构图，细节偏软，适合试提示词。",
    ),
    "balanced": QualityGear(
        key="balanced",
        steps=25,
        guidance=7.0,
        scheduler="euler",
        label="balanced（均衡：25 步，默认）",
        note="常规采样的画质/耗时平衡点，SDXL 官方推荐区间的下沿。",
    ),
    "quality": QualityGear(
        key="quality",
        steps=35,
        guidance=7.5,
        scheduler="euler",
        label="quality（高画质：35 步）",
        note="步数更足，纹理与光影更完整，耗时为 balanced 的约 1.4 倍。",
    ),
    "max": QualityGear(
        key="max",
        steps=50,
        guidance=8.0,
        scheduler="euler",
        label="max（最高：50 步）",
        note="把步数拉满，边际收益递减但适合作为质量上限参考。",
    ),
}


# --------------------------------------------------------------------------------------
# 模型档案
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelProfile:
    """一个可用底模的档案。"""

    key: str
    repo: str
    family: str
    label: str
    native_size: int  # 原生训练分辨率
    params_billion: float  # 参与前向的主要参数量（UNet + 文本编码器）
    supports_negative: bool
    default_gear: str
    note: str
    variant: str | None = None  # 默认下载的仓库变体（fp16 / None=fp32）
    aliases: tuple[str, ...] = ()
    # 已被实测否决的精度组合（单张 > 60 秒，性价比为负），不再提供
    blocked_precisions: tuple[str, ...] = ()
    blocked_reason: str = ""
    # 已被实测否决的画质挡位（如 sd-turbo 在 25 步下会崩），不再提供
    blocked_gears: tuple[str, ...] = ()
    blocked_gears_reason: str = ""
    extra: dict = field(default_factory=dict)


MODEL_PROFILES: dict[str, ModelProfile] = {
    "sdxl": ModelProfile(
        key="sdxl",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        family="sdxl",
        label="SDXL 1.0 base（画质默认）",
        native_size=1024,
        params_billion=3.4,
        supports_negative=True,
        default_gear="balanced",
        note="SDXL 是 12GB 显存上的画质/速度折中点，原生 1024。512 出图完全可行，但原生分辨率下细节更足。",
        variant="fp16",
        blocked_precisions=("fp32", "tf32"),
        blocked_reason=(
            "SDXL 在 fp32/tf32 下显存不够，被迫顺序卸载（sequential_offload）："
            "512 测 balanced 单张 58 秒、quality/max 更久，已超过 1 分钟预算，"
            "而画质与 fp16/bf16 无明显差异，因此不再提供该组合。"
        ),
        aliases=("xl", "sdxl-base"),
    ),
    "sd15": ModelProfile(
        key="sd15",
        repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
        family="sd15",
        label="Stable Diffusion 1.5（轻量/原生 512）",
        native_size=512,
        params_billion=0.98,
        supports_negative=True,
        default_gear="balanced",
        note="原生训练分辨率就是 512×512，显存占用与耗时最低，图生图最稳；细节与语义理解弱于 SDXL。",
        variant=None,
        aliases=("sd1.5", "1.5"),
    ),
    "sd-turbo": ModelProfile(
        key="sd-turbo",
        repo="stabilityai/sd-turbo",
        family="sd-turbo",
        label="SD-Turbo（极速，原生 512）",
        native_size=512,
        params_billion=1.2,
        supports_negative=False,
        default_gear="turbo",
        note="对抗蒸馏的 512 原生模型，1~4 步即可出图，是 512 分辨率下最快的可用档位；代价是不支持 CFG/负向提示词。",
        variant="fp16",
        aliases=("turbo", "sdturbo"),
        blocked_gears=("draft", "balanced", "quality", "max"),
        blocked_gears_reason=(
            "sd-turbo 是对抗蒸馏模型，只在 1–4 步（turbo 档）内有效。"
            "实测 25 步 + CFG 7 会把画面过曝成高饱和噪点（512 下已对照确认），"
            "因此只保留 turbo 档。"
        ),
    ),
}

# 别名 -> 规范键
MODEL_ALIASES: dict[str, str] = {}
for _k, _p in MODEL_PROFILES.items():
    MODEL_ALIASES[_k.lower()] = _k
    for _a in _p.aliases:
        MODEL_ALIASES[_a.lower()] = _k

PRECISION_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "tf32": "tf32",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp8": "fp8",
    "float8": "fp8",
}

QUALITY_ALIASES = {
    "turbo": "turbo",
    "lightning": "lightning",
    "lightning4": "lightning",
    "lightning8": "lightning8",
    "draft": "draft",
    "balanced": "balanced",
    "fast": "draft",
    "quality": "quality",
    "high": "quality",
    "max": "max",
    "best": "max",
}


# --------------------------------------------------------------------------------------
# 查询辅助
# --------------------------------------------------------------------------------------
def resolve_model(name: str) -> ModelProfile:
    key = MODEL_ALIASES.get(name.strip().lower())
    if key is None:
        raise KeyError(f"未知模型 {name!r}；可选：{', '.join(sorted(MODEL_PROFILES))}")
    return MODEL_PROFILES[key]


def resolve_precision(name: str) -> PrecisionGear:
    key = PRECISION_ALIASES.get(name.strip().lower())
    if key is None:
        raise KeyError(f"未知精度挡位 {name!r}；可选：{', '.join(sorted(PRECISION_GEARS))}")
    return PRECISION_GEARS[key]


def resolve_quality(name: str, profile: ModelProfile) -> QualityGear:
    key = QUALITY_ALIASES.get(name.strip().lower())
    if key is None:
        raise KeyError(f"未知画质挡位 {name!r}；可选：{', '.join(sorted(QUALITY_GEARS))}")
    gear = QUALITY_GEARS[key]
    if gear.only_family and profile.family not in gear.only_family:
        usable = [k for k, g in QUALITY_GEARS.items() if not g.only_family or profile.family in g.only_family]
        raise ValueError(f"画质挡位 {gear.key!r} 不适用于模型 {profile.key!r}；该模型可用：{', '.join(usable)}")
    if gear.key in getattr(profile, "blocked_gears", ()):
        reason = getattr(profile, "blocked_gears_reason", "") or "实测效果不达标"
        raise ValueError(f"画质挡位 {gear.key!r} 在模型 {profile.key!r} 上已禁用：{reason}")
    return gear


def quality_gears_for(profile: ModelProfile) -> list[str]:
    blocked = set(getattr(profile, "blocked_gears", ()))
    return [
        k
        for k, g in QUALITY_GEARS.items()
        if (not g.only_family or profile.family in g.only_family) and k not in blocked
    ]


# LoRA 档位：仓库 + 文件名（均为官方发布的 394MB 级小文件）
LORA_REGISTRY: dict[str, dict] = {
    "lightning-4step": {
        "repo": "ByteDance/SDXL-Lightning",
        "weight_name": "sdxl_lightning_4step_lora.safetensors",
        "family": "sdxl",
    },
    "lightning-8step": {
        "repo": "ByteDance/SDXL-Lightning",
        "weight_name": "sdxl_lightning_8step_lora.safetensors",
        "family": "sdxl",
    },
}


# --------------------------------------------------------------------------------------
# 硬件能力
# --------------------------------------------------------------------------------------
@dataclass
class HardwareInfo:
    """本机 GPU 能力快照。"""

    available: bool
    name: str = "无 CUDA 设备"
    cc: tuple[int, int] = (0, 0)
    total_gb: float = 0.0
    driver: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    supports_tf32: bool = False
    supports_bf16: bool = False
    supports_fp16_tc: bool = False
    supports_fp8: bool = False

    def summary(self) -> str:
        if not self.available:
            return "未检测到可用 CUDA 设备，将只能用 CPU 推理（会很慢）"
        feats = [
            ("TF32", self.supports_tf32),
            ("bf16", self.supports_bf16),
            ("fp16 张量核", self.supports_fp16_tc),
            ("fp8 张量核", self.supports_fp8),
        ]
        got = ", ".join(f"{n}={'有' if v else '无'}" for n, v in feats)
        return (
            f"{self.name} | 计算能力 sm_{self.cc[0]}{self.cc[1]} | 显存 {self.total_gb:.1f} GB | "
            f"驱动 {self.driver} | torch {self.torch_version} + CUDA {self.cuda_version} | {got}"
        )


def probe_hardware() -> HardwareInfo:
    """探测本机 CUDA 能力（不加载任何模型）。"""
    import torch

    info = HardwareInfo(available=False, torch_version=torch.__version__, cuda_version=torch.version.cuda or "")
    if not torch.cuda.is_available():
        return info
    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    info.available = True
    info.name = props.name
    info.cc = (major, minor)
    info.total_gb = props.total_memory / 1024**3
    info.supports_tf32 = major >= 8  # Ampere 起支持 TF32
    info.supports_bf16 = major >= 8
    info.supports_fp16_tc = major >= 7
    info.supports_fp8 = major >= 9  # Ada / Hopper 才有 fp8 张量核
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info.driver = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        info.driver = ""
    return info


def check_precision_support(gear: PrecisionGear, hw: HardwareInfo, profile=None) -> str | None:
    """返回不支持的原因；支持则返回 None。

    两个层面：硬件能力（fp8/bf16）与**模型实测否决**（如 SDXL 的 fp32/tf32 单张 > 60 秒）。
    """
    if not hw.available:
        return "本机没有可用的 CUDA 设备"
    if gear.min_cc and hw.cc < gear.min_cc:
        need = f"sm_{gear.min_cc[0]}{gear.min_cc[1]}"
        return (
            f"{gear.key} 需要计算能力 >= {need}（Ada / Hopper 及以上），"
            f"本机 GPU 为 sm_{hw.cc[0]}{hw.cc[1]}（Ampere），硬件层面没有 fp8 张量核心，无法通过软件绕过。"
        )
    if gear.key == "bf16" and not hw.supports_bf16:
        return f"{gear.key} 需要 Ampere 及以上算力，本机为 sm_{hw.cc[0]}{hw.cc[1]}"
    if profile is not None and gear.key in getattr(profile, "blocked_precisions", ()):
        return getattr(profile, "blocked_reason", "") or f"{profile.key} 不提供 {gear.key} 精度"
    return None

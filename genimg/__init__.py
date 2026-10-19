"""genimg —— 本机生图 agent（文生图 / 图生图 + 精度挡位 + 画质挡位）。"""

from .presets import (
    MODEL_PROFILES,
    PRECISION_GEARS,
    QUALITY_GEARS,
    probe_hardware,
    resolve_model,
    resolve_precision,
    resolve_quality,
)

__all__ = [
    "MODEL_PROFILES",
    "PRECISION_GEARS",
    "QUALITY_GEARS",
    "probe_hardware",
    "resolve_model",
    "resolve_precision",
    "resolve_quality",
]

__version__ = "0.2.0"

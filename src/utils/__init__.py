"""工具模块：配置、种子、设备管理、数学运算."""

from src.utils.config import load_config, merge_configs, validate_config
from src.utils.seed import set_global_seed, get_rng
from src.utils.device import DeviceManager
from src.utils.math_ops import (
    batch_cosine_similarity,
    pairwise_l2_distance,
    safe_normalize,
)

__all__ = [
    "load_config",
    "merge_configs",
    "validate_config",
    "set_global_seed",
    "get_rng",
    "DeviceManager",
    "batch_cosine_similarity",
    "pairwise_l2_distance",
    "safe_normalize",
]
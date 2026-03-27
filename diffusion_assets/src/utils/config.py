"""
配置管理模块
============
基于 OmegaConf 实现分层配置加载、合并、验证。
支持 CLI 覆盖与环境变量插值。

典型用法:
    cfg = load_config("configs/default.yaml", overrides=["data.target_dim=16384"])
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


# ------------------------------------------------------------------ #
#  自定义 OmegaConf 解析器（注册一次）
# ------------------------------------------------------------------ #

def _register_custom_resolvers() -> None:
    """注册自定义 OmegaConf 解析器，保证幂等。"""

    if not OmegaConf.has_resolver("now"):
        import datetime

        OmegaConf.register_new_resolver(
            "now",
            lambda fmt: datetime.datetime.now().strftime(fmt),
        )

    if not OmegaConf.has_resolver("sqrt_int"):
        OmegaConf.register_new_resolver(
            "sqrt_int",
            lambda x: int(math.isqrt(int(x))),
        )

    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env",
            lambda key, default="": os.environ.get(key, default),
        )


_register_custom_resolvers()


# ------------------------------------------------------------------ #
#  公开 API
# ------------------------------------------------------------------ #

def load_config(
    path: Union[str, Path],
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """
    加载单个 YAML 配置文件，并应用 CLI 覆盖。

    Parameters
    ----------
    path : str | Path
        YAML 文件路径。
    overrides : list[str], optional
        点号分隔的覆盖项，例如 ["data.target_dim=16384", "training.batch_size=32"]。

    Returns
    -------
    DictConfig
        解析后的不可变配置对象（调用 freeze 后）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path.resolve()}")

    cfg = OmegaConf.load(path)

    # 应用 CLI 覆盖
    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def merge_configs(
    base: Union[str, Path, DictConfig],
    *overlays: Union[str, Path, DictConfig],
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """
    分层合并多个配置：base ← overlay_1 ← overlay_2 ← ... ← CLI overrides。

    Parameters
    ----------
    base : path | DictConfig
        基础配置。
    *overlays : path | DictConfig
        覆盖层配置，按顺序优先级递增。
    overrides : list[str], optional
        最高优先级的 CLI 覆盖。

    Returns
    -------
    DictConfig
    """

    def _ensure_cfg(x: Union[str, Path, DictConfig]) -> DictConfig:
        if isinstance(x, (str, Path)):
            return load_config(x)
        return x

    merged = _ensure_cfg(base)
    for overlay in overlays:
        merged = OmegaConf.merge(merged, _ensure_cfg(overlay))

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))

    return merged


class ConfigValidationError(ValueError):
    """配置验证失败时抛出。"""
    pass


def validate_config(cfg: DictConfig) -> DictConfig:
    """
    验证配置的逻辑一致性，返回补全后的配置。

    校验项
    ------
    1. target_dim 必须是完全平方数。
    2. target_resolution = sqrt(target_dim)，自动补全。
    3. grayscale_size^2 <= target_dim。
    4. scheduler.type 必须在 {vp, ve, of} 中。
    5. output_dir 自动创建。

    Raises
    ------
    ConfigValidationError
    """
    errors: List[str] = []

    # ---- Circle 数据跳过图像维度校验 ----
    is_circle = str(cfg.data.get("parquet_path", "")).strip() == "__circle__"

    if not is_circle:
        # --- 数据维度校验（原有逻辑）---
        d = cfg.data.target_dim
        sqrt_d = math.isqrt(d)
        if sqrt_d * sqrt_d != d:
            errors.append(
                f"data.target_dim={d} 不是完全平方数。"
            )
        else:
            cfg = OmegaConf.merge(
                cfg, OmegaConf.create({"data": {"target_resolution": sqrt_d}}),
            )

        m = cfg.data.grayscale_size ** 2
        if m > d:
            errors.append(
                f"有效维度 m={m} 大于环境维度 d={d}。"
            )

    # --- 调度器校验 ---
    valid_schedulers = {"vp", "ve", "of"}
    if cfg.scheduler.type not in valid_schedulers:
        errors.append(
            f"scheduler.type='{cfg.scheduler.type}' 无效，"
            f"必须是 {valid_schedulers} 之一。"
        )

    # --- 模型输出模式与调度器匹配建议（警告，非错误）---
    output_type = cfg.model.output_type
    sched_type = cfg.scheduler.type
    recommended = {"vp": "noise", "ve": "score", "of": "velocity"}
    if output_type != recommended.get(sched_type, output_type):
        import warnings
        warnings.warn(
            f"scheduler={sched_type} 通常搭配 output_type='{recommended[sched_type]}', "
            f"当前设置为 '{output_type}'。如有意为之请忽略。",
            UserWarning,
            stacklevel=2,
        )

    # --- 采样步数校验 ---
    if cfg.sampling.num_sampling_steps > cfg.scheduler.num_diffusion_steps:
        errors.append(
                        f"sampling.num_sampling_steps ({cfg.sampling.num_sampling_steps}) "
            f"不应超过 scheduler.num_diffusion_steps ({cfg.scheduler.num_diffusion_steps})。"
        )

    # --- 轨迹捕获时间步校验 ---
    for t_val in cfg.sampling.trajectory_capture_timesteps:
        if not (0.0 <= t_val <= 1.0):
            errors.append(
                f"trajectory_capture_timesteps 中的值 {t_val} 不在 [0, 1] 范围内。"
            )

    # --- 混合精度校验 ---
    valid_mp = {"fp16", "bf16", "none"}
    if cfg.device.mixed_precision not in valid_mp:
        errors.append(
            f"device.mixed_precision='{cfg.device.mixed_precision}' 无效，"
            f"必须是 {valid_mp} 之一。"
        )

    # --- 汇总错误 ---
    if errors:
        msg = "配置验证失败:\n" + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        raise ConfigValidationError(msg)

    # --- 自动创建输出目录 ---
    output_dir = Path(OmegaConf.to_container(cfg, resolve=True)["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    return cfg


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """
    将 DictConfig 转换为普通 Python dict（解析所有插值）。
    
    用于序列化或传给第三方库（如 wandb.init(config=...)）。

    Parameters
    ----------
    cfg : DictConfig

    Returns
    -------
    dict
    """
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def pretty_print_config(cfg: DictConfig) -> str:
    """返回格式化的 YAML 字符串，用于日志输出。"""
    return OmegaConf.to_yaml(cfg, resolve=True)


def save_config(cfg: DictConfig, path: Union[str, Path]) -> None:
    """将配置保存为 YAML 文件（解析所有插值后保存）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
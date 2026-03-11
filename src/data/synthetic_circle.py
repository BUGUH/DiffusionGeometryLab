"""
Circle 数据适配 — D 维背景空间中的二维圆环 (S¹)
=================================================
内蕴维度 m=1, 背景维度 D 可配置。
数据点满足 x_1² + x_2² = r², 其余 D-2 维为 0 + 噪声。
"""

from __future__ import annotations
import logging
import math
from typing import Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


def generate_circle_data(
    N: int,
    D: int,
    r: float = 1.0,
    noise: float = 0.01,
    seed: int = 42,
) -> torch.Tensor:
    """
    生成 D 维背景空间中的二维圆环数据。

    Parameters
    ----------
    N : int       样本数量
    D : int       背景空间维度（必须 >= 2）
    r : float     圆环半径
    noise : float 高斯噪声标准差
    seed : int

    Returns
    -------
    torch.Tensor, shape (N, D)
    """
    assert D >= 2, f"背景维度 D={D} 必须 >= 2"

    rng = torch.Generator().manual_seed(seed)
    theta = torch.rand(N, generator=rng) * 2 * math.pi

    data = torch.zeros(N, D)
    data[:, 0] = r * theta.cos()
    data[:, 1] = r * theta.sin()

    # 所有维度加噪声
    data += torch.randn(N, D, generator=rng) * noise

    logger.info(
        f"Circle 数据生成: N={N}, D={D}, r={r}, noise={noise}, "
        f"range=[{data.min():.3f}, {data.max():.3f}]"
    )
    return data


def circle_data_to_model_input(
    data: torch.Tensor,
) -> torch.Tensor:
    """
    将 (N, D) 向量转为模型输入格式 (N, 1, D)。
    MLP 模型接收 (B, 1, D) 展平为 (B, D)；
    保留通道维度是为了与 UNet 管线兼容。
    """
    return data.unsqueeze(1)  # (N, 1, D)


def model_output_to_vectors(
    output: torch.Tensor,
) -> torch.Tensor:
    """将模型输出 (B, 1, D) 还原为 (B, D)。"""
    if output.dim() == 3:
        return output.squeeze(1)
    return output


def compute_circle_quality_metrics(
    generated: torch.Tensor,
    r: float = 1.0,
) -> dict:
    """
    Circle 数据适配的质量指标。

    Returns
    -------
    dict:
        radius_mean, radius_std: 生成点到原点距离的统计
        on_manifold_ratio: 点在圆环附近 (|r_i - r| < 0.1) 的比例
        residual_norm: 非流形维度 (dim 2:) 的范数均值
    """
    pts = generated.view(len(generated), -1)
    D = pts.shape[1]

    radii = torch.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    residual = pts[:, 2:].norm(dim=1) if D > 2 else torch.zeros(len(pts))

    return {
        "radius_mean": radii.mean().item(),
        "radius_std": radii.std().item(),
        "on_manifold_ratio": (torch.abs(radii - r) < 0.1).float().mean().item(),
        "residual_norm_mean": residual.mean().item(),
        "residual_norm_std": residual.std().item(),
    }


def build_circle_data(cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据配置构建圆环数据。

    Returns
    -------
    (model_input, raw_points): model_input=(N,1,D), raw_points=(N,D)
    """
    c = cfg.data.circle
    raw = generate_circle_data(
        N=c.num_samples,
        D=c.ambient_dim,
        r=c.get("radius", 1.0),
        noise=c.get("noise_std", 0.01),
        seed=cfg.experiment.get("seed", 42),
    )
    model_input = circle_data_to_model_input(raw)  # (N, 1, D)
    return model_input, raw
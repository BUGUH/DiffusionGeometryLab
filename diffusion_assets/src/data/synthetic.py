"""
合成数据生成器
==============
生成各种几何形状的合成数据，用于管线测试和理论验证。
"""

from __future__ import annotations
import math
import logging
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_circle_data(
    num_samples: int = 2000,
    noise_std: float = 0.05,
    seed: int = 42,
) -> torch.Tensor:
    """
    在单位圆上均匀采样 2D 点，加微小噪声。

    Returns
    -------
    torch.Tensor, shape (N, 2), 值域约 [-1.1, 1.1]
    """
    rng = torch.Generator().manual_seed(seed)
    theta = torch.rand(num_samples, generator=rng) * 2 * math.pi
    x = torch.stack([theta.cos(), theta.sin()], dim=1)
    x += torch.randn(num_samples, 2, generator=rng) * noise_std
    logger.info(f"生成圆环数据: N={num_samples}, noise_std={noise_std}")
    return x


def generate_swiss_roll(
    num_samples: int = 2000,
    noise_std: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """Swiss Roll 2D 投影。"""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(num_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1).astype(np.float32)
    # 归一化到 [-1, 1]
    data = data / np.abs(data).max()
    data += rng.normal(0, noise_std, data.shape).astype(np.float32)
    return torch.from_numpy(data)


def generate_two_moons(
    num_samples: int = 2000,
    noise_std: float = 0.05,
    seed: int = 42,
) -> torch.Tensor:
    """两个半月形。"""
    rng = np.random.default_rng(seed)
    n = num_samples // 2
    theta1 = np.linspace(0, np.pi, n)
    theta2 = np.linspace(0, np.pi, num_samples - n)

    x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    x2 = np.stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5], axis=1)
    data = np.vstack([x1, x2]).astype(np.float32)
    data = data / np.abs(data).max()
    data += rng.normal(0, noise_std, data.shape).astype(np.float32)
    return torch.from_numpy(data)


def points_to_images(
    points: torch.Tensor,
    image_size: int = 8,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """
    将 2D 点转换为小灰度图。

    策略：将 2D 坐标编码为图像——以点坐标为中心画高斯斑点。
    这样保留了流形结构，同时获得图像格式用于 UNet。

    Parameters
    ----------
    points : (N, 2), 值域约 [-1, 1]
    image_size : int
    normalize_range : tuple

    Returns
    -------
    torch.Tensor, shape (N, 1, image_size, image_size)
    """
    N = len(points)
    lo, hi = normalize_range

    # 创建坐标网格
    coords = torch.linspace(-1.5, 1.5, image_size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")  # (H, W)

    images = torch.zeros(N, 1, image_size, image_size)

    for i in range(N):
        px, py = points[i, 0].item(), points[i, 1].item()
        # 高斯斑点
        sigma = 0.3
        gaussian = torch.exp(-((grid_x - px) ** 2 + (grid_y - py) ** 2) / (2 * sigma ** 2))
        # 归一化到 [0, 1] 再映射
        gaussian = gaussian / (gaussian.max() + 1e-8)
        images[i, 0] = gaussian * (hi - lo) + lo

    logger.info(f"2D 点 → 图像: {points.shape} → {images.shape}, "
                f"range=[{images.min():.2f}, {images.max():.2f}]")
    return images


def build_synthetic_data(cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据配置生成合成数据。

    Returns
    -------
    (images, raw_points): images 用于训练，raw_points 用于可视化
    """
    syn_cfg = cfg.data.get("synthetic", {})
    syn_type = syn_cfg.get("type", "circle")
    num_samples = syn_cfg.get("num_samples", 2000)
    noise_std = syn_cfg.get("noise_std", 0.05)
    seed = cfg.experiment.get("seed", 42)

    if syn_type == "circle":
        points = generate_circle_data(num_samples, noise_std, seed)
    elif syn_type == "swiss_roll":
        points = generate_swiss_roll(num_samples, noise_std, seed)
    elif syn_type == "two_moons":
        points = generate_two_moons(num_samples, noise_std, seed)
    else:
        raise ValueError(f"未知合成数据类型: {syn_type}")

    images = points_to_images(
        points,
        image_size=cfg.data.grayscale_size,
        normalize_range=tuple(cfg.data.normalize_range),
    )

    return images, points
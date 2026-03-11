"""FID/MMD 评估器 — 适配图像和向量数据。"""

from __future__ import annotations
import logging
from typing import Dict, Optional

import numpy as np
import torch
from src.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


class FIDEvaluator:
    """
    分布距离评估器。
    图像数据：简化 FID（像素空间）。
    向量数据：MMD（最大均值差异）+ 简化 FID。
    """

    def __init__(self, feature_dim: int = 2048, device: torch.device = None):
        self.feature_dim = feature_dim
        self.device = device or torch.device("cpu")

    def compute_fid(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        """统一接口：自动选 FID 或 MMD。"""
        return self._simple_fid(real, fake)

    def compute_mmd(
        self, real: torch.Tensor, fake: torch.Tensor,
        kernel: str = "rbf", bandwidth: float = None,
    ) -> float:
        """
        MMD (Maximum Mean Discrepancy) — Circle 数据适配。

        MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        """
        x = real.reshape(len(real), -1).float()
        y = fake.reshape(len(fake), -1).float()

        if bandwidth is None:
            # 中值启发式
            with torch.no_grad():
                dists = torch.cdist(x[:500], y[:500])
                bandwidth = dists.median().item()
                bandwidth = max(bandwidth, 1e-4)

        if kernel == "rbf":
            kxx = self._rbf_kernel(x, x, bandwidth)
            kyy = self._rbf_kernel(y, y, bandwidth)
            kxy = self._rbf_kernel(x, y, bandwidth)
        else:
            raise ValueError(f"未知核函数: {kernel}")

        mmd2 = kxx.mean() + kyy.mean() - 2 * kxy.mean()
        return max(mmd2.item(), 0.0)

    @staticmethod
    def _rbf_kernel(x, y, bandwidth):
        dist_sq = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-dist_sq / (2 * bandwidth ** 2))

    def compute_trajectory_fid(
        self, real_data, trajectory_bundle, scheduler,
        timesteps=None, use_mmd=False,
    ) -> Dict[float, float]:
        """轨迹中间态 FID/MMD。"""
        if timesteps is None:
            timesteps = sorted(trajectory_bundle.x_ts.keys())

        results = {}
        N = min(len(real_data), 2048)

        for t_val in timesteps:
            if t_val not in trajectory_bundle.x_ts:
                continue

            fake_t = trajectory_bundle.x_ts[t_val]
            t_tensor = torch.full((N,), t_val)
            real_t, _ = scheduler.add_noise(real_data[:N], t_tensor)

            n = min(len(real_t), len(fake_t))
            if use_mmd:
                score = self.compute_mmd(real_t[:n], fake_t[:n])
            else:
                score = self._simple_fid(real_t[:n], fake_t[:n])
            results[t_val] = score

        return results

    def _simple_fid(self, real, fake):
        real_flat = real.reshape(len(real), -1).float().cpu()
        fake_flat = fake.reshape(len(fake), -1).float().cpu()

        max_dim = min(256, real_flat.shape[1], real_flat.shape[0] - 1, fake_flat.shape[0] - 1)
        if max_dim < 2:
            return ((real_flat.mean(0) - fake_flat.mean(0)) ** 2).sum().item()

        if real_flat.shape[1] > max_dim:
            real_flat, fake_flat = self._pca_reduce(real_flat, fake_flat, max_dim)

        mu_r, mu_f = real_flat.mean(0), fake_flat.mean(0)
        sigma_r, sigma_f = self._cov(real_flat), self._cov(fake_flat)

        mean_term = ((mu_r - mu_f) ** 2).sum().item()
        trace_r = sigma_r.diagonal().sum().item()
        trace_f = sigma_f.diagonal().sum().item()

        try:
            product = sigma_r @ sigma_f
            product = (product + product.T) / 2
            eigvals = torch.linalg.eigvalsh(product).clamp(min=0)
            trace_sqrt = eigvals.sqrt().sum().item()
        except Exception:
            trace_sqrt = min(trace_r, trace_f)

        return max(mean_term + trace_r + trace_f - 2 * trace_sqrt, 0.0)

    @staticmethod
    def _pca_reduce(a, b, dim):
        combined = torch.cat([a, b], dim=0)
        mean = combined.mean(dim=0, keepdim=True)
        centered = combined - mean
        try:
            _, _, V = torch.pca_lowrank(centered, q=dim)
            return (a - mean) @ V, (b - mean) @ V
        except Exception:
            proj = torch.randn(a.shape[1], dim) / (dim ** 0.5)
            return a @ proj, b @ proj

    @staticmethod
    def _cov(x):
        N = x.shape[0]
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        cov = (centered.T @ centered) / max(N - 1, 1)
        cov += torch.eye(cov.shape[0]) * 1e-6
        return cov
"""
动态记忆比例 f_mem(t) 计算
==========================
对当前 x_t，计算其在训练集加噪版本中的最近邻/次近邻距离比 R_t。
统计 R_t < threshold 的样本比例。
"""

from __future__ import annotations
import logging
from typing import Dict, Optional

import numpy as np
import torch
from src.metrics.nn_search import FaissNNIndex
from src.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


class MemoryRatioEvaluator:
    """
    f_mem(t) 评估器。

    对每个时间步 t:
    1. 对训练集加噪: a_t^μ = add_noise(x_train^μ, t)
    2. 建 Faiss 索引
    3. 对采样中间态 x_t 查询 2-NN
    4. R_t = ||x_t - a_t^μ1|| / ||x_t - a_t^μ2||
    5. f_mem(t) = fraction(R_t < threshold)
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        threshold: float = 1.0 / 3.0,
        num_timesteps: int = 50,
        t_min: float = 0.001,
        t_max: float = 0.999,
        nprobe: int = 32,
    ):
        self.scheduler = scheduler
        self.threshold = threshold
        self.t_values = np.linspace(t_min, t_max, num_timesteps)
        self.nprobe = nprobe

    @torch.no_grad()
    def compute(
        self,
        train_data: torch.Tensor,
        trajectory_bundle,
        device: torch.device = None,
    ) -> Dict[str, np.ndarray]:
        """
        计算 f_mem(t) 曲线。

        Parameters
        ----------
        train_data : (N_train, C, H, W)
        trajectory_bundle : TrajectoryBundle，含各时间步的 x_t

        Returns
        -------
        {"t": array, "f_mem": array, "mean_ratio": array}
        """
        f_mem_values = []
        mean_ratios = []
        actual_ts = []

        N_train = len(train_data)
        dim = train_data.view(N_train, -1).shape[1]

        for t_val in self.t_values:
            # 找最近的可用时间步
            closest_t = self._find_closest_t(t_val, trajectory_bundle)
            if closest_t is None:
                continue

            actual_ts.append(t_val)
            x_t_samples = trajectory_bundle.x_ts[closest_t]  # (B, C, H, W)

            # 对训练集加噪到同一时间步
            t_tensor = torch.full((N_train,), t_val)
            train_noisy, _ = self.scheduler.add_noise(train_data, t_tensor)
            train_flat = train_noisy.view(N_train, -1)

            # 建索引
            index = FaissNNIndex(dim=dim, nprobe=self.nprobe)
            index.build(train_flat)

            # 查询 2-NN
            query_flat = x_t_samples.view(len(x_t_samples), -1)
            dists, _ = index.query(query_flat, k=2)

            # R_t = d_1 / d_2
            d1 = dists[:, 0].sqrt().clamp(min=1e-10)  # faiss 返回 L2²
            d2 = dists[:, 1].sqrt().clamp(min=1e-10)
            R_t = d1 / d2

            f_mem = (R_t < self.threshold).float().mean().item()
            f_mem_values.append(f_mem)
            mean_ratios.append(R_t.mean().item())

        return {
            "t": np.array(actual_ts),
            "f_mem": np.array(f_mem_values),
            "mean_ratio": np.array(mean_ratios),
        }

    @torch.no_grad()
    def compute_at_single_t(
        self,
        train_data: torch.Tensor,
        x_t_samples: torch.Tensor,
        t_val: float,
    ) -> Dict[str, float]:
        """单个时间步的 f_mem 计算。"""
        N_train = len(train_data)
        dim = train_data.view(N_train, -1).shape[1]

        t_tensor = torch.full((N_train,), t_val)
        train_noisy, _ = self.scheduler.add_noise(train_data, t_tensor)
        train_flat = train_noisy.view(N_train, -1)

        index = FaissNNIndex(dim=dim, nprobe=self.nprobe)
        index.build(train_flat)

        query_flat = x_t_samples.view(len(x_t_samples), -1)
        dists, _ = index.query(query_flat, k=2)

        d1 = dists[:, 0].sqrt().clamp(min=1e-10)
        d2 = dists[:, 1].sqrt().clamp(min=1e-10)
        R_t = d1 / d2

        return {
            "f_mem": (R_t < self.threshold).float().mean().item(),
            "mean_ratio": R_t.mean().item(),
        }

    @staticmethod
    def _find_closest_t(t_val, bundle) -> Optional[float]:
        """从 bundle 中找最近的可用时间步。"""
        available = list(bundle.x_ts.keys())
        if not available:
            return None
        closest = min(available, key=lambda x: abs(x - t_val))
        if abs(closest - t_val) > 0.05:
            return None
        return closest
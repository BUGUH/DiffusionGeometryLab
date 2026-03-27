"""噪声调度器抽象基类。"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch


class BaseScheduler(ABC):
    """
    所有调度器的统一接口。

    约定:
    - 连续时间 t ∈ [0, 1]，t=0 为干净数据，t=1 为纯噪声。
    - 离散时间步 step ∈ [0, num_steps-1]，step=0 对应 t≈0，step=num_steps-1 对应 t≈1。
    - 前向过程: x_t = mean_coeff(t) * x_0 + std(t) * ε
    """

    def __init__(self, num_steps: int = 1000):
        self.num_steps = num_steps
        # 离散时间网格 (不含 0，避免除零)
        self.timesteps = torch.linspace(1e-5, 1.0, num_steps)

    # ---- 核心抽象方法 ----

    @abstractmethod
    def marginal_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 q(x_t | x_0) 的均值系数和标准差。
        
        Returns: (mean_coeff, std) 均与 t 同形状
        """
        ...

    @abstractmethod
    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """信噪比 SNR(t) = mean_coeff(t)^2 / std(t)^2"""
        ...

    # ---- 通用方法 ----

    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向加噪: x_t = mean_coeff * x_0 + std * noise

        Parameters
        ----------
        x_0 : (B, ...) 干净数据
        t : (B,) 连续时间 ∈ [0, 1]
        noise : (B, ...) 可选，默认随机采样

        Returns
        -------
        (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coeff, std = self.marginal_params(t)
        mean_coeff = self._broadcast(mean_coeff, x_0)
        std = self._broadcast(std, x_0)
        x_t = mean_coeff * x_0 + std * noise
        return x_t, noise

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """返回 σ(t)，即标准差。"""
        _, std = self.marginal_params(t)
        return std

    def step_to_t(self, step: torch.Tensor) -> torch.Tensor:
        """离散步 → 连续时间。"""
        return self.timesteps.to(step.device)[step.long()]

    def sample_timesteps(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """均匀采样连续时间 t ∈ (0, 1)。"""
        return torch.rand(batch_size, device=device).clamp(1e-5, 1.0 - 1e-5)

    def get_discrete_sigmas(self) -> torch.Tensor:
        """返回所有离散时间步对应的 σ 数组。"""
        return self.get_sigma(self.timesteps)

    def get_score_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """得分缩放因子: 用于 noise → score 转换: score = -noise / σ(t)"""
        return -1.0 / self.get_sigma(t).clamp(min=1e-8)

    @staticmethod
    def _broadcast(v: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """将 (B,) 广播到 (B, 1, 1, ...) 以匹配 target 维度。"""
        while v.dim() < target.dim():
            v = v.unsqueeze(-1)
        return v
"""OF (Optimal Flow / Flow Matching) 调度器。"""

from __future__ import annotations
from typing import Tuple

import torch
from src.schedulers.base_scheduler import BaseScheduler


class OFScheduler(BaseScheduler):
    """
    Optimal Transport / Flow Matching:
    x_t = (1 - t) * x_0 + t * ε

    前向核: q(x_t|x_0) = N(x_t; (1-t)*x_0, t² * I)  (当 ε ~ N(0,I))
    实际 Flow Matching 用条件路径: x_t = (1-t)*x_0 + t*x_1

    velocity: v(x_t, t) = x_1 - x_0 = ε - x_0
    """

    def __init__(self, cfg):
        num_steps = cfg.scheduler.num_diffusion_steps
        super().__init__(num_steps)

        of_cfg = cfg.scheduler.of
        self.sigma_min = of_cfg.get("sigma_min", 0.001)

    def marginal_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """OF: mean_coeff = (1-t), std = t"""
        mean_coeff = 1.0 - t
        std = t.clamp(min=self.sigma_min)
        return mean_coeff, std

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        mean_coeff, std = self.marginal_params(t)
        return (mean_coeff / std.clamp(min=1e-8)) ** 2

    def compute_target_velocity(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Flow Matching 的训练目标: v = ε - x_0"""
        return noise - x_0

    def add_noise(self, x_0, t, noise=None):
        """x_t = (1-t)*x_0 + t*ε"""
        if noise is None:
            noise = torch.randn_like(x_0)
        t_bc = self._broadcast(t, x_0)
        x_t = (1.0 - t_bc) * x_0 + t_bc * noise
        return x_t, noise
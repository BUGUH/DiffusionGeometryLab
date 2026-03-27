"""VE (Variance Exploding) 调度器 — NCSN / SMLD。"""

from __future__ import annotations
import math
from typing import Tuple

import torch
from src.schedulers.base_scheduler import BaseScheduler


class VEScheduler(BaseScheduler):
    """
    VE-SDE: dx = √(dσ²/dt) dw

    前向核: q(x_t|x_0) = N(x_t; x_0, σ(t)² I)
    σ(t) 按几何级数从 σ_min 增长到 σ_max。
    """

    def __init__(self, cfg):
        num_steps = cfg.scheduler.num_diffusion_steps
        super().__init__(num_steps)

        ve_cfg = cfg.scheduler.ve
        self.sigma_min = ve_cfg.get("sigma_min", 0.01)
        self.sigma_max = ve_cfg.get("sigma_max", 50.0)

        # 预计算离散 σ 表（几何级数）
        self.sigmas = torch.exp(
            torch.linspace(math.log(self.sigma_min), math.log(self.sigma_max), num_steps)
        )

    def marginal_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """VE: mean_coeff = 1, std = σ(t)"""
        mean_coeff = torch.ones_like(t)
        # σ(t) = σ_min * (σ_max/σ_min)^t
        log_sigma = math.log(self.sigma_min) + t * (math.log(self.sigma_max) - math.log(self.sigma_min))
        std = torch.exp(log_sigma)
        return mean_coeff, std

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        _, std = self.marginal_params(t)
        return 1.0 / (std ** 2).clamp(min=1e-10)

    def sigma_at_step(self, step: torch.Tensor) -> torch.Tensor:
        return self.sigmas.to(step.device)[step.long()]
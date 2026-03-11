"""VP (Variance Preserving) 调度器 — DDPM / Score SDE。"""

from __future__ import annotations
import math
from typing import Tuple

import torch
from src.schedulers.base_scheduler import BaseScheduler


class VPScheduler(BaseScheduler):
    """
    VP-SDE: dx = -0.5 * β(t) * x dt + √β(t) dw

    前向核: q(x_t|x_0) = N(x_t; √ᾱ(t) x_0, (1-ᾱ(t)) I)
    其中 ᾱ(t) = exp(-∫₀ᵗ β(s) ds)

    支持 linear 和 cosine β 调度。
    """

    def __init__(self, cfg):
        num_steps = cfg.scheduler.num_diffusion_steps
        super().__init__(num_steps)

        vp_cfg = cfg.scheduler.vp
        self.beta_schedule = vp_cfg.get("beta_schedule", "linear")
        self.beta_start = vp_cfg.get("beta_start", 0.0001)
        self.beta_end = vp_cfg.get("beta_end", 0.02)

        # 预计算离散 β 和 ᾱ
        if self.beta_schedule == "linear":
            self.betas = torch.linspace(self.beta_start, self.beta_end, num_steps)
        elif self.beta_schedule == "cosine":
            self.betas = self._cosine_betas(num_steps)
        else:
            raise ValueError(f"未知 beta_schedule: {self.beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def marginal_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """连续时间版本，通过积分 β(s) 计算。"""
        # β(t) = β_min + t * (β_max - β_min)
        beta_integral = self.beta_start * t + 0.5 * (self.beta_end - self.beta_start) * t ** 2
        log_mean_coeff = -0.5 * beta_integral
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt((1.0 - torch.exp(2.0 * log_mean_coeff)).clamp(min=1e-10))
        return mean_coeff, std

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        mean_coeff, std = self.marginal_params(t)
        return (mean_coeff / std.clamp(min=1e-8)) ** 2

    def marginal_params_discrete(self, step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """离散版本，使用预计算表。"""
        abar = self.alpha_cumprod.to(step.device)[step.long()]
        return abar.sqrt(), (1.0 - abar).sqrt()

    def add_noise_discrete(self, x_0, step, noise=None):
        """离散时间步加噪（训练常用）。"""
        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coeff, std = self.marginal_params_discrete(step)
        mean_coeff = self._broadcast(mean_coeff, x_0)
        std = self._broadcast(std, x_0)
        return mean_coeff * x_0 + std * noise, noise

    @staticmethod
    def _cosine_betas(num_steps: int, s: float = 0.008) -> torch.Tensor:
        t = torch.linspace(0, 1, num_steps + 1)
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999)
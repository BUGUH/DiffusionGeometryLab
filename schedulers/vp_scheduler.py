"""VP (Variance Preserving) 调度器 — 完美兼容理论与训练代码。"""
from __future__ import annotations
import math
from typing import Tuple
import torch
from src.schedulers.base_scheduler import BaseScheduler

class VPScheduler(BaseScheduler):
    """
    VP-SDE (Trigonometric form): 
    严格满足 α_t^2 + σ_t^2 = 1。
    对应理论要求 (t=0 为数据，t=1 为纯噪声):
    α(t) = cos(πt/2)
    σ(t) = sin(πt/2)
    """
    def __init__(self, cfg):
        num_steps = cfg.scheduler.num_diffusion_steps
        super().__init__(num_steps)
        
        # 为了防止你的外部训练代码强依赖离散属性，我们根据理论公式反推出 betas 和 alphas
        # 连续时间 t_grid
        t_grid = torch.linspace(0, 1, num_steps + 1)
        
        # 理论要求的连续 alpha_bar (即 α_t^2)
        # 因为代码的 mean_coeff 是 α_t，而离散形式里的 alpha_cumprod 是 α_t^2
        f = torch.cos(t_grid * math.pi / 2.0) ** 2
        
        # 按照 DDPM 的标准转换方式计算离散的 alphas 和 betas
        alpha_bar_discrete = f / f[0]
        self.alphas = alpha_bar_discrete[1:] / alpha_bar_discrete[:-1]
        self.betas = (1.0 - self.alphas).clamp(max=0.999)
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def marginal_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """连续时间版本：直接使用你理论推导的严谨三角函数。"""
        # t 是代码时间 (0: 数据 -> 1: 噪声)
        pi_over_2 = math.pi / 2.0
        
        # α_t = cos(πt/2)
        mean_coeff = torch.cos(pi_over_2 * t)
        # σ_t = sin(πt/2)
        std = torch.sin(pi_over_2 * t).clamp(min=1e-8)
        
        return mean_coeff, std

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        mean_coeff, std = self.marginal_params(t)
        return (mean_coeff / std.clamp(min=1e-8)) ** 2

    def marginal_params_discrete(self, step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """离散版本，使用预计算表 (保持与原代码 API 完全一致)。"""
        abar = self.alpha_cumprod.to(step.device)[step.long()]
        return abar.sqrt(), (1.0 - abar).sqrt()

    def add_noise_discrete(self, x_0, step, noise=None):
        """离散时间步加噪（保持与原代码 API 完全一致）。"""
        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coeff, std = self.marginal_params_discrete(step)
        mean_coeff = self._broadcast(mean_coeff, x_0)
        std = self._broadcast(std, x_0)
        return mean_coeff * x_0 + std * noise, noise
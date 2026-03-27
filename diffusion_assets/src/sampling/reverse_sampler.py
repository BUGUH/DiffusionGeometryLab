"""
反向去噪采样器
==============
支持 Euler-Maruyama / DDIM / Heun 求解器。
支持任意时间步截取中间态 x_t 及对应得分向量。
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from src.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


class TrajectoryBundle:
    """采样轨迹包：存储中间状态和得分。"""

    def __init__(self):
        self.x_ts: Dict[float, torch.Tensor] = {}     # t → (B, C, H, W)
        self.scores: Dict[float, torch.Tensor] = {}    # t → (B, C, H, W)
        self.pred_x0s: Dict[float, torch.Tensor] = {}  # t → (B, C, H, W)

    def add(self, t_val: float, x_t: torch.Tensor,
            score: Optional[torch.Tensor] = None,
            pred_x0: Optional[torch.Tensor] = None):
        self.x_ts[t_val] = x_t.cpu()
        if score is not None:
            self.scores[t_val] = score.cpu()
        if pred_x0 is not None:
            self.pred_x0s[t_val] = pred_x0.cpu()

    @property
    def timesteps(self) -> List[float]:
        return sorted(self.x_ts.keys(), reverse=True)

    @property
    def final_samples(self) -> torch.Tensor:
        min_t = min(self.x_ts.keys())
        return self.x_ts[min_t]


class ReverseSampler:
    """
    统一反向采样器。

    Parameters
    ----------
    model : nn.Module
        得分网络（ScoreNetwork）。
    scheduler : BaseScheduler
    method : str
        "euler_maruyama" | "ddim" | "heun"
    num_steps : int
        采样步数。
    capture_timesteps : list[float]
        需要截取的归一化时间点 ∈ [0, 1]。
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: BaseScheduler,
        method: str = "ddim",
        num_steps: int = 100,
        capture_timesteps: Optional[List[float]] = None,
    ):
        self.model = model
        self.scheduler = scheduler
        self.method = method.lower()
        self.num_steps = num_steps
        self.capture_timesteps = set(capture_timesteps or [])

        # 构建采样时间表（从 1→0）
        self.ts = torch.linspace(1.0, 1e-5, num_steps + 1)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        initial_noise: Optional[torch.Tensor] = None,
        return_trajectory: bool = True,
    ) -> Tuple[torch.Tensor, Optional[TrajectoryBundle]]:
        """
        执行反向采样。

        Parameters
        ----------
        shape : (B, C, H, W)
        device : torch.device
        initial_noise : 可选的固定初始噪声
        return_trajectory : 是否记录轨迹

        Returns
        -------
        (final_samples, trajectory_bundle)
        """
        self.model.eval()
        ts = self.ts.to(device)

        # 初始化
        if initial_noise is not None:
            x = initial_noise.to(device)
        else:
            x = torch.randn(shape, device=device)

        bundle = TrajectoryBundle() if return_trajectory else None

        # 记录初始状态
        if bundle and self._should_capture(1.0):
            bundle.add(1.0, x)

        for i in range(self.num_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            t_batch = t_cur.expand(shape[0])

            if self.method == "ddim":
                x, pred_noise, pred_x0 = self._step_ddim(x, t_batch, t_cur, t_next)
            elif self.method == "euler_maruyama":
                x, pred_noise, pred_x0 = self._step_euler(x, t_batch, t_cur, t_next)
            elif self.method == "heun":
                x, pred_noise, pred_x0 = self._step_heun(x, t_batch, t_cur, t_next)
            else:
                raise ValueError(f"未知采样方法: {self.method}")

            # 捕获中间态
            t_val = t_next.item()
            if bundle and self._should_capture(t_val):
                sigma = self.scheduler.get_sigma(t_batch)
                sigma_bc = self.scheduler._broadcast(sigma, x)
                score = -pred_noise / sigma_bc.clamp(min=1e-8)
                bundle.add(t_val, x, score=score, pred_x0=pred_x0)

        # 最终样本
        if bundle and 0.0 not in bundle.x_ts:
            bundle.add(0.0, x)

        return x, bundle

    def _step_ddim(self, x, t_batch, t_cur, t_next):
        """DDIM 确定性采样步。"""
        pred_noise = self.model(x, t_batch)
        mean_cur, std_cur = self.scheduler.marginal_params(t_cur.unsqueeze(0))
        mean_next, std_next = self.scheduler.marginal_params(t_next.unsqueeze(0))

        mean_cur = self.scheduler._broadcast(mean_cur, x)
        std_cur = self.scheduler._broadcast(std_cur, x)
        mean_next = self.scheduler._broadcast(mean_next, x)
        std_next = self.scheduler._broadcast(std_next, x)

        # 预测 x_0
        pred_x0 = (x - std_cur * pred_noise) / mean_cur.clamp(min=1e-8)
        pred_x0 = pred_x0.clamp(-3, 3)  # 稳定性裁剪

        # DDIM 更新
        x_next = mean_next * pred_x0 + std_next * pred_noise
        return x_next, pred_noise, pred_x0

    def _step_euler(self, x, t_batch, t_cur, t_next):
        """Euler-Maruyama 随机采样步。"""
        pred_noise = self.model(x, t_batch)
        mean_cur, std_cur = self.scheduler.marginal_params(t_cur.unsqueeze(0))
        mean_next, std_next = self.scheduler.marginal_params(t_next.unsqueeze(0))

        mean_cur = self.scheduler._broadcast(mean_cur, x)
        std_cur = self.scheduler._broadcast(std_cur, x)
        mean_next = self.scheduler._broadcast(mean_next, x)
        std_next = self.scheduler._broadcast(std_next, x)

        pred_x0 = (x - std_cur * pred_noise) / mean_cur.clamp(min=1e-8)
        pred_x0 = pred_x0.clamp(-3, 3)

        # 确定性部分
        x_next = mean_next * pred_x0 + std_next * pred_noise
        # 随机部分
        dt = (t_cur - t_next).abs()
        noise_scale = (dt.sqrt() * 0.5).clamp(max=0.1)
        x_next = x_next + noise_scale * torch.randn_like(x)
        return x_next, pred_noise, pred_x0

    def _step_heun(self, x, t_batch, t_cur, t_next):
        """Heun 二阶求解器。"""
        # 第一步 Euler 预测
        pred_noise_1 = self.model(x, t_batch)
        mean_cur, std_cur = self.scheduler.marginal_params(t_cur.unsqueeze(0))
        mean_next, std_next = self.scheduler.marginal_params(t_next.unsqueeze(0))

        mean_cur = self.scheduler._broadcast(mean_cur, x)
        std_cur = self.scheduler._broadcast(std_cur, x)
        mean_next = self.scheduler._broadcast(mean_next, x)
        std_next = self.scheduler._broadcast(std_next, x)

        pred_x0_1 = (x - std_cur * pred_noise_1) / mean_cur.clamp(min=1e-8)
        pred_x0_1 = pred_x0_1.clamp(-3, 3)
        x_euler = mean_next * pred_x0_1 + std_next * pred_noise_1

        # 第二步：在 x_euler 处再求一次
        t_next_batch = t_next.expand(x.shape[0])
        pred_noise_2 = self.model(x_euler, t_next_batch)
        pred_x0_2 = (x_euler - std_next * pred_noise_2) / mean_next.clamp(min=1e-8)
        pred_x0_2 = pred_x0_2.clamp(-3, 3)

        # Heun 平均
        pred_x0 = 0.5 * (pred_x0_1 + pred_x0_2)
        x_next = mean_next * pred_x0 + std_next * 0.5 * (pred_noise_1 + pred_noise_2)
        return x_next, pred_noise_1, pred_x0

    def _should_capture(self, t_val: float) -> bool:
        """判断当前时间步是否需要截取。"""
        if not self.capture_timesteps:
            return True
        for tc in self.capture_timesteps:
            if abs(t_val - tc) < 0.5 / self.num_steps:
                return True
        return False
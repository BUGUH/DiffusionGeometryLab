"""训练/测试集重建误差差距分析。"""

from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


class ReconstructionGapEvaluator:
    """
    在多个时间步处计算单步去噪重建误差。
    分别统计训练/测试集，报告差距。
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        timesteps: List[float] = None,
        num_samples: int = 1000,
    ):
        self.scheduler = scheduler
        self.timesteps = timesteps or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.num_samples = num_samples

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """
        Returns
        -------
        {
            "timesteps": array,
            "train_mse": array,
            "test_mse": array,
            "gap": array,  # (test - train) / train
        }
        """
        model.eval()
        train_mse = []
        test_mse = []

        for t_val in self.timesteps:
            t_mse = self._compute_mse_for_split(model, train_loader, t_val, device)
            v_mse = self._compute_mse_for_split(model, test_loader, t_val, device)
            train_mse.append(t_mse)
            test_mse.append(v_mse)

        train_mse = np.array(train_mse)
        test_mse = np.array(test_mse)
        gap = (test_mse - train_mse) / np.maximum(train_mse, 1e-10)

        return {
            "timesteps": np.array(self.timesteps),
            "train_mse": train_mse,
            "test_mse": test_mse,
            "gap": gap,
        }

    def _compute_mse_for_split(
        self, model, loader, t_val, device,
    ) -> float:
        total_mse, count = 0.0, 0

        for x_0, _ in loader:
            x_0 = x_0.to(device)
            B = x_0.shape[0]
            if count + B > self.num_samples:
                x_0 = x_0[:self.num_samples - count]
                B = x_0.shape[0]

            t = torch.full((B,), t_val, device=device)
            noise = torch.randn_like(x_0)
            x_t, _ = self.scheduler.add_noise(x_0, t, noise)

            pred_noise = model(x_t, t)
            mean_c, std = self.scheduler.marginal_params(t)
            mean_c = self.scheduler._broadcast(mean_c, x_0)
            std = self.scheduler._broadcast(std, x_0)
            x_0_pred = (x_t - std * pred_noise) / mean_c.clamp(min=1e-8)

            mse = ((x_0_pred - x_0) ** 2).mean(dim=[1, 2, 3])
            total_mse += mse.sum().item()
            count += B

            if count >= self.num_samples:
                break

        return total_mse / max(count, 1)
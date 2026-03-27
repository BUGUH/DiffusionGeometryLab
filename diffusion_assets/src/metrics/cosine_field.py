"""
余弦相似度场分析
================
实现三种余弦相似度场计算:
A1. S_cos(M_gen, M_mem): 同一 x_t 输入两个模型的输出对齐度
A2. S_cos(model, empirical): 模型预测 vs 指向训练样本的理想方向
A3. S_cos(train, test): 语义相似对注入相同噪声后的输出对齐度
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.schedulers.base_scheduler import BaseScheduler
from src.utils.math_ops import batch_cosine_similarity

logger = logging.getLogger(__name__)


class CosineFieldEvaluator:
    """
    余弦相似度场统一评估器。

    Parameters
    ----------
    scheduler : BaseScheduler
    num_timesteps : int
        t 轴采样点数。
    batch_size : int
        每个时间步的计算批量。
    t_min, t_max : float
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        num_timesteps: int = 50,
        batch_size: int = 256,
        t_min: float = 0.001,
        t_max: float = 0.999,
    ):
        self.scheduler = scheduler
        self.t_values = np.linspace(t_min, t_max, num_timesteps)
        self.batch_size = batch_size

    @torch.no_grad()
    def compute_gen_mem_similarity(
        self,
        model_gen: nn.Module,
        model_mem: nn.Module,
        data: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """
        S_cos(M_gen, M_mem): 同一噪声样本在两个模型下的输出对齐度。

        对每个 t:
        1. 从 data 采样 x_0, 加噪得 x_t
        2. 分别输入 model_gen 和 model_mem 得到预测
        3. 计算余弦相似度

        Returns
        -------
        {"t": array, "mean": array, "std": array}
        """
        model_gen.eval()
        model_mem.eval()
        means, stds = [], []

        for t_val in self.t_values:
            sims = self._compute_at_t(
                t_val, data, device,
                lambda x_t, t_b: model_gen(x_t, t_b),
                lambda x_t, t_b: model_mem(x_t, t_b),
            )
            means.append(sims.mean().item())
            stds.append(sims.std().item())

        return {"t": self.t_values, "mean": np.array(means), "std": np.array(stds)}

    @torch.no_grad()
    def compute_model_empirical_similarity(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """
        S_cos(model, empirical): 模型预测 vs 指向最近训练样本的理想梯度。

        M_emp 定义: 对加噪样本 x_t，理想方向是 (x_0_nearest - x_t) 归一化。
        模型方向: v = -ŝ_θ(x_t, t)（去噪方向）。

        Returns
        -------
        {"t": array, "mean": array, "std": array}
        """
        model.eval()
        means, stds = [], []
        train_flat = train_data.view(len(train_data), -1).to(device)

        for t_val in self.t_values:
            t_tensor = torch.full((self.batch_size,), t_val, device=device)
            # 随机选训练样本加噪
            idx = torch.randint(0, len(train_data), (self.batch_size,))
            x_0 = train_data[idx].to(device)
            noise = torch.randn_like(x_0)
            x_t, _ = self.scheduler.add_noise(x_0, t_tensor, noise)

            # 模型预测方向（去噪方向 = -noise_pred 的方向）
            pred_noise = model(x_t, t_tensor)
            model_dir = -pred_noise  # 去噪方向

            # 经验方向：x_0 - x_t（指向干净数据）
            empirical_dir = x_0 - x_t

            # 展平计算余弦
            model_flat = model_dir.view(self.batch_size, -1)
            emp_flat = empirical_dir.view(self.batch_size, -1)
            sims = batch_cosine_similarity(model_flat, emp_flat)

            means.append(sims.mean().item())
            stds.append(sims.std().item())

        return {"t": self.t_values, "mean": np.array(means), "std": np.array(stds)}

    @torch.no_grad()
    def compute_train_test_similarity(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        pairs: List[Tuple[int, int]],
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """
        S_cos(Train, Test): 语义配对注入相同噪声后的输出对齐度。

        对每个配对 (x_train, x_test):
        1. 生成相同噪声 ε
        2. x_t^train = add_noise(x_train, t, ε)
        3. x_t^test = add_noise(x_test, t, ε)
        4. cos_sim(model(x_t^train), model(x_t^test))

        Returns
        -------
        {"t": array, "mean": array, "std": array}
        """
        model.eval()
        means, stds = [], []
        n_pairs = min(len(pairs), self.batch_size)

        # 提取配对数据
        train_idx = [p[0] for p in pairs[:n_pairs]]
        test_idx = [p[1] for p in pairs[:n_pairs]]
        x_train = train_data[train_idx].to(device)
        x_test = test_data[test_idx].to(device)

        # 固定噪声（关键：相同噪声注入）
        shared_noise = torch.randn_like(x_train)

        for t_val in self.t_values:
            t_tensor = torch.full((n_pairs,), t_val, device=device)

            x_t_train, _ = self.scheduler.add_noise(x_train, t_tensor, shared_noise)
            x_t_test, _ = self.scheduler.add_noise(x_test, t_tensor, shared_noise)

            pred_train = model(x_t_train, t_tensor)
            pred_test = model(x_t_test, t_tensor)

            pred_train_flat = pred_train.view(n_pairs, -1)
            pred_test_flat = pred_test.view(n_pairs, -1)
            sims = batch_cosine_similarity(pred_train_flat, pred_test_flat)

            means.append(sims.mean().item())
            stds.append(sims.std().item())

        return {"t": self.t_values, "mean": np.array(means), "std": np.array(stds)}

    def _compute_at_t(self, t_val, data, device, fn_a, fn_b):
        """在时间步 t 处批量计算两个函数输出的余弦相似度。"""
        idx = torch.randint(0, len(data), (self.batch_size,))
        x_0 = data[idx].to(device)
        t_tensor = torch.full((self.batch_size,), t_val, device=device)
        noise = torch.randn_like(x_0)
        x_t, _ = self.scheduler.add_noise(x_0, t_tensor, noise)

        out_a = fn_a(x_t, t_tensor)
        out_b = fn_b(x_t, t_tensor)

        flat_a = out_a.view(self.batch_size, -1)
        flat_b = out_b.view(self.batch_size, -1)
        return batch_cosine_similarity(flat_a, flat_b)
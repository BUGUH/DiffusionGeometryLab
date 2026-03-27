"""
MLP 得分网络 — 适配 D 维向量数据（Circle 等）
=============================================
替代 UNet，接收 (B, 1, D) 输入。
"""

from __future__ import annotations
import math
from typing import List

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLPScoreNet(nn.Module):
    """
    MLP 得分网络。

    输入: (B, 1, D) 或 (B, D) 向量 + (B,) 时间步
    输出: (B, 1, D) 预测噪声/得分/速度

    Parameters
    ----------
    data_dim : int
        数据维度 D。
    hidden_dims : list[int]
        隐藏层维度列表。
    time_embed_dim : int
    dropout : float
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dims: List[int] = (512, 512, 512),
        time_embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 主干网络
        dims = [data_dim + time_embed_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
        layers.append(nn.Linear(dims[-1], data_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, D) 或 (B, D)
        t : (B,)
        返回 : 与 x 同形状
        """
        input_shape = x.shape
        # 展平
        if x.dim() == 3:
            x_flat = x.view(x.shape[0], -1)  # (B, D)
        else:
            x_flat = x

        t_emb = self.time_embed(t)  # (B, time_embed_dim)
        h = torch.cat([x_flat, t_emb], dim=-1)
        out = self.net(h)  # (B, D)

        # 恢复原始形状
        return out.view(input_shape)
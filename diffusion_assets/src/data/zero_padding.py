"""零填充：将低分辨率图像嵌入高维空间。"""

from __future__ import annotations
import math
import logging

import torch

logger = logging.getLogger(__name__)


class ZeroPadder:
    """
    将 (1, H_src, H_src) 图像零填充至 (1, H_tgt, H_tgt)。

    Parameters
    ----------
    source_size : int
        原始图像边长（如 32）。
    target_size : int
        目标图像边长（如 64, 128）。
    mode : str
        "center": 图像居中，四周补零。
        "top_left": 图像贴左上角。
    """

    def __init__(self, source_size: int, target_size: int, mode: str = "center"):
        assert target_size >= source_size, (
            f"target_size({target_size}) 必须 >= source_size({source_size})"
        )
        self.source_size = source_size
        self.target_size = target_size
        self.mode = mode

        # 预计算填充偏移
        if mode == "center":
            self.offset_h = (target_size - source_size) // 2
            self.offset_w = (target_size - source_size) // 2
        elif mode == "top_left":
            self.offset_h = 0
            self.offset_w = 0
        else:
            raise ValueError(f"未知填充模式: {mode}")

        self.m = source_size * source_size      # 有效维度
        self.d = target_size * target_size       # 环境维度
        logger.info(f"ZeroPadder: {source_size}²(m={self.m}) → {target_size}²(d={self.d}), "
                     f"d/m={self.d/self.m:.1f}, mode={mode}")

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 1, H_src, H_src) 或 (1, H_src, H_src)

        Returns
        -------
        torch.Tensor, shape (..., 1, H_tgt, H_tgt)
        """
        if self.source_size == self.target_size:
            return x

        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True

        N = x.shape[0]
        out = torch.zeros(N, 1, self.target_size, self.target_size,
                          dtype=x.dtype, device=x.device)
        oh, ow = self.offset_h, self.offset_w
        s = self.source_size
        out[:, :, oh:oh+s, ow:ow+s] = x

        return out.squeeze(0) if squeeze else out

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        """逆操作：从填充图像中提取原始区域。"""
        if self.source_size == self.target_size:
            return x

        oh, ow, s = self.offset_h, self.offset_w, self.source_size
        if x.dim() == 3:
            return x[:, oh:oh+s, ow:ow+s]
        return x[:, :, oh:oh+s, ow:ow+s]

    def __repr__(self) -> str:
        return f"ZeroPadder({self.source_size}→{self.target_size}, mode={self.mode})"
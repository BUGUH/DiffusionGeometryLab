"""LPIPS + SSIM 过拟合度量。"""

from __future__ import annotations
import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from src.metrics.nn_search import FaissNNIndex

logger = logging.getLogger(__name__)


class LPIPSSSIMEvaluator:
    """
    对生成样本在训练集中找最近邻，计算 LPIPS 和 SSIM。
    LPIPS 极低 + SSIM 极高 → 记忆化（过拟合）。
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self._lpips_fn = None

    @torch.no_grad()
    def evaluate(
        self,
        generated: torch.Tensor,
        train_data: torch.Tensor,
        num_samples: int = 500,
    ) -> Dict[str, float]:
        """
        Returns
        -------
        {"lpips_mean", "lpips_std", "ssim_mean", "ssim_std"}
        """
        N = min(num_samples, len(generated), len(train_data))
        gen = generated[:N]
        dim = train_data.view(len(train_data), -1).shape[1]

        # 找最近邻
        index = FaissNNIndex(dim=dim)
        index.build(train_data.view(len(train_data), -1))
        _, nn_idx = index.query(gen.view(N, -1), k=1)
        nn_idx = nn_idx[:, 0]
        nearest = train_data[nn_idx]

        # SSIM
        ssim_vals = self._batch_ssim(gen, nearest)

        # LPIPS（尝试使用 lpips 库，否则用简化版本）
        lpips_vals = self._batch_lpips(gen, nearest)

        return {
            "lpips_mean": lpips_vals.mean().item(),
            "lpips_std": lpips_vals.std().item(),
            "ssim_mean": ssim_vals.mean().item(),
            "ssim_std": ssim_vals.std().item(),
        }

    def _batch_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """计算 LPIPS，回退到 L2 特征距离。"""
        try:
            import lpips
            if self._lpips_fn is None:
                self._lpips_fn = lpips.LPIPS(net="squeeze").to(self.device)
                self._lpips_fn.eval()

            # LPIPS 需要 3 通道 [-1, 1]
            i1 = self._prepare_for_lpips(img1).to(self.device)
            i2 = self._prepare_for_lpips(img2).to(self.device)

            # 分批计算避免 OOM
            results = []
            bs = 64
            for start in range(0, len(i1), bs):
                end = min(start + bs, len(i1))
                d = self._lpips_fn(i1[start:end], i2[start:end])
                results.append(d.squeeze().cpu())
            return torch.cat(results)

        except ImportError:
            logger.warning("lpips 未安装，使用 L2 特征距离代替")
            return self._simple_perceptual_distance(img1, img2)

    @staticmethod
    def _prepare_for_lpips(images: torch.Tensor) -> torch.Tensor:
        """转换为 LPIPS 要求的格式：3通道 [-1, 1]。"""
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if images.min() >= 0:
            images = images * 2 - 1  # [0,1] → [-1,1]
        return images.clamp(-1, 1)

    @staticmethod
    def _simple_perceptual_distance(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """简化感知距离：多尺度 L2。"""
        distances = []
        a, b = img1.float(), img2.float()
        for scale in range(3):
            diff = ((a - b) ** 2).mean(dim=[1, 2, 3])
            distances.append(diff)
            if a.shape[-1] > 4:
                a = F.avg_pool2d(a, 2)
                b = F.avg_pool2d(b, 2)
        return torch.stack(distances).mean(dim=0)

    @staticmethod
    def _batch_ssim(
        img1: torch.Tensor, img2: torch.Tensor,
        window_size: int = 7, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
    ) -> torch.Tensor:
        """
        批量 SSIM 计算。

        Parameters
        ----------
        img1, img2 : (B, C, H, W)

        Returns
        -------
        (B,) SSIM 值
        """
        # 归一化到 [0, 1]
        a = img1.float()
        b = img2.float()
        if a.min() < 0:
            a = (a + 1) / 2
            b = (b + 1) / 2

        C = a.shape[1]
        # 高斯窗口
        kernel = _gaussian_kernel_2d(window_size, 1.5).to(a.device)
        kernel = kernel.expand(C, 1, -1, -1)
        pad = window_size // 2

        mu1 = F.conv2d(a, kernel, padding=pad, groups=C)
        mu2 = F.conv2d(b, kernel, padding=pad, groups=C)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(a * a, kernel, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(b * b, kernel, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(a * b, kernel, padding=pad, groups=C) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean(dim=[1, 2, 3])


def _gaussian_kernel_2d(size: int, sigma: float) -> torch.Tensor:
    """创建 2D 高斯核。"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.unsqueeze(0).unsqueeze(0)
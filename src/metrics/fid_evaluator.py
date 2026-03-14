"""FID/MMD 评估器 — 适配图像和向量数据 (融合业界标准 Inception 提取)."""
from __future__ import annotations  # 修复了 __future__
import logging
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

from src.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)

class FIDEvaluator:
    """
    分布距离评估器 (终极版)。
    支持三种模式计算 Fréchet 距离：
      1. mode="inception": 业界标准 FID (适合 t 接近 1 的最终图像生成质量)
      2. mode="pca": 极速像素级/PCA级特征 (适合全程 t 的流形轨迹追踪)
      3. mode="mmd": 最大均值差异 (适合低维 Circle/瑞士卷 等向量数据)
    """
    def __init__(self, feature_dim: int = 2048, device: torch.device = None): # 修复了 __init__
        self.feature_dim = feature_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inception_model = None  # 懒加载，节省显存

    def _load_inception(self):
        """懒加载 Inception-V3 网络"""
        if self.inception_model is None:
            logger.info("首次调用，正在加载 InceptionV3 预训练模型...")
            self.inception_model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False
            ).to(self.device)
            self.inception_model.fc = torch.nn.Identity() # 截断分类层，保留特征池化层
            self.inception_model.eval()

    @torch.no_grad()
    def _extract_inception_features(self, images: torch.Tensor, batch_size=32) -> torch.Tensor:
        """用神经网络提取特征。输入要求：[-1, 1] 范围的 Tensor (N, C, H, W)"""
        self._load_inception()
        features = []
        # 确保输入在 GPU
        images = images.to(self.device)
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            # 1. 映射到 [0, 1]
            batch = (batch + 1.0) / 2.0
            # 2. 如果是单通道灰度图，复制为 3 通道 (Inception 必需)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            # 3. 插值到 299x299 (Inception 规定尺寸)
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            # 4. 标准化到 Inception 要求的范围 (约 [-1, 1])
            batch = batch * 2.0 - 1.0
            pred = self.inception_model(batch)
            features.append(pred)
        return torch.cat(features, dim=0)

    def compute_fid(self, real: torch.Tensor, fake: torch.Tensor, mode: str = "pca") -> float:
        """
        统一的 FID 计算接口。
        :param mode: "inception" (标准神经网络 FID) 或 "pca" (你的极速像素降维 FID)
        """
        if mode == "inception":
            real_features = self._extract_inception_features(real)
            fake_features = self._extract_inception_features(fake)
            return self._calculate_frechet_distance(real_features, fake_features)
        else:
            real_flat = real.reshape(len(real), -1).float()
            fake_flat = fake.reshape(len(fake), -1).float()
            return self._calculate_frechet_distance(real_flat, fake_flat, use_pca=True)

    def _calculate_frechet_distance(self, real_feat: torch.Tensor, fake_feat: torch.Tensor, use_pca=False):
        real_feat = real_feat.float().cpu()
        fake_feat = fake_feat.float().cpu()
        
        if use_pca:
            max_dim = min(256, real_feat.shape[1], real_feat.shape[0] - 1, fake_feat.shape[0] - 1)
            if max_dim < 2:
                return ((real_feat.mean(0) - fake_feat.mean(0)) ** 2).sum().item()
            if real_feat.shape[1] > max_dim:
                real_feat, fake_feat = self._pca_reduce(real_feat, fake_feat, max_dim)
                
        mu_r, mu_f = real_feat.mean(0), fake_feat.mean(0)
        sigma_r, sigma_f = self._cov(real_feat), self._cov(fake_feat)
        mean_term = ((mu_r - mu_f) ** 2).sum().item()
        trace_r = sigma_r.diagonal().sum().item()
        trace_f = sigma_f.diagonal().sum().item()
        
        try:
            product = sigma_r @ sigma_f
            product = (product + product.T) / 2
            eigvals = torch.linalg.eigvalsh(product).clamp(min=0)
            trace_sqrt = eigvals.sqrt().sum().item()
        except Exception as e:
            logger.warning(f"矩阵求根退化: {e}")
            trace_sqrt = min(trace_r, trace_f)
            
        return max(mean_term + trace_r + trace_f - 2 * trace_sqrt, 0.0)

    def compute_mmd(self, real: torch.Tensor, fake: torch.Tensor, kernel: str = "rbf", bandwidth: float = None) -> float:
        x = real.reshape(len(real), -1).float().to(self.device)
        y = fake.reshape(len(fake), -1).float().to(self.device)
        if bandwidth is None:
            with torch.no_grad():
                dists = torch.cdist(x[:500], y[:500])
                bandwidth = dists.median().item()
                bandwidth = max(bandwidth, 1e-4)
        if kernel == "rbf":
            kxx = self._rbf_kernel(x, x, bandwidth)
            kyy = self._rbf_kernel(y, y, bandwidth)
            kxy = self._rbf_kernel(x, y, bandwidth)
        else:
            raise ValueError(f"未知核函数: {kernel}")
        mmd2 = kxx.mean() + kyy.mean() - 2 * kxy.mean()
        return max(mmd2.item(), 0.0)

    @staticmethod
    def _rbf_kernel(x, y, bandwidth):
        dist_sq = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-dist_sq / (2 * bandwidth ** 2)) # 修复了语法错

    @staticmethod
    def _pca_reduce(a, b, dim):
        combined = torch.cat([a, b], dim=0)
        mean = combined.mean(dim=0, keepdim=True)
        centered = combined - mean
        try:
            _, _, V = torch.pca_lowrank(centered, q=dim) # 修复了 _,_ , V
            return (a - mean) @ V, (b - mean) @ V
        except Exception:
            proj = torch.randn(a.shape[1], dim) / (dim ** 0.5)
            return a @ proj, b @ proj

    @staticmethod
    def _cov(x):
        N = x.shape[0]
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        cov = (centered.T @ centered) / max(N - 1, 1)
        cov += torch.eye(cov.shape[0]) * 1e-6
        return cov

    def compute_trajectory_fid(
        self, real_data: torch.Tensor, trajectory_bundle, scheduler=None,
        timesteps=None, metric: str = "inception", comparison_mode: str = "clean_x0"
    ) -> Dict[float, float]:
        """
        计算轨迹演化评估 (修改版以支持测“图像质量变好”)。
        
        :param metric: "inception" (标准网络), "pca" (像素级), "mmd" (核距离)
        :param comparison_mode: 
               - "clean_x0": (默认) 把所有中间态 x_t 直接和干净原图 x_0 比。用来证明“随着去噪，图像质量越来越高”。
               - "marginal": 把中间态 x_t 和对应加噪的真图 q(x_t) 比。用来证明“ODE轨迹完全贴合物理解析解”。
        """
        if timesteps is None:
            timesteps = sorted(trajectory_bundle.x_ts.keys())
        results = {}
        N = min(len(real_data), 2048)
        
        for t_val in timesteps:
            if t_val not in trajectory_bundle.x_ts:
                continue
                
            fake_t = trajectory_bundle.x_ts[t_val]
            
            # 核心逻辑分支：对比的目标是什么？
            if comparison_mode == "clean_x0":
                # 直接和干净的真实图像 (真理) 比
                real_target = real_data[:N]
            elif comparison_mode == "marginal":
                # 和同等噪声水平的真实图像比 (需传入 scheduler)
                if scheduler is None:
                    raise ValueError("使用 'marginal' 模式必须传入 scheduler")
                t_tensor = torch.full((N,), t_val)
                real_target, _ = scheduler.add_noise(real_data[:N], t_tensor)
            else:
                raise ValueError("comparison_mode 必须是 'clean_x0' 或 'marginal'")
                
            n = min(len(real_target), len(fake_t))
            
            # 根据你选择的指标来算
            if metric == "mmd":
                score = self.compute_mmd(real_target[:n], fake_t[:n])
            elif metric == "inception" or metric == "pca":  # 修复了漏掉的 ==
                score = self.compute_fid(real_target[:n], fake_t[:n], mode=metric)
            else:
                raise ValueError("metric must be 'mmd', 'pca', or 'inception'")
                
            results[t_val] = score
            logger.info(f"Step {t_val:.3f} | Mode: {comparison_mode} | Metric: {metric} | Score: {score:.4f}")
            
        return results
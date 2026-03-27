"""
数学运算工具模块
================
提供实验中高频使用的批量数学运算，全部支持 GPU 加速。

核心函数:
- batch_cosine_similarity: 批量余弦相似度
- pairwise_l2_distance: 成对 L2 距离
- safe_normalize: 安全归一化（避免除零）
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def batch_cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    批量计算两组向量的余弦相似度。

    Parameters
    ----------
    a : torch.Tensor, shape (..., D)
        第一组向量。
    b : torch.Tensor, shape (..., D)
        第二组向量，形状需与 a 广播兼容。
    dim : int, default -1
        沿哪个维度计算内积。
    eps : float, default 1e-8
        数值稳定性常数。

    Returns
    -------
    torch.Tensor, shape (...)
        余弦相似度，值域 [-1, 1]。

    Examples
    --------
    >>> a = torch.randn(64, 1024)  # 64 个 1024 维向量
    >>> b = torch.randn(64, 1024)
    >>> sim = batch_cosine_similarity(a, b)  # shape (64,)
    """
    a_norm = a.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)
    b_norm = b.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)
    return (a * b).sum(dim=dim) / (a_norm * b_norm).squeeze(dim)


def pairwise_l2_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    squared: bool = False,
) -> torch.Tensor:
    """
    计算两组向量之间的成对 L2 距离矩阵。

    使用展开公式 ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b> 避免逐对相减。

    Parameters
    ----------
    a : torch.Tensor, shape (N, D)
        第一组 N 个 D 维向量。
    b : torch.Tensor, shape (M, D)
        第二组 M 个 D 维向量。
    squared : bool, default False
        如果 True，返回平方距离（省去 sqrt）。

    Returns
    -------
    torch.Tensor, shape (N, M)
        距离矩阵，dist[i, j] = ||a[i] - b[j]||_2。

    Notes
    -----
    内存复杂度 O(N*M)，当 N、M 很大时请分批计算。
    """
    # ||a||^2: (N, 1)
    a_sq = (a * a).sum(dim=-1, keepdim=True)
    # ||b||^2: (1, M)
    b_sq = (b * b).sum(dim=-1, keepdim=True).t()
    # -2 * <a, b>: (N, M)
    cross = torch.mm(a, b.t())

    dist_sq = a_sq + b_sq - 2.0 * cross
    # 数值安全：消除浮点误差导致的微小负值
    dist_sq = dist_sq.clamp(min=0.0)

    if squared:
        return dist_sq
    return dist_sq.sqrt()


def pairwise_l2_distance_chunked(
    a: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 1024,
    squared: bool = False,
) -> torch.Tensor:
    """
    分块计算成对 L2 距离矩阵，控制峰值内存。

    Parameters
    ----------
    a : torch.Tensor, shape (N, D)
    b : torch.Tensor, shape (M, D)
    chunk_size : int, default 1024
        每次处理 a 中的行数。
    squared : bool, default False

    Returns
    -------
    torch.Tensor, shape (N, M)
    """
    N = a.shape[0]
    results = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_dist = pairwise_l2_distance(a[start:end], b, squared=squared)
        results.append(chunk_dist)
    return torch.cat(results, dim=0)


def safe_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    安全的 L2 归一化（避免零向量导致 NaN）。

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
    dim : int, default -1
    eps : float, default 1e-8

    Returns
    -------
    torch.Tensor, shape (..., D)
        归一化后的向量，零向量保持为零。
    """
    norm = x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)
    return x / norm


def flat_to_spatial(
    x: torch.Tensor,
    channels: int = 1,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    """
    将展平向量恢复为空间张量。

    Parameters
    ----------
    x : torch.Tensor, shape (B, D) 或 (B, C*H*W)
    channels : int, default 1
    height : int, optional
        如果 None，假设 H = W = sqrt(D / C)。
    width : int, optional

    Returns
    -------
    torch.Tensor, shape (B, C, H, W)
    """
    B, D = x.shape[0], x.shape[-1]
    if height is None:
        import math
        hw = int(math.isqrt(D // channels))
        assert hw * hw * channels == D, (
            f"无法将 D={D} 分解为 C={channels} × H × W（H=W）。"
        )
        height = width = hw
    if width is None:
        width = height
    return x.view(B, channels, height, width)


def spatial_to_flat(x: torch.Tensor) -> torch.Tensor:
    """
    将空间张量展平为向量。

    Parameters
    ----------
    x : torch.Tensor, shape (B, C, H, W)

    Returns
    -------
    torch.Tensor, shape (B, C*H*W)
    """
    return x.view(x.shape[0], -1)


def compute_snr_weights(
    snr: torch.Tensor,
    strategy: str = "truncated_snr",
    min_snr_gamma: float = 5.0,
) -> torch.Tensor:
    """
    根据 SNR 计算损失权重（Min-SNR-γ 策略）。

    Parameters
    ----------
    snr : torch.Tensor, shape (B,)
        信噪比。
    strategy : str, default "truncated_snr"
        权重策略：
        - "uniform": 均匀权重（全 1）。
        - "snr": 权重 = SNR。
        - "truncated_snr": 权重 = min(SNR, γ) / SNR（Min-SNR-γ）。
    min_snr_gamma : float, default 5.0
        截断阈值 γ。

    Returns
    -------
    torch.Tensor, shape (B,)
    """
    if strategy == "uniform":
        return torch.ones_like(snr)
    elif strategy == "snr":
        return snr
    elif strategy == "truncated_snr":
        # Min-SNR-γ: w(t) = min(SNR(t), γ) / SNR(t)
        clamped = snr.clamp(max=min_snr_gamma)
        return clamped / snr.clamp(min=1e-8)
    else:
        raise ValueError(f"未知的 SNR 权重策略: '{strategy}'")


def extract_into_tensor(
    arr: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: torch.Size,
) -> torch.Tensor:
    """
    从 1D 数组中按 timestep 索引提取值，并广播到目标形状。

    这是扩散模型中的标准操作：从预计算的 alpha/sigma 表中提取当前时间步的值。

    Parameters
    ----------
    arr : torch.Tensor, shape (T,)
        预计算的参数数组（如 alpha_cumprod）。
    timesteps : torch.Tensor, shape (B,)
        当前批次的时间步索引。
    broadcast_shape : torch.Size
        目标广播形状，通常为 (B, C, H, W)。

    Returns
    -------
    torch.Tensor, shape = broadcast_shape
        提取并广播后的参数值。

    Examples
    --------
    >>> alphas = torch.linspace(1.0, 0.01, 1000)
    >>> t = torch.tensor([0, 100, 500, 999])
    >>> result = extract_into_tensor(alphas, t, (4, 1, 64, 64))
    """
    res = arr.to(timesteps.device)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res.expand(broadcast_shape)


@torch.no_grad()
def estimate_manifold_dimension(
    data: torch.Tensor,
    k: int = 10,
    subsample: int = 1000,
) -> float:
    """
    使用最近邻距离的 MLE 估计数据流形维度（Levina & Bickel, 2004）。

    Parameters
    ----------
    data : torch.Tensor, shape (N, D)
        数据矩阵。
    k : int, default 10
        使用的近邻数量。
    subsample : int, default 1000
        随机子采样数量（加速大数据集）。

    Returns
    -------
    float
        估计的内在维度。

    Notes
    -----
    公式: d_hat = [ (1/N) Σ_i (1/(k-1)) Σ_{j=1}^{k-1} log(r_k(i)/r_j(i)) ]^{-1}
    其中 r_j(i) 是第 i 个点到其第 j 近邻的距离。
    """
    N, D = data.shape
    if N > subsample:
        indices = torch.randperm(N)[:subsample]
        data = data[indices]
        N = subsample

    # 计算成对距离
    dists = pairwise_l2_distance(data, data)  # (N, N)
    # 排除自身（对角线设为 inf）
    dists.fill_diagonal_(float("inf"))
    # 取前 k 个最近邻距离
    knn_dists, _ = dists.topk(k, dim=1, largest=False)  # (N, k)

    # MLE 估计
    # r_k: 第 k 个近邻距离 (N,)
    r_k = knn_dists[:, -1].clamp(min=1e-10)
    # log(r_k / r_j) for j = 1..k-1
    log_ratios = torch.log(r_k.unsqueeze(1) / knn_dists[:, :-1].clamp(min=1e-10))
    # 对每个点平均，再对所有点平均
    m_hat = log_ratios.mean(dim=1)  # (N,)
    d_hat = 1.0 / m_hat.mean().item()

    return max(d_hat, 1.0)  # 至少为 1
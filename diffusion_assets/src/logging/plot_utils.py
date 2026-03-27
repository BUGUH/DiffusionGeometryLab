"""
绘图工具模块
============
生成实验专用的高质量图表，支持保存为文件和直接传给日志后端。

所有绘图函数返回 matplotlib Figure 对象，调用方决定保存/显示/日志。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合服务器
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


# ---- 全局样式 ----
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(STYLE_CONFIG)

# 维度对应的颜色映射
DIMENSION_COLORS = {
    1024: "#1f77b4",   # 蓝（m=1024, 基准）
    4096: "#ff7f0e",   # 橙（d=4096）
    16384: "#2ca02c",  # 绿（d=16384）
    65536: "#d62728",  # 红（d=65536）
}

# 相态颜色
PHASE_COLORS = {
    "gen": "#2196F3",  # 蓝
    "mem": "#F44336",  # 红
    "empirical": "#4CAF50",  # 绿
}


def plot_cosine_field_curves(
    results: Dict[str, Dict[str, np.ndarray]],
    title: str = "Cosine Similarity Field",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    绘制余弦相似度场随时间步 t 的变化曲线。

    Parameters
    ----------
    results : dict
        嵌套字典结构:
        {
            "S_cos(M_gen, M_mem)": {"t": np.array, "mean": np.array, "std": np.array},
            "S_cos(model, empirical)": {"t": ..., "mean": ..., "std": ...},
            "S_cos(train, test)": {"t": ..., "mean": ..., "std": ...},
        }
    title : str
    save_path : str | Path, optional

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    line_styles = ["-", "--", "-."]
    colors = list(PHASE_COLORS.values())

    for i, (label, data) in enumerate(results.items()):
        t = data["t"]
        mean = data["mean"]
        std = data.get("std", np.zeros_like(mean))

        color = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]

        ax.plot(t, mean, label=label, color=color, linestyle=ls)
        ax.fill_between(t, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Diffusion Time $t$")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc="best")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_memory_ratio_curve(
    t_values: np.ndarray,
    f_mem_values: np.ndarray,
    dimension: int,
    threshold: float = 1 / 3,
    model_label: str = "model",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    绘制动态记忆比例 f_mem(t) 曲线。

    Parameters
    ----------
    t_values : np.ndarray, shape (T,)
    f_mem_values : np.ndarray, shape (T,)
    dimension : int
        环境维度 d。
    threshold : float
        R_t 判定阈值（标注在图上）。
    model_label : str
    save_path : str | Path, optional

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    color = DIMENSION_COLORS.get(dimension, "#9467bd")
    ax.plot(t_values, f_mem_values, color=color, marker="o", markersize=3,
            label=f"$d={dimension}$ ({model_label})")
    ax.fill_between(t_values, 0, f_mem_values, alpha=0.1, color=color)

    ax.set_xlabel("Diffusion Time $t$")
    ax.set_ylabel("$f_{mem}(t)$")
    ax.set_title(f"Dynamic Memory Ratio ($d={dimension}$, threshold=$R_t < {threshold:.3f}$)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_cross_dimension_comparison(
    dimension_results: Dict[int, Dict[str, np.ndarray]],
    metric_name: str = "S_cos(M_gen, M_mem)",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    跨维度对比图：同一指标在不同 d 下的曲线叠加。

    Parameters
    ----------
    dimension_results : dict
        {
            4096:  {"t": np.array, "mean": np.array, "std": np.array},
            16384: {"t": ..., "mean": ..., "std": ...},
        }
    metric_name : str
    save_path : str | Path, optional

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for dim in sorted(dimension_results.keys()):
        data = dimension_results[dim]
        t = data["t"]
        mean = data["mean"]
        std = data.get("std", np.zeros_like(mean))

        color = DIMENSION_COLORS.get(dim, None)
        ratio = dim / 1024  # d/m 比率
        label = f"$d={dim}$ ($d/m={ratio:.0f}$)"

        ax.plot(t, mean, label=label, color=color)
        ax.fill_between(t, mean - std, mean + std, alpha=0.12, color=color)

    ax.set_xlabel("Diffusion Time $t$")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} — Cross-Dimension Comparison")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_phase_transition_diagram(
    dimensions: List[int],
    gen_steps: List[int],
    mem_steps: List[int],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    相变图：以 d/m 为横轴，τ_gen 和 τ_mem 为纵轴。

    Parameters
    ----------
    dimensions : list[int]
        环境维度列表。
    gen_steps : list[int]
        对应 τ_gen。
    mem_steps : list[int]
        对应 τ_mem。
    save_path : str | Path, optional

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    m = 1024  # 固定有效维度
    ratios = [d / m for d in dimensions]

    ax.plot(ratios, gen_steps, "o-", color=PHASE_COLORS["gen"],
            label=r"$\tau_{gen}$ (Generalization)", markersize=8)
    ax.plot(ratios, mem_steps, "s-", color=PHASE_COLORS["mem"],
            label=r"$\tau_{mem}$ (Memorization)", markersize=8)

    # 填充相变区间
    ax.fill_between(ratios, gen_steps, mem_steps, alpha=0.1, color="purple",
                    label="Phase Transition Gap")

    ax.set_xlabel("$d/m$ (Ambient / Intrinsic Dimension Ratio)")
    ax.set_ylabel("Training Steps $\\tau$")
    ax.set_title("Phase Transition Diagram")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_trajectory_snapshots(
    trajectories: Dict[float, torch.Tensor],
    num_samples: int = 8,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    可视化反向采样轨迹的中间态快照。

    Parameters
    ----------
    trajectories : dict
        {t_value: tensor of shape (B, 1, H, W), ...}，按 t 降序排列。
    num_samples : int
        展示的样本数量。
    save_path : str | Path, optional

    Returns
    -------
    plt.Figure
    """
    timesteps = sorted(trajectories.keys(), reverse=True)
    n_cols = len(timesteps)
    n_rows = min(num_samples, trajectories[timesteps[0]].shape[0])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, t in enumerate(timesteps):
        axes[0, col].set_title(f"$t={t:.2f}$", fontsize=10)
        images = trajectories[t].detach().cpu()
        for row in range(n_rows):
            ax = axes[row, col]
            img = images[row, 0].numpy()  # (H, W)
            # 归一化显示
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

    fig.suptitle("Reverse Diffusion Trajectory", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_fid_trajectory(
    t_values: np.ndarray,
    fid_values: np.ndarray,
    gen_step: Optional[int] = None,
    mem_step: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    绘制轨迹中间态 FID 曲线。

    Parameters
    ----------
    t_values : np.ndarray
    fid_values : np.ndarray
    gen_step : int, optional
        标注 M_gen 对应的 FID 值。
    mem_step : int, optional
        标注 M_mem 对应的 FID 值。
    save_path : str | Path, optional

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(t_values, fid_values, "o-", color="#7B1FA2", markersize=4,
            label="Trajectory FID")

    # 标注最小 FID 点
    min_idx = np.argmin(fid_values)
    ax.annotate(
        f"min FID={fid_values[min_idx]:.1f}",
        xy=(t_values[min_idx], fid_values[min_idx]),
        xytext=(t_values[min_idx] + 0.05, fid_values[min_idx] + 10),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )

    ax.set_xlabel("Diffusion Time $t$")
    ax.set_ylabel("FID Score")
    ax.set_title("Trajectory Mid-State FID")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_reconstruction_gap(
    timesteps: np.ndarray,
    mse_train: np.ndarray,
    mse_test: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    绘制训练集/测试集重建误差对比曲线。

    Parameters
    ----------
    timesteps : np.ndarray, shape (T,)
    mse_train : np.ndarray, shape (T,)
    mse_test : np.ndarray, shape (T,)
    save_path : str | Path, optional

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：绝对 MSE
    ax = axes[0]
    ax.plot(timesteps, mse_train, "o-", color=PHASE_COLORS["gen"],
            label="Train MSE", markersize=4)
    ax.plot(timesteps, mse_test, "s-", color=PHASE_COLORS["mem"],
            label="Test MSE", markersize=4)
    ax.fill_between(timesteps, mse_train, mse_test, alpha=0.15, color="purple")
    ax.set_xlabel("Diffusion Time $t$")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction Error")
    ax.legend(loc="best")

    # 右图：相对差距 (gap)
    ax = axes[1]
    gap = (mse_test - mse_train) / (mse_train + 1e-8)
    ax.plot(timesteps, gap, "D-", color="#FF9800", markersize=4)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Diffusion Time $t$")
    ax.set_ylabel("Relative Gap $(MSE_{test} - MSE_{train}) / MSE_{train}$")
    ax.set_title("Generalization Gap")

    fig.suptitle("Reconstruction Gap Analysis", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def figure_to_tensor(fig: plt.Figure) -> torch.Tensor:
    """
    将 matplotlib Figure 转换为 PyTorch 张量，用于传给日志后端。

    Parameters
    ----------
    fig : plt.Figure

    Returns
    -------
    torch.Tensor, shape (3, H, W), dtype float32, 值域 [0, 1]
    """
    fig.canvas.draw()
    # 从 canvas 获取 RGBA 像素
    buf = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    # (H, W, 3) -> (3, H, W), [0, 255] -> [0, 1]
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    plt.close(fig)
    return tensor
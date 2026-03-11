"""
Toy Demo: 2D 圆环上的扩散过程可视化
=====================================
1. 在单位圆上采样 → 正向加噪 → 反向去噪
2. 全程可视化，直观展示扩散模型如何学习流形结构
"""

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ================================================================== #
#  1. 数据：2D 圆环
# ================================================================== #

def sample_circle(n: int, noise_std: float = 0.05, seed: int = 42) -> torch.Tensor:
    """在单位圆上均匀采样，加微小噪声模拟流形。"""
    rng = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, generator=rng) * 2 * math.pi
    x = torch.stack([theta.cos(), theta.sin()], dim=1)  # (N, 2)
    x += torch.randn_like(x) * noise_std
    return x


# ================================================================== #
#  2. 简单 VP 调度器（2D 版本）
# ================================================================== #

class SimpleVPScheduler:
    """最小 VP-SDE 调度器。"""

    def __init__(self, num_steps: int = 1000, beta_min: float = 0.1, beta_max: float = 20.0):
        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        # 离散化
        self.dt = 1.0 / num_steps
        t = torch.linspace(self.dt, 1.0, num_steps)
        betas = beta_min + (beta_max - beta_min) * t
        log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
        self.alpha_t = torch.exp(2.0 * log_mean_coeff)   # α(t)^2 = exp(...)
        self.sigma_t = torch.sqrt(1.0 - self.alpha_t)     # σ(t)

    def add_noise(self, x0: torch.Tensor, t_idx: torch.Tensor,
                  noise: torch.Tensor = None) -> tuple:
        """x_t = sqrt(α_t) * x0 + σ_t * ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha = self.alpha_t[t_idx].sqrt().unsqueeze(-1)  # (B, 1)
        sigma = self.sigma_t[t_idx].unsqueeze(-1)
        x_t = alpha * x0 + sigma * noise
        return x_t, noise

    def get_sigma(self, t_idx: torch.Tensor) -> torch.Tensor:
        return self.sigma_t[t_idx]


# ================================================================== #
#  3. 简单 MLP 得分网络
# ================================================================== #

class ToyScoreNet(nn.Module):
    """2D 数据的得分网络：输入 (x, t) → 预测噪声 ε。"""

    def __init__(self, data_dim: int = 2, hidden: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, 2), t: (B,) int timestep → (B, 2) predicted noise"""
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 1000.0)
        return self.net(torch.cat([x, t_emb], dim=-1))


# ================================================================== #
#  4. 训练
# ================================================================== #

def train(
    model: ToyScoreNet,
    scheduler: SimpleVPScheduler,
    data: torch.Tensor,
    num_epochs: int = 300,
    batch_size: int = 512,
    lr: float = 3e-4,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    N = len(data)
    losses = []

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            x0 = data[perm[i:i + batch_size]]
            B = len(x0)

            # 随机时间步
            t_idx = torch.randint(0, scheduler.num_steps, (B,))
            noise = torch.randn_like(x0)
            x_t, _ = scheduler.add_noise(x0, t_idx, noise)

            # 预测噪声
            pred_noise = model(x_t, t_idx)
            loss = ((pred_noise - noise) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if epoch % 50 == 0:
            pbar.set_postfix(loss=f"{avg_loss:.6f}")

    return losses


# ================================================================== #
#  5. 反向采样（DDPM 式）
# ================================================================== #

@torch.no_grad()
def sample_reverse(
    model: ToyScoreNet,
    scheduler: SimpleVPScheduler,
    num_samples: int = 1000,
    record_steps: list = None,
) -> dict:
    """
    反向去噪采样，记录中间轨迹。

    Returns
    -------
    dict: {step_idx: (N, 2) tensor}
    """
    if record_steps is None:
        record_steps = list(range(0, scheduler.num_steps, scheduler.num_steps // 20))

    x = torch.randn(num_samples, 2)  # 从纯噪声开始
    trajectories = {}

    for i in reversed(range(scheduler.num_steps)):
        t_idx = torch.full((num_samples,), i, dtype=torch.long)
        pred_noise = model(x, t_idx)

        # DDPM 更新
        alpha_t = scheduler.alpha_t[i]
        sigma_t = scheduler.sigma_t[i]
        alpha_prev = scheduler.alpha_t[i - 1] if i > 0 else torch.tensor(1.0)

        # x_{t-1} 预测
        x0_pred = (x - sigma_t * pred_noise) / alpha_t.sqrt().clamp(min=1e-8)
        # 简化的 DDPM 步（确定性 DDIM 风格）
        sigma_prev = scheduler.sigma_t[i - 1] if i > 0 else torch.tensor(0.0)
        x = alpha_prev.sqrt() * x0_pred + sigma_prev * pred_noise

        if i > 0 and i < scheduler.num_steps - 1:
            # 加一点随机性（标准 DDPM）
            noise_scale = 0.2 * sigma_prev
            x += noise_scale * torch.randn_like(x)

        if i in record_steps or i == 0:
            trajectories[i] = x.clone()

    return trajectories


# ================================================================== #
#  6. 可视化
# ================================================================== #

def plot_forward_process(data: torch.Tensor, scheduler: SimpleVPScheduler,
                         save_dir: Path):
    """可视化正向加噪过程。"""
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    steps = [0, 50, 200, 500, 800, 999]

    for ax, step in zip(axes, steps):
        t_idx = torch.full((len(data),), step, dtype=torch.long)
        x_t, _ = scheduler.add_noise(data, t_idx)
        x_np = x_t.numpy()

        ax.scatter(x_np[:, 0], x_np[:, 1], s=1, alpha=0.5, c="steelblue")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(f"$t={step}$\n$\\sigma={scheduler.sigma_t[step]:.2f}$", fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Forward Process: Circle → Noise", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "forward_process.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 保存: {save_dir / 'forward_process.png'}")


def plot_reverse_process(trajectories: dict, data: torch.Tensor,
                         scheduler: SimpleVPScheduler, save_dir: Path):
    """可视化反向去噪过程。"""
    steps = sorted(trajectories.keys(), reverse=True)
    # 选 8 个代表性步骤
    if len(steps) > 8:
        indices = np.linspace(0, len(steps) - 1, 8, dtype=int)
        steps = [steps[i] for i in indices]

    fig, axes = plt.subplots(1, len(steps), figsize=(3.5 * len(steps), 3.5))
    if len(steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, steps):
        x_np = trajectories[step].numpy()
        # 背景：真实数据
        real_np = data.numpy()
        ax.scatter(real_np[:, 0], real_np[:, 1], s=2, alpha=0.15, c="lightcoral",
                   label="Real" if step == steps[0] else None)
        # 当前状态
        ax.scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.6, c="steelblue")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(f"$t={step}$", fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Reverse Process: Noise → Circle", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "reverse_process.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 保存: {save_dir / 'reverse_process.png'}")


def plot_score_field(model: ToyScoreNet, scheduler: SimpleVPScheduler,
                     data: torch.Tensor, save_dir: Path):
    """可视化不同时间步的得分向量场。"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    steps = [10, 100, 400, 800]

    for ax, step in zip(axes, steps):
        # 网格
        lim = 2.5 if step < 200 else 3.5
        gx = np.linspace(-lim, lim, 25)
        gy = np.linspace(-lim, lim, 25)
        xx, yy = np.meshgrid(gx, gy)
        grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
        t_idx = torch.full((len(grid),), step, dtype=torch.long)

        with torch.no_grad():
            pred_noise = model(grid, t_idx)
            sigma = scheduler.get_sigma(t_idx[:1])
            # score ≈ -noise / sigma
            score = -pred_noise / sigma.clamp(min=1e-4)

        score_np = score.numpy()
        grid_np = grid.numpy()

        # 背景：真实数据加噪后
        t_idx_data = torch.full((len(data),), step, dtype=torch.long)
        x_t, _ = scheduler.add_noise(data, t_idx_data)
        ax.scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), s=1, alpha=0.2, c="lightcoral")

        # 得分向量场
        magnitude = np.sqrt(score_np[:, 0] ** 2 + score_np[:, 1] ** 2)
        ax.quiver(grid_np[:, 0], grid_np[:, 1],
                  score_np[:, 0], score_np[:, 1],
                  magnitude, cmap="viridis", alpha=0.7, scale=80)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"Score Field at $t={step}$\n$\\sigma={scheduler.sigma_t[step]:.2f}$",
                     fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Learned Score Field $\\nabla_x \\log p_t(x)$", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "score_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 保存: {save_dir / 'score_field.png'}")


def plot_loss_curve(losses: list, save_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"保存: {save_dir / 'loss_curve.png'}")


def plot_generation_quality(data: torch.Tensor, generated: torch.Tensor, save_dir: Path):
    """对比真实数据与生成数据的分布。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    real = data.numpy()
    fake = generated.numpy()

    # 左：真实数据
    axes[0].scatter(real[:, 0], real[:, 1], s=2, alpha=0.5, c="lightcoral")
    axes[0].set_title("Real Data (Circle)", fontsize=12)

    # 中：生成数据
    axes[1].scatter(fake[:, 0], fake[:, 1], s=2, alpha=0.5, c="steelblue")
    axes[1].set_title("Generated Data", fontsize=12)

    # 右：叠加对比
    axes[2].scatter(real[:, 0], real[:, 1], s=2, alpha=0.3, c="lightcoral", label="Real")
    axes[2].scatter(fake[:, 0], fake[:, 1], s=2, alpha=0.3, c="steelblue", label="Generated")
    axes[2].legend(fontsize=10, markerscale=5)
    axes[2].set_title("Overlay", fontsize=12)

    for ax in axes:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # 统计信息
    real_radius = np.sqrt((real ** 2).sum(axis=1))
    fake_radius = np.sqrt((fake ** 2).sum(axis=1))
    fig.suptitle(
        f"Generation Quality — "
        f"Real radius: {real_radius.mean():.3f}±{real_radius.std():.3f}, "
        f"Generated radius: {fake_radius.mean():.3f}±{fake_radius.std():.3f}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_dir / "generation_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {save_dir / 'generation_quality.png'}")


def plot_embedding_dimension_experiment(save_dir: Path):
    """
    核心实验：固定圆环（m=2），零填充到不同环境维度 d，
    观察 d/m 比率对学习质量的影响。
    """
    dims = [2, 8, 32, 128]
    n_train = 2000
    num_epochs = 200
    num_steps = 500

    fig, axes = plt.subplots(2, len(dims), figsize=(5 * len(dims), 10))

    for col, d in enumerate(dims):
        print(f"\n  🔬 环境维度 d={d}, d/m={d/2:.0f}")

        # 生成 2D 数据并零填充到 d 维
        data_2d = sample_circle(n_train, noise_std=0.05, seed=42)
        data_d = torch.zeros(n_train, d)
        data_d[:, :2] = data_2d  # 前 2 维放真实数据，其余为零

        # 用 d 维 MLP 训练
        model = ToyScoreNetND(data_dim=d, hidden=256)
        scheduler = SimpleVPScheduler(num_steps=num_steps, beta_min=0.1, beta_max=20.0)

        losses = train_nd(model, scheduler, data_d, num_epochs=num_epochs, batch_size=512)

        # 采样
        generated_d = sample_reverse_nd(model, scheduler, num_samples=1000, data_dim=d)
        generated_2d = generated_d[:, :2].numpy()

        # 上排：生成结果（前 2 维投影）
        ax = axes[0, col]
        real_2d = data_2d.numpy()
        ax.scatter(real_2d[:, 0], real_2d[:, 1], s=2, alpha=0.2, c="lightcoral", label="Real")
        ax.scatter(generated_2d[:, 0], generated_2d[:, 1], s=2, alpha=0.4, c="steelblue", label="Gen")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title(f"$d={d}$, $d/m={d/2:.0f}$", fontsize=13)
        ax.grid(True, alpha=0.2)
        if col == 0:
            ax.set_ylabel("Generation (2D proj)", fontsize=11)
            ax.legend(fontsize=8, markerscale=5)

        # 下排：损失曲线
        ax = axes[1, col]
        ax.plot(losses, color="steelblue")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(f"Loss (d={d})", fontsize=11)
        ax.grid(True, alpha=0.2)
        if col == 0:
            ax.set_ylabel("MSE Loss", fontsize=11)

        # 质量统计
        gen_radius = np.sqrt((generated_2d ** 2).sum(axis=1))
        residual_norm = np.sqrt((generated_d[:, 2:].numpy() ** 2).sum(axis=1))
        print(f"    生成半径: {gen_radius.mean():.3f}±{gen_radius.std():.3f}")
        print(f"    填充维度残差: {residual_norm.mean():.4f}±{residual_norm.std():.4f}")

    fig.suptitle(
        "Ambient Dimension Experiment: Circle ($m=2$) embedded in $\\mathbb{R}^d$",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_dir / "dimension_experiment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✅ 保存: {save_dir / 'dimension_experiment.png'}")


# ================================================================== #
#  7. 高维版本的模型和训练（用于维度实验）
# ================================================================== #

class ToyScoreNetND(nn.Module):
    """任意维度的得分网络。"""

    def __init__(self, data_dim: int = 2, hidden: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t.float().unsqueeze(-1) / 1000.0)
        return self.net(torch.cat([x, t_emb], dim=-1))


def train_nd(model, scheduler, data, num_epochs=200, batch_size=512, lr=3e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    N = len(data)
    losses = []

    for epoch in range(num_epochs):
        perm = torch.randperm(N)
        epoch_loss, n_b = 0.0, 0
        for i in range(0, N, batch_size):
            x0 = data[perm[i:i + batch_size]]
            B = len(x0)
            t_idx = torch.randint(0, scheduler.num_steps, (B,))
            noise = torch.randn_like(x0)
            x_t, _ = scheduler.add_noise(x0, t_idx, noise)
            loss = ((model(x_t, t_idx) - noise) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        losses.append(epoch_loss / n_b)
    return losses


@torch.no_grad()
def sample_reverse_nd(model, scheduler, num_samples=1000, data_dim=2):
    x = torch.randn(num_samples, data_dim)
    for i in reversed(range(scheduler.num_steps)):
        t_idx = torch.full((num_samples,), i, dtype=torch.long)
        pred_noise = model(x, t_idx)
        alpha_t = scheduler.alpha_t[i]
        sigma_t = scheduler.sigma_t[i]
        alpha_prev = scheduler.alpha_t[i - 1] if i > 0 else torch.tensor(1.0)
        sigma_prev = scheduler.sigma_t[i - 1] if i > 0 else torch.tensor(0.0)
        x0_pred = (x - sigma_t * pred_noise) / alpha_t.sqrt().clamp(min=1e-8)
        x = alpha_prev.sqrt() * x0_pred + sigma_prev * pred_noise
        if i > 0:
            x += 0.1 * sigma_prev * torch.randn_like(x)
    return x


# ================================================================== #
#  8. 主函数
# ================================================================== #

def main():
    save_dir = Path("outputs/toy_circle_demo")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔵 Toy Circle Diffusion Demo")
    print("=" * 60)

    # ---- 数据 ----
    print("\n[1/6] 生成圆环数据...")
    data = sample_circle(5000, noise_std=0.05, seed=42)
    print(f"  数据形状: {data.shape}, 半径均值: {data.norm(dim=1).mean():.3f}")

    # ---- 调度器 ----
    scheduler = SimpleVPScheduler(num_steps=1000, beta_min=0.1, beta_max=20.0)

    # ---- 正向过程可视化 ----
    print("\n[2/6] 可视化正向加噪过程...")
    plot_forward_process(data, scheduler, save_dir)

    # ---- 训练 ----
    print("\n[3/6] 训练得分网络...")
    model = ToyScoreNet(data_dim=2, hidden=256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")

    losses = train(model, scheduler, data, num_epochs=300, batch_size=512)
    plot_loss_curve(losses, save_dir)
    print(f"  最终损失: {losses[-1]:.6f}")

    # ---- 得分场可视化 ----
    print("\n[4/6] 可视化得分向量场...")
    plot_score_field(model, scheduler, data, save_dir)

    # ---- 反向采样 ----
    print("\n[5/6] 反向采样...")
    record_steps = list(range(0, 1000, 50)) + [999]
    trajectories = sample_reverse(model, scheduler, num_samples=2000,
                                  record_steps=record_steps)
    plot_reverse_process(trajectories, data, scheduler, save_dir)

    # 生成质量
    final_samples = trajectories[0]
    plot_generation_quality(data, final_samples, save_dir)

    # ---- 维度实验 ----
    print("\n[6/6] 环境维度实验 (d/m 比率)...")
    plot_embedding_dimension_experiment(save_dir)

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print(f"所有结果已保存至: {save_dir.resolve()}")
    print("forward_process.png      — 正向加噪过程")
    print("loss_curve.png            — 训练损失曲线")
    print("score_field.png           — 得分向量场")
    print("reverse_process.png       — 反向去噪过程")
    print("generation_quality.png    — 生成质量对比")
    print("dimension_experiment.png  — d/m 维度比率实验")
    print("=" * 60)


if __name__ == "__main__":
    main()
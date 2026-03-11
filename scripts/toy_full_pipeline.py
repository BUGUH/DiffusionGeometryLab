"""
合成数据（圆环）端到端管线测试
===============================
用最小参数走通: 数据→训练→采样→全部指标→可视化

用法:
    cd /data/zhouzhangchen/bzw/#Sun_Jinghuan/diffusion
    python scripts/toy_full_pipeline.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils.config import load_config, validate_config, pretty_print_config
from src.utils.seed import set_global_seed
from src.utils.device import DeviceManager
from src.data.dataset import build_dataloaders
from src.data.semantic_pairs import build_semantic_pairs
from src.data.zero_padding import ZeroPadder
from src.models.score_network import ScoreNetwork
from src.schedulers import build_scheduler
from src.training.trainer import DiffusionTrainer
from src.training.checkpoint_manager import CheckpointManager
from src.sampling.reverse_sampler import ReverseSampler
from src.metrics.cosine_field import CosineFieldEvaluator
from src.metrics.fid_evaluator import FIDEvaluator
from src.metrics.memory_ratio import MemoryRatioEvaluator
from src.metrics.reconstruction_gap import ReconstructionGapEvaluator
from src.metrics.lpips_ssim import LPIPSSSIMEvaluator


def main():
    print("=" * 70)
    print("🔵 DiffusionGeometryLab — 合成数据端到端测试")
    print("=" * 70)

    # ================================================================
    # 1. 配置
    # ================================================================
    print("\n[1/8] 加载配置...")
    cfg = load_config("configs/toy_test.yaml")
    cfg = validate_config(cfg)

    save_dir = Path(cfg.experiment.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  输出目录: {save_dir}")

    set_global_seed(cfg.experiment.seed)
    dm = DeviceManager(cfg.device)
    print(f"  设备: {dm.device}")

    # ================================================================
    # 2. 数据
    # ================================================================
    print("\n[2/8] 构建合成数据 (圆环)...")
    train_loader, test_loader, padder = build_dataloaders(cfg)
    train_data = train_loader.dataset.data
    test_data = test_loader.dataset.data
    print(f"  训练集: {train_data.shape}, 测试集: {test_data.shape}")
    print(f"  m={padder.m}, d={padder.d}, d/m={padder.d/padder.m:.1f}")

    # 加载原始 2D 点用于可视化
    raw_points_path = save_dir / "raw_points.pt"
    raw_points = torch.load(raw_points_path) if raw_points_path.exists() else None

    # 可视化原始数据
    _plot_synthetic_data(train_data, raw_points, padder, save_dir)

    # ================================================================
    # 3. 模型 + 调度器
    # ================================================================
    print("\n[3/8] 初始化模型和调度器...")
    model = ScoreNetwork(cfg)
    scheduler = build_scheduler(cfg)
    print(f"  调度器: {cfg.scheduler.type}, steps={cfg.scheduler.num_diffusion_steps}")

    # ================================================================
    # 4. 训练（自动相变检测）
    # ================================================================
    print("\n[4/8] 开始训练...")
    trainer = DiffusionTrainer(
        cfg=cfg, model=model, scheduler=scheduler,
        train_loader=train_loader, test_loader=test_loader,
        device_manager=dm,
    )
    summary = trainer.train()
    print(f"\n  训练摘要: {summary}")

    # ================================================================
    # 5. 采样 + 轨迹
    # ================================================================
    print("\n[5/8] 反向采样...")
    model.eval()
    model.to(dm.device)

    sampler = ReverseSampler(
        model=model, scheduler=scheduler,
        method="ddim", num_steps=50,
        capture_timesteps=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    sample_shape = (256,) + tuple(train_data.shape[1:])
    generated, bundle = sampler.sample(sample_shape, dm.device)
    print(f"  生成样本: {generated.shape}")
    print(f"  轨迹时间步: {sorted(bundle.x_ts.keys())}")

    _plot_trajectory(bundle, padder, save_dir)

    # ================================================================
    # 6. 核心指标计算
    # ================================================================
    print("\n[6/8] 计算核心指标...")

    # 6a. FID
    print("  [6a] FID...")
    fid_eval = FIDEvaluator()
    fid = fid_eval.compute_fid(train_data[:500], generated.cpu()[:500])
    print(f"    标准 FID = {fid:.2f}")

    traj_fid = fid_eval.compute_trajectory_fid(train_data, bundle, scheduler)
    print(f"    轨迹 FID: { {f't={k:.2f}': f'{v:.1f}' for k, v in sorted(traj_fid.items())} }")

    # 6b. 重建差距
    print("  [6b] 重建差距...")
    recon_eval = ReconstructionGapEvaluator(
        scheduler=scheduler,
        timesteps=[0.1, 0.3, 0.5, 0.7, 0.9],
        num_samples=200,
    )
    recon = recon_eval.evaluate(model, train_loader, test_loader, dm.device)
    print(f"    Train MSE: {recon['train_mse']}")
    print(f"    Test MSE:  {recon['test_mse']}")
    print(f"    Gap:       {recon['gap']}")

    _plot_reconstruction_gap(recon, save_dir)

    # 6c. 余弦场（用同一模型模拟 gen 和 mem——真实场景需两个 checkpoint）
    print("  [6c] 余弦相似度场...")
    cosine_eval = CosineFieldEvaluator(
        scheduler=scheduler, num_timesteps=20, batch_size=128,
    )

    # model vs empirical
    cos_emp = cosine_eval.compute_model_empirical_similarity(
        model, train_data, dm.device,
    )
    print(f"    S_cos(model, empirical): mean={cos_emp['mean'].mean():.4f}")

    # train vs test
    pairs = build_semantic_pairs(train_data, test_data, num_pairs=200)
    cos_tt = cosine_eval.compute_train_test_similarity(
        model, train_data, test_data, pairs, dm.device,
    )
    print(f"    S_cos(train, test): mean={cos_tt['mean'].mean():.4f}")

    _plot_cosine_fields(cos_emp, cos_tt, save_dir)

    # 6d. 记忆比例
    print("  [6d] 动态记忆比例 f_mem(t)...")
    try:
        mem_eval = MemoryRatioEvaluator(
            scheduler=scheduler, threshold=1.0/3.0,
            num_timesteps=10, nprobe=8,
        )
        mem_result = mem_eval.compute(train_data, bundle)
        print(f"    f_mem: {mem_result['f_mem']}")
        print(f"    mean_ratio: {mem_result['mean_ratio']}")
        _plot_memory_ratio(mem_result, save_dir)
    except Exception as e:
        print(f"    ⚠️ 记忆比例计算跳过: {e}")

    # 6e. LPIPS/SSIM
    print("  [6e] LPIPS/SSIM...")
    try:
        lpips_eval = LPIPSSSIMEvaluator(device=dm.device)
        lpips_res = lpips_eval.evaluate(generated.cpu(), train_data, num_samples=100)
        print(f"    LPIPS: {lpips_res['lpips_mean']:.4f} ± {lpips_res['lpips_std']:.4f}")
        print(f"    SSIM:  {lpips_res['ssim_mean']:.4f} ± {lpips_res['ssim_std']:.4f}")
    except Exception as e:
        print(f"    ⚠️ LPIPS/SSIM 跳过: {e}")

    # ================================================================
    # 7. 多维度对比（快速版）
    # ================================================================
    print("\n[7/8] 多维度快速对比...")
    _dimension_comparison(cfg, dm, save_dir)

    # ================================================================
    # 8. 汇总
    # ================================================================
    print("\n[8/8] 生成汇总报告...")
    _generate_summary(summary, fid, recon, cos_emp, cos_tt, save_dir)

    print("\n" + "=" * 70)
    print(f"🎉 端到端测试完成！所有结果保存在: {save_dir.resolve()}")
    print("=" * 70)
    _list_outputs(save_dir)


# ================================================================
# 辅助绘图函数
# ================================================================

def _plot_synthetic_data(train_data, raw_points, padder, save_dir):
    """可视化合成数据。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始 2D 点
    if raw_points is not None:
        pts = raw_points.numpy()
        axes[0].scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5, c="steelblue")
        axes[0].set_aspect("equal")
        axes[0].set_title("Raw 2D Points (Circle)")
        axes[0].grid(True, alpha=0.2)

    # 编码后的图像样例
    for i in range(min(8, len(train_data))):
        if i < 4:
            ax = axes[1]
        else:
            ax = axes[2]
        # 取原始区域
        img = padder.unpad(train_data[i:i+1])[0, 0].numpy()
        row, col = i % 4, 0
        if i < 4:
            axes[1].set_title(f"Encoded Images (padded {padder.source_size}→{padder.target_size})")
        else:
            axes[2].set_title("More samples")

    # 直接显示几张
    axes[1].imshow(_make_grid(train_data[:4], padder), cmap="viridis")
    axes[1].set_title(f"Train Images ({padder.target_size}×{padder.target_size})")
    axes[1].axis("off")

    axes[2].imshow(_make_grid(train_data[4:8], padder), cmap="viridis")
    axes[2].set_title("More Train Samples")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(save_dir / "01_synthetic_data.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: 01_synthetic_data.png")


def _make_grid(images, padder, nrow=2):
    """简单图像网格。"""
    imgs = []
    for i in range(min(len(images), 4)):
        img = images[i, 0].numpy()
        imgs.append(img)

    # 2x2 网格
    rows = []
    for r in range(0, len(imgs), nrow):
        row_imgs = imgs[r:r+nrow]
        while len(row_imgs) < nrow:
            row_imgs.append(np.zeros_like(imgs[0]))
        rows.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(rows, axis=0)


def _plot_trajectory(bundle, padder, save_dir):
    """可视化采样轨迹。"""
    ts = sorted(bundle.x_ts.keys(), reverse=True)
    # 选 6 个代表时间步
    if len(ts) > 6:
        indices = np.linspace(0, len(ts) - 1, 6, dtype=int)
        ts = [ts[i] for i in indices]

    fig, axes = plt.subplots(2, len(ts), figsize=(3 * len(ts), 6))

    for col, t in enumerate(ts):
        x_t = bundle.x_ts[t]
        # 上排：填充图像
        img = x_t[0, 0].numpy()
        axes[0, col].imshow(img, cmap="viridis", vmin=-1, vmax=1)
        axes[0, col].set_title(f"t={t:.2f}", fontsize=10)
        axes[0, col].axis("off")

        # 下排：unpad 后
        unpadded = padder.unpad(x_t[:1])[0, 0].numpy()
        axes[1, col].imshow(unpadded, cmap="viridis", vmin=-1, vmax=1)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Padded", fontsize=10)
    axes[1, 0].set_ylabel("Unpadded", fontsize=10)
    fig.suptitle("Reverse Sampling Trajectory", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "02_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: 02_trajectory.png")


def _plot_reconstruction_gap(recon, save_dir):
    """重建差距图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ts = recon["timesteps"]
    axes[0].plot(ts, recon["train_mse"], "o-", color="#2196F3", label="Train MSE", markersize=6)
    axes[0].plot(ts, recon["test_mse"], "s-", color="#F44336", label="Test MSE", markersize=6)
    axes[0].fill_between(ts, recon["train_mse"], recon["test_mse"], alpha=0.15, color="purple")
    axes[0].set_xlabel("Diffusion Time t")
    axes[0].set_ylabel("Reconstruction MSE")
    axes[0].set_title("Reconstruction Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts, recon["gap"], "D-", color="#FF9800", markersize=6)
    axes[1].axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    axes[1].set_xlabel("Diffusion Time t")
    axes[1].set_ylabel("Relative Gap")
    axes[1].set_title("Generalization Gap (test-train)/train")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "03_reconstruction_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 保存: 03_reconstruction_gap.png")


def _plot_cosine_fields(cos_emp, cos_tt, save_dir):
    """余弦场图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(cos_emp["t"], cos_emp["mean"], "o-", color="#4CAF50", markersize=3)
    axes[0].fill_between(cos_emp["t"],
                         cos_emp["mean"] - cos_emp["std"],
                         cos_emp["mean"] + cos_emp["std"],
                         alpha=0.2, color="#4CAF50")
    axes[0].set_xlabel("Diffusion Time t")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("$S_{cos}$(model, empirical)")
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cos_tt["t"], cos_tt["mean"], "o-", color="#9C27B0", markersize=3)
    axes[1].fill_between(cos_tt["t"],
                         cos_tt["mean"] - cos_tt["std"],
                         cos_tt["mean"] + cos_tt["std"],
                         alpha=0.2, color="#9C27B0")
    axes[1].set_xlabel("Diffusion Time t")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("$S_{cos}$(train, test)")
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_dir / "04_cosine_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 保存: 04_cosine_fields.png")


def _plot_memory_ratio(mem_result, save_dir):
    """记忆比例图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mem_result["t"], mem_result["f_mem"], "o-", color="#E91E63", markersize=5)
    ax.fill_between(mem_result["t"], 0, mem_result["f_mem"], alpha=0.15, color="#E91E63")
    ax.set_xlabel("Diffusion Time t")
    ax.set_ylabel("$f_{mem}(t)$")
    ax.set_title("Dynamic Memory Ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "05_memory_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 保存: 05_memory_ratio.png")


def _dimension_comparison(cfg, dm, save_dir):
    """快速多维度对比：d=64, 256, 1024 下训练 500 步。"""
    from omegaconf import OmegaConf
    from src.data.dataset import build_dataloaders
    from src.models.score_network import ScoreNetwork
    from src.schedulers import build_scheduler
    from src.training.trainer import DiffusionTrainer

    dims = [64, 256]  # 8×8, 16×16
    results = {}

    for d in dims:
        res = int(math.sqrt(d))
        print(f"  维度 d={d} (res={res}×{res}, d/m={d/64:.1f})...")

        # 克隆并修改配置
        cfg_d = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_d.data.target_dim = d
        cfg_d.data.target_resolution = res
        cfg_d.training.max_steps = 500
        cfg_d.training.eval_every_steps = 250
        cfg_d.training.log_every_steps = 250
        cfg_d.training.save_every_steps = 500
        cfg_d.training.phase_detection.enabled = False
        cfg_d.experiment.name = f"dim_compare_d{d}"
        cfg_d.experiment.output_dir = str(save_dir / f"dim_d{d}")

        try:
            set_global_seed(42)
            train_ld, test_ld, pad = build_dataloaders(cfg_d)
            model_d = ScoreNetwork(cfg_d)
            sched_d = build_scheduler(cfg_d)

            trainer_d = DiffusionTrainer(
                cfg=cfg_d, model=model_d, scheduler=sched_d,
                train_loader=train_ld, test_loader=test_ld,
                device_manager=dm,
            )
            summary_d = trainer_d.train()

            # 采样
            model_d.eval()
            sampler_d = ReverseSampler(
                model=model_d, scheduler=sched_d,
                method="ddim", num_steps=30,
            )
            shape_d = (128,) + tuple(train_ld.dataset.data.shape[1:])
            gen_d, _ = sampler_d.sample(shape_d, dm.device)

            # 简单 FID
            fid_eval = FIDEvaluator()
            fid_d = fid_eval.compute_fid(train_ld.dataset.data[:256], gen_d.cpu()[:256])

            results[d] = {"fid": fid_d, "summary": summary_d}
            print(f"    d={d}: FID={fid_d:.2f}")

        except Exception as e:
            print(f"    d={d}: 失败 - {e}")
            results[d] = {"fid": float("inf"), "error": str(e)}

    # 绘制对比
    if len(results) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ds = sorted(results.keys())
        fids = [results[d]["fid"] for d in ds]
        ratios = [d / 64 for d in ds]

        ax.bar(range(len(ds)), fids, color=["#2196F3", "#FF9800", "#4CAF50", "#F44336"][:len(ds)])
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels([f"d={d}\nd/m={d/64:.0f}" for d in ds])
        ax.set_ylabel("FID Score")
        ax.set_title("FID vs Ambient Dimension (fixed m=64)")
        ax.grid(True, alpha=0.3, axis="y")

        for i, (d, f) in enumerate(zip(ds, fids)):
            ax.text(i, f + 0.5, f"{f:.1f}", ha="center", fontsize=10)

        fig.tight_layout()
        fig.savefig(save_dir / "06_dimension_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ 保存: 06_dimension_comparison.png")


def _generate_summary(summary, fid, recon, cos_emp, cos_tt, save_dir):
    """生成文本摘要报告。"""
    report = []
    report.append("=" * 60)
    report.append("DiffusionGeometryLab — 合成数据测试报告")
    report.append("=" * 60)
    report.append("")
    report.append(f"训练步数: {summary['final_step']}")
    report.append(f"Epochs: {summary['epochs']}")
    report.append(f"Gen 检测: {summary['gen_detected']} (step={summary.get('gen_step', 'N/A')})")
    report.append(f"Mem 检测: {summary['mem_detected']}")
    report.append(f"设备: {summary['device']}")
    report.append("")
    report.append(f"标准 FID: {fid:.2f}")
    report.append("")
    report.append("重建差距 (各时间步):")
    for i, t in enumerate(recon["timesteps"]):
        report.append(f"  t={t:.1f}: train_mse={recon['train_mse'][i]:.6f}, "
                      f"test_mse={recon['test_mse'][i]:.6f}, "
                      f"gap={recon['gap'][i]:.4f}")
    report.append("")
    report.append(f"S_cos(model, empirical) 平均: {cos_emp['mean'].mean():.4f}")
    report.append(f"S_cos(train, test) 平均: {cos_tt['mean'].mean():.4f}")
    report.append("")
    report.append("输出文件:")
    for f in sorted(save_dir.glob("*.png")):
        report.append(f"  📊 {f.name}")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    with open(save_dir / "report.txt", "w") as f:
        f.write(report_text)
    print(f"  ✅ 报告已保存: report.txt")


def _list_outputs(save_dir):
    """列出所有输出文件。"""
    print("\n📁 输出文件列表:")
    for f in sorted(save_dir.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            rel = f.relative_to(save_dir)
            print(f"  {rel} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
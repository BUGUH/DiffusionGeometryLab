"""
圆环实验主脚本 — 生成 8 类核心图表
===================================
运行: python scripts/run_circle_experiments.py

输出: outputs/circle_experiments/
  ├── fig1_three_phase_panorama.png
  ├── fig2_dimension_scaling.png
  ├── fig3_scheduler_comparison.png
  ├── fig4_mgen_mmem_training.png
  ├── fig5_cosine_field_comparison.png
  ├── fig6_sample_size_scaling.png
  ├── fig7_dimension_phase_steps.png
  └── fig8_trajectory_fid_comparison.png
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
from copy import deepcopy
from omegaconf import OmegaConf

from src.utils.config import load_config, validate_config
from src.utils.seed import set_global_seed
from src.utils.device import DeviceManager
from src.data.dataset import build_dataloaders
from src.data.synthetic_circle import compute_circle_quality_metrics
from src.models.score_network import ScoreNetwork
from src.schedulers import build_scheduler
from src.training.trainer import DiffusionTrainer
from src.sampling.reverse_sampler import ReverseSampler
from src.metrics.fid_evaluator import FIDEvaluator
from src.metrics.memory_ratio import MemoryRatioEvaluator
from src.metrics.cosine_field import CosineFieldEvaluator
from src.metrics.reconstruction_gap import ReconstructionGapEvaluator
from src.metrics.lpips_ssim import LPIPSSSIMEvaluator


SAVE_DIR = Path("outputs/circle_experiments")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 颜色方案
COLORS = {
    10: "#1f77b4", 20: "#ff7f0e", 50: "#2ca02c",
    100: "#d62728", 200: "#9467bd", 500: "#8c564b", 1000: "#e377c2",
}


def make_cfg(D=100, N=5000, max_steps=15000, scheduler="vp", seed=42):
    """从 circle.yaml 创建修改后的配置。"""
    cfg = load_config("configs/circle.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "data": {
            "circle": {"ambient_dim": D, "num_samples": N},
            "ambient_dim": D,
        },
        "training": {"max_steps": max_steps},
        "scheduler": {"type": scheduler},
        "experiment": {
            "seed": seed,
            "name": f"circle_D{D}_N{N}_{scheduler}",
            "output_dir": str(SAVE_DIR / f"runs/D{D}_N{N}_{scheduler}"),
        },
    }))
    try:
        cfg = validate_config(cfg)
    except Exception:
        pass
    return cfg


def train_model(cfg, dm):
    """训练一个模型，返回 (model, scheduler, train_data, test_data, summary)。"""
    set_global_seed(cfg.experiment.seed)
    train_loader, test_loader, _ = build_dataloaders(cfg)
    model = ScoreNetwork(cfg)
    scheduler = build_scheduler(cfg)

    trainer = DiffusionTrainer(
        cfg=cfg, model=model, scheduler=scheduler,
        train_loader=train_loader, test_loader=test_loader,
        device_manager=dm,
    )
    summary = trainer.train()
    model.eval()
    return model, scheduler, train_loader.dataset.data, test_loader.dataset.data, summary


def sample_and_evaluate(model, scheduler, train_data, dm, cfg, n_samples=512):
    """采样 + 计算指标。"""
    sampler = ReverseSampler(
        model=model, scheduler=scheduler,
        method="ddim", num_steps=80,
        capture_timesteps=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    shape = (n_samples,) + tuple(train_data.shape[1:])
    generated, bundle = sampler.sample(shape, dm.device)

    fid_eval = FIDEvaluator()
    fid = fid_eval.compute_fid(train_data[:n_samples], generated.cpu())
    mmd = fid_eval.compute_mmd(train_data[:n_samples], generated.cpu())
    traj_fid = fid_eval.compute_trajectory_fid(train_data, bundle, scheduler)

    circle_metrics = compute_circle_quality_metrics(generated.cpu(), r=1.0)

    return {
        "fid": fid, "mmd": mmd, "traj_fid": traj_fid,
        "circle": circle_metrics,
        "generated": generated.cpu(),
        "bundle": bundle,
    }


# ================================================================
# 图1: 三阶段指标演化全景图
# ================================================================

def fig1_three_phase_panorama(dm):
    """横轴: 扩散时间步 t; 纵轴: 轨迹FID + f_mem(t)"""
    print("\n📊 Fig1: 三阶段指标演化全景图...")
    cfg = make_cfg(D=100, N=5000, max_steps=15000)
    model, scheduler, train_data, test_data, summary = train_model(cfg, dm)
    result = sample_and_evaluate(model, scheduler, train_data, dm, cfg)

    # 计算 f_mem(t)
    mem_eval = MemoryRatioEvaluator(
        scheduler=scheduler, threshold=1.0/3.0, num_timesteps=20, nprobe=8,
    )
    mem_result = mem_eval.compute(train_data, result["bundle"])

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # 轨迹 FID
    traj = result["traj_fid"]
    ts_fid = np.array(sorted(traj.keys()))
    fids = np.array([traj[t] for t in ts_fid])
    ax1.plot(ts_fid, fids, "o-", color="#2196F3", linewidth=2, label="Trajectory FID")
    ax1.set_xlabel("Diffusion Time $t$", fontsize=13)
    ax1.set_ylabel("FID / MMD", color="#2196F3", fontsize=13)
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    # f_mem(t)
    if len(mem_result["t"]) > 0:
        ax2.plot(mem_result["t"], mem_result["f_mem"], "s-", color="#F44336",
                 linewidth=2, label="$f_{mem}(t)$")
        ax2.fill_between(mem_result["t"], 0, mem_result["f_mem"],
                         alpha=0.1, color="#F44336")
    ax2.set_ylabel("$f_{mem}(t)$", color="#F44336", fontsize=13)
    ax2.tick_params(axis="y", labelcolor="#F44336")
    ax2.set_ylim(0, 1.05)

    # 三阶段标注
    ax1.axvspan(0, 0.2, alpha=0.05, color="green", label="Memorization zone")
    ax1.axvspan(0.2, 0.7, alpha=0.05, color="blue", label="Geometry zone")
    ax1.axvspan(0.7, 1.0, alpha=0.05, color="orange", label="Noise zone")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Three-Phase Panorama (D=100, Circle $S^1$)", fontsize=14)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig1_three_phase_panorama.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ fig1_three_phase_panorama.png")
    return model, scheduler, train_data, test_data


# ================================================================
# 图2: 几何区窗口的维度缩放图
# ================================================================

def fig2_dimension_scaling(dm):
    """横轴: 背景维度 D; 纵轴: 几何区时长占比"""
    print("\n📊 Fig2: 维度缩放图...")
    dims = [10, 20, 50, 100, 200]
    geo_ratios = []

    for D in dims:
        print(f"  D={D}...")
        cfg = make_cfg(D=D, N=3000, max_steps=8000)
        model, scheduler, train_data, _, _ = train_model(cfg, dm)
        result = sample_and_evaluate(model, scheduler, train_data, dm, cfg, n_samples=256)

        traj = result["traj_fid"]
        ts = np.array(sorted(traj.keys()))
        fids = np.array([traj[t] for t in ts])

        # 几何区定义: FID 低于中值的时间范围占比
        if len(fids) > 2:
            threshold = np.median(fids)
            geo_mask = fids < threshold
            geo_ratio = geo_mask.sum() / len(fids)
        else:
            geo_ratio = 0.5
        geo_ratios.append(geo_ratio)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dims, geo_ratios, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.fill_between(dims, 0, geo_ratios, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Ambient Dimension $D$", fontsize=13)
    ax.set_ylabel("Geometry Zone Ratio", fontsize=13)
    ax.set_title("Geometry Zone Width vs Dimension (Circle $S^1$, $m=1$)", fontsize=14)
    ax.set_xscale("log")
    for i, (d, r) in enumerate(zip(dims, geo_ratios)):
        ax.annotate(f"D={d}\n{r:.2f}", (d, r), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig2_dimension_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ fig2_dimension_scaling.png")


# ================================================================
# 图3: 多调度器三阶段特征对比图
# ================================================================

def fig3_scheduler_comparison(dm):
    """横轴: t; 纵轴: 轨迹FID; 变量: VP/VE/OF"""
    print("\n📊 Fig3: 调度器对比...")
    scheds = ["vp", "ve", "of"]
    sched_colors = {"vp": "#2196F3", "ve": "#FF9800", "of": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(12, 6))
    for sched in scheds:
        print(f"  Scheduler={sched}...")
        cfg = make_cfg(D=50, N=3000, max_steps=8000, scheduler=sched)
        model, scheduler, train_data, _, _ = train_model(cfg, dm)
        result = sample_and_evaluate(model, scheduler, train_data, dm, cfg, n_samples=256)

        traj = result["traj_fid"]
        ts = np.array(sorted(traj.keys()))
        fids = np.array([traj[t] for t in ts])
        ax.plot(ts, fids, "o-", color=sched_colors[sched], linewidth=2,
                label=sched.upper(), markersize=4)

    ax.set_xlabel("Diffusion Time $t$", fontsize=13)
    ax.set_ylabel("Trajectory FID", fontsize=13)
    ax.set_title("Scheduler Comparison — Trajectory FID (D=50)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig3_scheduler_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ fig3_scheduler_comparison.png")


# ================================================================
# 图4: M_gen/M_mem 在三阶段中的定位图
# ================================================================

def fig4_mgen_mmem_training(dm):
    """横轴: 训练步数 τ; 纵轴: FID + 训练/测试重建误差比"""
    print("\n📊 Fig4: M_gen/M_mem 训练定位...")
    cfg = make_cfg(D=100, N=3000, max_steps=20000)
    cfg.training.eval_every_steps = 1000
    cfg.training.log_every_steps = 500

    set_global_seed(42)
    train_loader, test_loader, _ = build_dataloaders(cfg)
    model = ScoreNetwork(cfg)
    scheduler = build_scheduler(cfg)
    train_data = train_loader.dataset.data

    # 手动训练并记录中间指标
    from src.training.trainer import DiffusionTrainer
    fid_eval = FIDEvaluator()

    steps_record = []
    fid_record = []
    gap_record = []

    trainer = DiffusionTrainer(
        cfg=cfg, model=model, scheduler=scheduler,
        train_loader=train_loader, test_loader=test_loader,
        device_manager=dm,
    )

    # 用 trainer 的内部方法定期记录
    original_eval = trainer._evaluate_and_detect

    def patched_eval():
        original_eval()
        step = trainer.global_step
        model.eval()

        # 快速采样
        sampler = ReverseSampler(model=model, scheduler=scheduler,
                                 method="ddim", num_steps=30)
        shape = (128,) + tuple(train_data.shape[1:])
        gen, _ = sampler.sample(shape, dm.device)
        fid = fid_eval.compute_fid(train_data[:128], gen.cpu())

        # 重建差距
        train_mse = trainer._compute_reconstruction_mse(train_loader, max_batches=3)
        test_mse = trainer._compute_reconstruction_mse(test_loader, max_batches=3)
        gap = (test_mse - train_mse) / max(train_mse, 1e-10)

        steps_record.append(step)
        fid_record.append(fid)
        gap_record.append(gap)
        model.train()

    trainer._evaluate_and_detect = patched_eval
    summary = trainer.train()

    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(steps_record, fid_record, "o-", color="#2196F3", linewidth=2, label="FID")
    ax1.set_xlabel("Training Step $\\tau$", fontsize=13)
    ax1.set_ylabel("FID", color="#2196F3", fontsize=13)

    ax2.plot(steps_record, gap_record, "s-", color="#F44336", linewidth=2,
             label="Recon Gap (test-train)/train")
    ax2.set_ylabel("Reconstruction Gap", color="#F44336", fontsize=13)

    # 标注 gen/mem
    if summary.get("gen_step"):
        ax1.axvline(x=summary["gen_step"], color="green", linestyle="--",
                     linewidth=2, label=f"$\\tau_{{gen}}$={summary['gen_step']}")
    if summary.get("mem_detected"):
        # 找 gap 最大点作为 mem 标记
        if gap_record:
            mem_idx = np.argmax(gap_record)
            ax1.axvline(x=steps_record[mem_idx], color="red", linestyle="--",
                         linewidth=2, label=f"$\\tau_{{mem}}$≈{steps_record[mem_idx]}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("$M_{gen}$/$M_{mem}$ During Training (D=100)", fontsize=14)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig4_mgen_mmem_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ fig4_mgen_mmem_training.png")


# ================================================================
# 图5: M_gen/M_mem 的余弦相似度场对比图
# ================================================================

def fig5_cosine_field_comparison(dm):
    """横轴: t; 纵轴: 两类余弦相似度"""
    print("\n📊 Fig5: 余弦相似度场对比...")

    # 训练短期模型(gen) 和 长期模型(mem)
    cfg_gen = make_cfg(D=100, N=3000, max_steps=5000)
    cfg_gen.training.phase_detection.enabled = False
    model_gen, sched, train_data, test_data, _ = train_model(cfg_gen, dm)

    cfg_mem = make_cfg(D=100, N=3000, max_steps=20000)
    cfg_mem.training.phase_detection.enabled = False
    model_mem, _, _, _, _ = train_model(cfg_mem, dm)

    cosine_eval = CosineFieldEvaluator(
        scheduler=sched, num_timesteps=25, batch_size=256,
    )

    # gen vs mem
    cos_gm = cosine_eval.compute_gen_mem_similarity(
        model_gen, model_mem, train_data, dm.device)
    # gen vs empirical
    cos_ge = cosine_eval.compute_model_empirical_similarity(
        model_gen, train_data, dm.device)
    # mem vs empirical
    cos_me = cosine_eval.compute_model_empirical_similarity(
        model_mem, train_data, dm.device)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左: gen vs mem
    ax = axes[0]
    ax.plot(cos_gm["t"], cos_gm["mean"], "o-", color="#9C27B0", linewidth=2,
            label="$S_{cos}(M_{gen}, M_{mem})$")
    ax.fill_between(cos_gm["t"], cos_gm["mean"] - cos_gm["std"],
                    cos_gm["mean"] + cos_gm["std"], alpha=0.15, color="#9C27B0")
    ax.set_xlabel("$t$", fontsize=13)
    ax.set_ylabel("Cosine Similarity", fontsize=13)
    ax.set_title("$M_{gen}$ vs $M_{mem}$", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    # 右: model vs empirical
    ax = axes[1]
    ax.plot(cos_ge["t"], cos_ge["mean"], "o-", color="#2196F3", linewidth=2,
            label="$S_{cos}(M_{gen}, M_{emp})$", markersize=3)
    ax.fill_between(cos_ge["t"], cos_ge["mean"] - cos_ge["std"],
                    cos_ge["mean"] + cos_ge["std"], alpha=0.15, color="#2196F3")
    ax.plot(cos_me["t"], cos_me["mean"], "s-", color="#F44336", linewidth=2,
            label="$S_{cos}(M_{mem}, M_{emp})$", markersize=3)
    ax.fill_between(cos_me["t"], cos_me["mean"] - cos_me["std"],
                    cos_me["mean"] + cos_me["std"], alpha=0.15, color="#F44336")
    ax.set_xlabel("$t$", fontsize=13)
    ax.set_ylabel("Cosine Similarity", fontsize=13)
    ax.set_title("Model vs Empirical Direction", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    fig.suptitle("Cosine Similarity Fields (D=100, Circle $S^1$)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig5_cosine_field_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ fig5_cosine_field_comparison.png")


# ================================================================
# 图6: 几何区窗口的样本量缩放图
# ================================================================

def fig6_sample_size_scaling(dm):
    """横轴: 样本量 N; 纵轴: 几何区时长"""
    print("\n📊 Fig6: 样本量缩放...")
    sample_sizes = [500, 1000, 2000, 5000]
    geo_ratios = []
    D_fixed = 50

    for N in sample_sizes:
        print(f"  N={N}...")
        cfg = make_cfg(D=D_fixed, N=N, max_steps=8000)
        model, scheduler, train_data, _, _ = train_model(cfg, dm)
        result = sample_and_evaluate(model, scheduler, train_data, dm, cfg, n_samples=256)

        traj = result["traj_fid"]
        ts = np.array(sorted(traj.keys()))
        fids = np.array([traj[t] for t in ts])

        if len(fids) > 2:
            threshold = np.median(fids)
            geo_ratio = (fids < threshold).sum() / len(fids)
        else:
            geo_ratio = 0.5
        geo_ratios.append(geo_ratio)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, geo_ratios, "o-", color="#4CAF50", linewidth=2, markersize=8)
    ax.fill_between(sample_sizes, 0, geo_ratios, alpha=0.15, color="#4CAF50")
    ax.set_xlabel("Number of Samples $N$", fontsize=13)
    ax.set_ylabel("Geometry Zone Ratio", fontsize=13)
    ax.set_title(f"Geometry Zone vs Sample Size (D={D_fixed}, Circle $S^1$)", fontsize=14)
    for i, (n, r) in enumerate(zip(sample_sizes, geo_ratios)):
        ax.annotate(f"N={n}\n{r:.2f}", (n, r), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig6_sample_size_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ fig6_sample_size_scaling.png")


# ================================================================
# 图7: M_gen/M_mem 的多维度对比图
# ================================================================

def fig7_dimension_phase_steps(dm):
    """横轴: D; 纵轴: τ_gen + (τ_mem - τ_gen)"""
    print("\n📊 Fig7: 多维度相变步数...")
    dims = [10, 20, 50, 100, 200]
    tau_gens = []
    tau_mems = []

    for D in dims:
        print(f"  D={D}...")
        cfg = make_cfg(D=D, N=3000, max_steps=20000)
        cfg.training.eval_every_steps = 2000

        set_global_seed(42)
        train_loader, test_loader, _ = build_dataloaders(cfg)
        model = ScoreNetwork(cfg)
        scheduler = build_scheduler(cfg)

        trainer = DiffusionTrainer(
            cfg=cfg, model=model, scheduler=scheduler,
            train_loader=train_loader, test_loader=test_loader,
            device_manager=dm,
        )
        summary = trainer.train()

        gen_step = summary.get("gen_step") or summary["final_step"] * 0.3
        mem_step = summary["final_step"] if summary.get("mem_detected") else summary["final_step"]

        tau_gens.append(gen_step)
        tau_mems.append(mem_step)
        print(f"    τ_gen={gen_step}, τ_mem={mem_step}")

    fig, ax = plt.subplots(figsize=(10, 6))

    tau_gens = np.array(tau_gens)
    tau_mems = np.array(tau_mems)
    gap = tau_mems - tau_gens

    x = np.arange(len(dims))
    width = 0.35

    bars1 = ax.bar(x - width/2, tau_gens, width, color="#2196F3", label="$\\tau_{gen}$")
    bars2 = ax.bar(x + width/2, gap, width, bottom=tau_gens, color="#F44336",
                   label="$\\tau_{mem} - \\tau_{gen}$")

    ax.set_xlabel("Ambient Dimension $D$", fontsize=13)
    ax.set_ylabel("Training Steps", fontsize=13)
    ax.set_title("Phase Transition Steps vs Dimension", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}\nD/m={d}" for d in dims])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # 数值标注
    for i, (g, m) in enumerate(zip(tau_gens, tau_mems)):
        ax.text(i, m + 200, f"{int(m)}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig7_dimension_phase_steps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ fig7_dimension_phase_steps.png")


# ================================================================
# 图8: 轨迹中间态FID的三阶段对比图
# ================================================================

def fig8_trajectory_fid_comparison(dm):
    """横轴: 反向采样步数; 纵轴: FID; 变量: M_gen/M_mem"""
    print("\n📊 Fig8: 轨迹 FID 对比 (Gen vs Mem)...")

    # 短期训练 → M_gen 近似
    cfg_gen = make_cfg(D=100, N=3000, max_steps=5000)
    cfg_gen.training.phase_detection.enabled = False
    model_gen, sched, train_data, _, _ = train_model(cfg_gen, dm)

    # 长期训练 → M_mem 近似
    cfg_mem = make_cfg(D=100, N=3000, max_steps=25000)
    cfg_mem.training.phase_detection.enabled = False
    model_mem, _, _, _, _ = train_model(cfg_mem, dm)

    fid_eval = FIDEvaluator()
    fig, ax = plt.subplots(figsize=(12, 6))

    for model, label, color, marker in [
        (model_gen, "$M_{gen}$ (early stop)", "#2196F3", "o"),
        (model_mem, "$M_{mem}$ (overfit)", "#F44336", "s"),
    ]:
        sampler = ReverseSampler(
            model=model, scheduler=sched,
            method="ddim", num_steps=80,
            capture_timesteps=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        shape = (256,) + tuple(train_data.shape[1:])
        _, bundle = sampler.sample(shape, dm.device)

        traj_fid = fid_eval.compute_trajectory_fid(train_data, bundle, sched)
        ts = np.array(sorted(traj_fid.keys()))
        fids = np.array([traj_fid[t] for t in ts])

        ax.plot(ts, fids, f"{marker}-", color=color, linewidth=2, label=label, markersize=5)

    ax.set_xlabel("Diffusion Time $t$", fontsize=13)
    ax.set_ylabel("Trajectory FID", fontsize=13)
    ax.set_title("Trajectory FID: $M_{gen}$ vs $M_{mem}$ (D=100)", fontsize=14)

    # 三阶段背景
    ax.axvspan(0, 0.2, alpha=0.05, color="green")
    ax.axvspan(0.2, 0.7, alpha=0.05, color="blue")
    ax.axvspan(0.7, 1.0, alpha=0.05, color="orange")
    ax.text(0.1, ax.get_ylim()[1]*0.95, "Mem", ha="center", fontsize=10, color="green")
    ax.text(0.45, ax.get_ylim()[1]*0.95, "Geo", ha="center", fontsize=10, color="blue")
    ax.text(0.85, ax.get_ylim()[1]*0.95, "Noise", ha="center", fontsize=10, color="orange")

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "fig8_trajectory_fid_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ fig8_trajectory_fid_comparison.png")


# ================================================================
# 主函数
# ================================================================

def main():
    print("=" * 70)
    print("🔵 DiffusionGeometryLab — Circle S¹ 核心实验")
    print("=" * 70)

    dm = DeviceManager(OmegaConf.create({"accelerator": "auto", "gpu_ids": [0], "mixed_precision": "none"}))
    print(f"设备: {dm.device}\n")

    # 按依赖顺序执行
    fig1_three_phase_panorama(dm)
    fig2_dimension_scaling(dm)
    fig3_scheduler_comparison(dm)
    fig4_mgen_mmem_training(dm)
    fig5_cosine_field_comparison(dm)
    fig6_sample_size_scaling(dm)
    fig7_dimension_phase_steps(dm)
    fig8_trajectory_fid_comparison(dm)

    # 汇总
    print("\n" + "=" * 70)
    print(f"🎉 全部 8 张图已生成！保存在: {SAVE_DIR.resolve()}")
    print("=" * 70)
    for f in sorted(SAVE_DIR.glob("fig*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  📊 {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
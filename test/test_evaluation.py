import os
import sys
import time
import json
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import random

# ==========================================
# 0. 环境路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from sampling.numerical_sampling import simulate_ode_trajectory
from sampling.trajectory_utils import extract_hat_x0_from_trajectory
from metric.feature_extractor import load_feature_extractor, get_image_features
from metric.evaluate_metrics import calculate_memorization_ratio, calculate_precision_knn, precompute_test_radii
from models.unet import UNet
from schedulers.vp_scheduler import VPScheduler
from schedulers.ve_scheduler import VEScheduler
from schedulers.of_scheduler import OFScheduler

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 核心生成函数与绘图辅助函数
# ==========================================
class MockConfig:
    class SchedulerConfig:
        num_diffusion_steps = 1000
        of = {"sigma_min": 0.001}
        ve = {"sigma_min": 0.01, "sigma_max": 50.0}
    scheduler = SchedulerConfig()

@torch.no_grad()
def generate_nn_trajectories(model, scheduler, sched_name, device, num_samples=1000, img_size=64, num_steps=10):
    model.eval()
    B = num_samples
    shape = (B, 1, img_size, img_size)
    
    if sched_name == "VE":
        _, std_max = scheduler.marginal_params(torch.tensor([0.999], device=device))
        x_t = torch.randn(shape, device=device) * std_max.view(-1, 1, 1, 1)
    else:
        x_t = torch.randn(shape, device=device)
        
    t_steps = torch.linspace(0.999, 0.001, num_steps, device=device)
    dt = t_steps[0] - t_steps[1]
    
    trajectories_xt, trajectories_x0 = [], []
    
    for i, t in enumerate(tqdm(t_steps, desc=f"⏳ 提取 NN 轨迹 [{sched_name}]", leave=False)):
        t_batch = t.expand(B)
        mean_coeff, std = scheduler.marginal_params(t_batch)
        mean_coeff_view = mean_coeff.view(B, 1, 1, 1)
        std_view = std.view(B, 1, 1, 1)
        scale_factor = torch.sqrt(mean_coeff_view**2 + std_view**2 + 1e-8)
        
        x_in = x_t / scale_factor
        pred = model(x_in, t_batch * 1000.0)
        
        if sched_name == "OF":
            x_0_pred = x_t - pred * t_batch.view(-1, 1, 1, 1)
            x_t_next = x_t - pred * dt
        else:
            x_0_pred = (x_t - std_view * pred) / mean_coeff_view
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)
            if i < num_steps - 1:
                t_next = t_steps[i + 1].expand(B)
                mean_next, std_next = scheduler.marginal_params(t_next)
                x_t_next = mean_next.view(B, 1, 1, 1) * x_0_pred + std_next.view(B, 1, 1, 1) * pred
            else:
                x_t_next = x_0_pred
                
        trajectories_xt.append(x_t.cpu().numpy())
        trajectories_x0.append(x_0_pred.cpu().numpy())
        x_t = x_t_next
        
    return np.stack(trajectories_xt, axis=1), np.stack(trajectories_x0, axis=1), t_steps.cpu().numpy()

def plot_and_save(trajectories, hat_x0_trajectories, t_axis, precisions, memorizations, title, save_path, num_plot=4, num_steps=10, is_nn=False):
    """把繁琐的画图代码封装起来，供 ODE 和 NN 共同调用"""
    fig = plt.figure(figsize=(18, 2.5 * num_plot * 2 + 5))
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[num_plot * 2, 2], hspace=0.3)
    inner_gs = gridspec.GridSpecFromSubplotSpec(num_plot * 2, num_steps, subplot_spec=outer_gs[0], wspace=0.1, hspace=0.1)
    
    for i in range(num_plot):
        for j in range(num_steps):
            ax_xt = fig.add_subplot(inner_gs[i * 2, j])
            ax_xt.imshow(np.clip((trajectories[i, j, 0] + 1.0) / 2.0, 0.0, 1.0), cmap='gray')
            ax_xt.axis('off')
            
            ax_x0 = fig.add_subplot(inner_gs[i * 2 + 1, j])
            ax_x0.imshow(np.clip((hat_x0_trajectories[i, j, 0] + 1.0) / 2.0, 0.0, 1.0), cmap='gray')
            ax_x0.axis('off')
            
            if i == 0:
                t_label = f"Start\nt={t_axis[j]:.3f}" if j==0 else f"End\nt={t_axis[j]:.3f}" if j==num_steps-1 else f"t={t_axis[j]:.3f}"
                ax_xt.set_title(t_label, fontsize=11, fontweight='bold' if j==0 or j==num_steps-1 else 'normal')
            if j == 0:
                ax_xt.text(-0.2, 0.5, r"Actual $x_t$", transform=ax_xt.transAxes, fontsize=14, va='center', ha='right', fontweight='bold')
                ax_x0.text(-0.2, 0.5, r"Predicted $\hat{x}_0$", transform=ax_x0.transAxes, fontsize=14, va='center', ha='right', fontweight='bold', color='blue')
                
    metric_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1], wspace=0.2)
    
    ax_m1 = fig.add_subplot(metric_gs[0])
    ax_m1.plot(t_axis, precisions, marker='o', color='#1f77b4', linewidth=2)
    if is_nn: ax_m1.invert_xaxis()
    ax_m1.set_title(r'Precision of Predicted $\hat{x}_0$ over Time', fontsize=12)
    ax_m1.set_xlabel('Time Step (t)', fontsize=10)
    ax_m1.set_ylabel('Precision (%)', fontsize=10)
    ax_m1.set_ylim(-5, 105)
    ax_m1.grid(True, linestyle='--', alpha=0.7)
    
    ax_m2 = fig.add_subplot(metric_gs[1])
    ax_m2.plot(t_axis, memorizations, marker='s', color='#d62728', linewidth=2)
    if is_nn: ax_m2.invert_xaxis()
    ax_m2.set_title(r'Memorization of Predicted $\hat{x}_0$ over Time', fontsize=12)
    ax_m2.set_xlabel('Time Step (t)', fontsize=10)
    ax_m2.set_ylabel('Memorization (%)', fontsize=10)
    ax_m2.set_ylim(-5, 105)
    ax_m2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.92)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# ==========================================
# 2. 主流程：统一评估 ODE 与 NN
# ==========================================
def main():
    seed_everything(42)

    NUM_EVAL_SAMPLES = 1000 
    NUM_PLOT_SAMPLES = 4
    NUM_STEPS = 10
    DEFAULT_K = 3
    MODES_TO_TEST = ["VP", "VE", "OF"]

    DATA_CONFIG = {
        1024: 50000,
        4096: 50000,
        32768: 100000
    }

    # 路径
    RAW_TEST_DATA = os.path.join(project_root, "kaggle_export", "celeba_test_FULL_N202599.npy")
    TRAIN_DATA_TEMPLATE = os.path.join(project_root, "kaggle_export", "celeba_train_N{}.npy")
    CKPT_TEMPLATE = os.path.join(project_root, "checkpoints_v2", "N{}_{}", "history_slices", "step_{}.pth")
    
    RESULTS_DIR = os.path.join(project_root, "results", "eval_combined")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CACHE_DIR = os.path.join(project_root, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # --- 阶段 A：全局加载特征提取器与测试集特征 (只需做一次) ---
    model = load_feature_extractor(device)
    model.eval() 
    for param in model.parameters(): param.requires_grad = False

    print("\n📦 [全局准备] 提取测试集 20万 张图片的特征...")
    raw_test_images = np.load(RAW_TEST_DATA)
    test_features = get_image_features(
        raw_test_images, model, device, batch_size=128,
        desc="提取 Test 特征", cache_path=os.path.join(CACHE_DIR, "test_inception_features.npy")
    )
    radii_k3 = precompute_test_radii(test_features, k_values=[DEFAULT_K])[DEFAULT_K]

    all_results = {"ODE": {}, "NN": {}}
    cfg = MockConfig()

    # --- 阶段 B：遍历不同的数据量 ---
    for n_samples, target_step in DATA_CONFIG.items():
        print(f"\n{'='*70}")
        print(f"📊 正在处理数据集: N={n_samples} (NN对应检查点: {target_step}步)")
        print(f"{'='*70}")

        # 每个数据量提取一次训练集特征 (NN和ODE共享)
        raw_train_images = np.load(TRAIN_DATA_TEMPLATE.format(n_samples))
        train_features = get_image_features(raw_train_images, model, device, batch_size=64, desc=f"提取 Train 特征 (N={n_samples})")

        for mode in MODES_TO_TEST:
            print(f"\n🚀 --- 开始测试组合: N={n_samples} | 模式={mode} ---")

            # ==========================================
            # 任务 1：评测精确数值 ODE 轨迹
            # ==========================================
            print(f"▶️ [任务 1/2] 正在评估精确数值 ODE (Mode: {mode})")
            ode_mode = "OT" if mode == "OF" else mode # 解决 OF 报错问题
            
            trajectories = simulate_ode_trajectory(
                z_data=raw_train_images, schedule_mode=ode_mode, num_samples=NUM_EVAL_SAMPLES, steps=NUM_STEPS, method='RK45'
            )
            ode_t_axis = np.linspace(0, 1, NUM_STEPS)
            hat_x0_trajectories = extract_hat_x0_from_trajectory(trajectories, ode_t_axis, ode_mode)

            ode_prec, ode_memo = [], []
            for j in tqdm(range(NUM_STEPS), desc=f"ODE 指标评估"):
                feats = get_image_features(np.clip(hat_x0_trajectories[:, j, ...], -1, 1).astype(np.float32), model, device, batch_size=64, desc="")
                prec = calculate_precision_knn(feats, test_features, radii_k3)
                memo = calculate_memorization_ratio(feats, train_features, ratio_threshold=1.0/3.0) # 统一用 1.0/3.0
                ode_prec.append(prec * 100); ode_memo.append(memo * 100)

            all_results["ODE"][f"N{n_samples}_{mode}"] = {"prec": ode_prec, "memo": ode_memo, "t": ode_t_axis.tolist()}
            
            plot_and_save(
                trajectories, hat_x0_trajectories, ode_t_axis, ode_prec, ode_memo,
                title=f"Exact ODE Trajectory (N={n_samples}, Mode: {mode})",
                save_path=os.path.join(RESULTS_DIR, f"ODE_N{n_samples}_{mode}.png"),
                num_plot=NUM_PLOT_SAMPLES, num_steps=NUM_STEPS, is_nn=False
            )
            # 释放 ODE 内存
            del trajectories, hat_x0_trajectories, feats
            gc.collect(); torch.cuda.empty_cache()

            # ==========================================
            # 任务 2：评测神经网络 NN 轨迹
            # ==========================================
            print(f"▶️ [任务 2/2] 正在评估神经网络 NN (Mode: {mode}, Step: {target_step})")
            ckpt_path = CKPT_TEMPLATE.format(n_samples, mode, target_step)
            
            if not os.path.exists(ckpt_path):
                print(f"⚠️ 跳过 NN: 找不到权重文件 {ckpt_path}")
            else:
                if mode == "VP": scheduler = VPScheduler(cfg)
                elif mode == "VE": scheduler = VEScheduler(cfg)
                elif mode == "OF": scheduler = OFScheduler(cfg)

                unet = UNet(in_channels=1, out_channels=1, base_channels=64, channel_multipliers=(1, 2, 4, 8)).to(device)
                unet.load_state_dict(torch.load(ckpt_path, map_location=device))

                nn_traj, nn_hat_x0, nn_t_axis = generate_nn_trajectories(
                    unet, scheduler, mode, device, num_samples=NUM_EVAL_SAMPLES, num_steps=NUM_STEPS
                )

                nn_prec, nn_memo = [], []
                for j in tqdm(range(NUM_STEPS), desc=f"NN 指标评估"):
                    feats = get_image_features(np.clip(nn_hat_x0[:, j, ...], -1, 1).astype(np.float32), model, device, batch_size=64, desc="")
                    prec = calculate_precision_knn(feats, test_features, radii_k3)
                    memo = calculate_memorization_ratio(feats, train_features, ratio_threshold=1.0/3.0) # 统一用 1.0/3.0
                    nn_prec.append(prec * 100); nn_memo.append(memo * 100)

                all_results["NN"][f"N{n_samples}_{mode}_S{target_step}"] = {"prec": nn_prec, "memo": nn_memo, "t": nn_t_axis.tolist()}

                plot_and_save(
                    nn_traj, nn_hat_x0, nn_t_axis, nn_prec, nn_memo,
                    title=f"NN Process vs. Predicted Clean Data (N={n_samples}, Mode: {mode}, Step={target_step})",
                    save_path=os.path.join(RESULTS_DIR, f"NN_N{n_samples}_{mode}.png"),
                    num_plot=NUM_PLOT_SAMPLES, num_steps=NUM_STEPS, is_nn=True
                )
                # 释放 NN 内存
                unet.cpu()
                del unet, scheduler, nn_traj, nn_hat_x0, feats
                gc.collect(); torch.cuda.empty_cache()

    # --- 阶段 C：保存全局日志 ---
    log_path = os.path.join(RESULTS_DIR, "combined_metrics.json")
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n🎉 所有测试圆满完成！结果已保存在: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
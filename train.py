import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import math

# ==========================================
# 导入你的模块 (请确保路径正确)
# ==========================================
from models.unet import UNet
from schedulers.vp_scheduler import VPScheduler
from schedulers.ve_scheduler import VEScheduler
from schedulers.of_scheduler import OFScheduler

class MockConfig:
    class SchedulerConfig:
        num_diffusion_steps = 1000
        of = {"sigma_min": 0.001}
        ve = {"sigma_min": 0.01, "sigma_max": 50.0}
    scheduler = SchedulerConfig()

def cycle_dataloader(dataloader):
    """无限数据生成器，摆脱 Epoch 限制"""
    while True:
        for batch in dataloader:
            yield batch[0]

# ==========================================
# 核心训练流程
# ==========================================
def train_experiment(N, sched_name, args, device):
    print(f"\n{'='*60}")
    print(f"🚀 [物理相变实验 V2 启动] 数据量 N={N} | 调度器={sched_name}")
    print(f"{'='*60}")

    # 1. 路径与数据准备 (全部指向独立的 V2 文件夹，彻底物理隔离)
    TRAIN_DATA_PATH = f"./kaggle_export/celeba_train_N{N}.npy"
    # 全局共享验证集，无论 N 是多少，都用这 1000 张图考试！
    VAL_DATA_PATH = f"./kaggle_export/celeba_val_shared_N1000.npy"

    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VAL_DATA_PATH):
        print(f"⚠️ [跳过] 找不到数据文件。请检查: {TRAIN_DATA_PATH} 或 {VAL_DATA_PATH}")
        return

    # 存储目录 (本地防断电 + 云盘存最佳/切片) -> 全部加上 _v2
    LOCAL_CKPT_DIR = f"/root/local_checkpoints_v2/N{N}_{sched_name}"
    CLOUD_CKPT_DIR = f"/root/shared-nvme/wangjianxing/checkpoints_v2/N{N}_{sched_name}"
    SLICE_DIR = os.path.join(CLOUD_CKPT_DIR, "history_slices") # 专门存物理切片的文件夹

    os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
    os.makedirs(CLOUD_CKPT_DIR, exist_ok=True)
    os.makedirs(SLICE_DIR, exist_ok=True)
    
    local_resume_path = os.path.join(LOCAL_CKPT_DIR, "latest_resume.pth")
    cloud_best_path = os.path.join(CLOUD_CKPT_DIR, "best_generalization_model.pth")

    print("📥 加载 训练集 与 共享验证集...")
    train_data = np.load(TRAIN_DATA_PATH)
    val_data = np.load(VAL_DATA_PATH)
    
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    val_size = len(val_dataset)
    print(f"📊 数据划分: 物理训练集={len(train_dataset)} | 共享模拟考={val_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # 验证集不打乱，确保每次评估的条件绝对一致
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    data_iterator = cycle_dataloader(train_dataloader)

    # 2. 初始化模型与调度器
    model = UNet(in_channels=1, out_channels=1, base_channels=64, channel_multipliers=(1, 2, 4, 8)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    cfg = MockConfig()
    if sched_name == "VP": scheduler = VPScheduler(cfg)
    elif sched_name == "VE": scheduler = VEScheduler(cfg)
    elif sched_name == "OF": scheduler = OFScheduler(cfg)
    else: raise ValueError("未知的调度器")

    # 3. 恢复断点机制
    global_step = 0
    best_val_loss = float('inf') # 🌟 核心修正：使用 Val Loss 而不是 Train Loss 来选模型
    
    if os.path.exists(local_resume_path):
        print(f"🔄 检测到本地续传节点: {local_resume_path}")
        ckpt = torch.load(local_resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt['step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"✅ 成功从第 {global_step} 步继续！历史最佳 Val Loss: {best_val_loss:.5f}")

    # 4. 开始训练
    model.train()
    pbar = tqdm(total=args.total_steps, initial=global_step, desc=f"N={N} [{sched_name}]", dynamic_ncols=True)
    loss_buffer = []

    while global_step < args.total_steps:
        x_0 = next(data_iterator).to(device)
        B = x_0.shape[0]

        # (a) 采样连续时间 t ∈ [0.001, 0.999]
        t = torch.rand(B, device=device) * 0.998 + 0.001 
        
        # (b) 获取边际分布参数并加噪
        mean_coeff, std = scheduler.marginal_params(t)
        mean_coeff_view = mean_coeff.view(B, 1, 1, 1)
        std_view = std.view(B, 1, 1, 1)
        
        noise = torch.randn_like(x_0)
        x_t = mean_coeff_view * x_0 + std_view * noise

        # 🌟 关键修正：将所有调度器的输入方差归一化到 1，绝对防止爆炸
        scale_factor = torch.sqrt(mean_coeff_view**2 + std_view**2 + 1e-8)
        x_in = x_t / scale_factor

        # (c) 确定目标并预测
        target = (noise - x_0) if sched_name == "OF" else noise
        pred = model(x_in, t * 1000.0)

        # (d) 计算损失
        loss = F.mse_loss(pred, target)

        # (e) 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # (f) 日志与平滑 Train Loss 计算
        loss_val = loss.item()
        loss_buffer.append(loss_val)
        if len(loss_buffer) > 100: loss_buffer.pop(0)
            
        smooth_train_loss = sum(loss_buffer) / len(loss_buffer)
        global_step += 1
        pbar.update(1)
        pbar.set_postfix({"Tr_Loss": f"{smooth_train_loss:.4f}"})

        # ==========================================
        # 5. 严苛评估与物理切片存档机制
        # ==========================================
        
        # [常规检查] 每 1000 步：在全局验证集上跑一遍
        if global_step % 1000 == 0:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for val_x_0 in val_dataloader:
                    val_x_0 = val_x_0[0].to(device)
                    val_B = val_x_0.shape[0]
                    
                    val_t = torch.rand(val_B, device=device) * 0.998 + 0.001 
                    val_m, val_s = scheduler.marginal_params(val_t)
                    val_m_view = val_m.view(val_B, 1, 1, 1)
                    val_s_view = val_s.view(val_B, 1, 1, 1)
                    
                    val_noise = torch.randn_like(val_x_0)
                    val_x_t = val_m_view * val_x_0 + val_s_view * val_noise
                    val_scale = torch.sqrt(val_m_view**2 + val_s_view**2 + 1e-8)
                    
                    val_pred = model(val_x_t / val_scale, val_t * 1000.0)
                    val_target = (val_noise - val_x_0) if sched_name == "OF" else val_noise
                    
                    # 累加 Loss（乘以 batch size 还原）
                    val_loss_sum += F.mse_loss(val_pred, val_target).item() * val_B
            
            # 计算全量 Val Loss
            current_val_loss = val_loss_sum / val_size
            model.train() # 切回训练模式
            
            # 在控制台打出对比日志
            pbar.write(f"🔍 Step {global_step} | Train: {smooth_train_loss:.4f} | Val: {current_val_loss:.4f}")

            # 存本地续传节点
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, local_resume_path)

            # 🌟 [存最佳泛化点] 突破 Val Loss 记录时保存 (防过拟合)
            if global_step > 5000 and current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model.state_dict(), cloud_best_path)
                pbar.write(f"🌟 [N={N}] 发现更优泛化点! 最佳模型已覆盖 (Val Loss={best_val_loss:.5f})")

        # 🌟🌟 [最核心：物理学切片] 无论泛化好坏，每 5000 步强制留底
        if global_step % 5000 == 0:
            slice_path = os.path.join(SLICE_DIR, f"step_{global_step:05d}.pth")
            torch.save(model.state_dict(), slice_path)
            pbar.write(f"📸 物理学时间切片已存档: {slice_path}")

    pbar.close()
    print(f"🎉 实验完成: N={N} | {sched_name}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=100000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用计算设备: {device}")

    # ========================================================
    # ⚠️ 行动指南控制台 ⚠️
    # ========================================================
    
    # 精简版满载测试阵列 (3种N * 3种调度器)
    N_list = [32768]
    schedulers = ["VP", "VE", "OF"]

    for N in N_list:
        for sched in schedulers:
            try:
                train_experiment(N, sched, args, device)
            except Exception as e:
                print(f"\n❌ [致命错误] 实验 N={N}, {sched} 崩溃: {e}")
                print("➡️ 将自动跳过该实验，继续下一个...")
                continue
                
    print("\n✅ 所有安排的训练计划已全部执行完毕！")
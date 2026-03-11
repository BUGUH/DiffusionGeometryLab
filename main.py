# main.py
import torch
import os
import sys
import argparse

# 获取当前 main.py 所在的目录，并添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.utils import load_config, set_seed, setup_logger, get_experiment_dir
from src.data.celeba_loader import get_dataloader, CelebAParquetDataset
from src.diffusion.schedulers import get_scheduler
from src.models.unet import SimpleUNet
from src.engine.trainer import DiffusionTrainer
from src.metrics.cosine_field import CosineFieldEvaluator
from src.metrics.memory_stats import MemoryRatioTracker
from src.visualization.plotter import GeometryPlotter

def run_training_phase(cfg, device, logger, exp_dir):
    """阶段一：训练与状态捕获"""
    logger.info("Phase 1: Training & State Capturing...")
    
    # 1. 数据加载
    train_loader = get_dataloader(
        parquet_path=cfg.data.source_path,
        target_dim=cfg.data.target_dim,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True
    )
    
    # 构建验证集加载器 (不shuffle)
    test_loader = get_dataloader(
        parquet_path=cfg.data.source_path, # 实际应切分数据集，此处演示复用
        target_dim=cfg.data.target_dim,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False
    )
    
    # 2. 调度器
    scheduler = get_scheduler(cfg)
    
    # 3. 训练器
    trainer = DiffusionTrainer(cfg, train_loader, test_loader, scheduler, device, exp_dir)
    trainer.run()
    
    return trainer.model, scheduler

def run_analysis_phase(cfg, device, logger, exp_dir, scheduler):
    """阶段二：几何指标分析"""
    logger.info("Phase 2: Geometry Analysis...")
    
    gen_path = os.path.join(exp_dir, "model_gen.pt")
    mem_path = os.path.join(exp_dir, "model_mem.pt")
    
    # 检查 Checkpoint 是否存在
    if not os.path.exists(gen_path) or not os.path.exists(mem_path):
        logger.error("Checkpoints not found. Skipping analysis.")
        return

    # 1. 加载模型
    model_gen = SimpleUNet().to(device)
    model_mem = SimpleUNet().to(device)
    
    ckpt_gen = torch.load(gen_path, map_location=device)
    model_gen.load_state_dict(ckpt_gen['model_state_dict'])
    model_gen.eval()
    
    ckpt_mem = torch.load(mem_path, map_location=device)
    model_mem.load_state_dict(ckpt_mem['model_state_dict'])
    model_mem.eval()
    
    # 2. 准备评估工具
    field_evaluator = CosineFieldEvaluator(scheduler, device)
    
    # 准备 Faiss 数据 (为了速度，仅取训练集前 1000 张)
    # 生产环境需加载全量数据
    dataset = CelebAParquetDataset(cfg.data.source_path, cfg.data.target_dim)
    subset = torch.utils.data.Subset(dataset, range(min(1000, len(dataset))))
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=100, shuffle=False)
    
    # 构建 Faiss 索引数据 [N, D]
    train_vectors = []
    for batch in subset_loader:
        train_vectors.append(batch.view(batch.shape[0], -1))
    train_vectors = torch.cat(train_vectors, dim=0).cpu().numpy()
    
    mem_tracker = MemoryRatioTracker(train_vectors, str(device), d_dim=cfg.data.target_dim)
    
    # 3. 循环计算时间步指标
    timesteps = range(0, cfg.diffusion.num_timesteps, 50) # 每隔 50 步采样
    results = {'S_cos_gen_mem': [], 'S_cos_emp_gen': [], 'S_cos_emp_mem': [], 'f_mem': []}
    
    # 获取一批测试数据
    data_iter = iter(subset_loader)
    x_0_batch = next(data_iter).to(device)
    
    logger.info("Analyzing timesteps...")
    with torch.no_grad():
        for t in timesteps:
            t_tensor = torch.full((x_0_batch.shape[0],), t, device=device, dtype=torch.long)
            noise = torch.randn_like(x_0_batch)
            
            # 加噪
            x_t = scheduler.add_noise(x_0_batch, noise, t_tensor)
            
            # 指标计算
            # 1. Gen vs Mem
            s1 = field_evaluator.compute_model_alignment(model_gen, model_mem, x_t, t_tensor)
            
            # 2. Empirical Alignment
            s2 = field_evaluator.compute_empirical_alignment(model_gen, x_t, x_0_batch, t_tensor)
            s3 = field_evaluator.compute_empirical_alignment(model_mem, x_t, x_0_batch, t_tensor)
            
            # 3. Memory Ratio
            f_mem, _ = mem_tracker.compute_dynamic_memory_ratio(x_t)
            
            results['S_cos_gen_mem'].append(s1)
            results['S_cos_emp_gen'].append(s2)
            results['S_cos_emp_mem'].append(s3)
            results['f_mem'].append(f_mem)
            
            logger.info(f"Step {t}: S_cos(gen,mem)={s1:.3f}, f_mem={f_mem:.3f}")

    # 4. 可视化
    plotter = GeometryPlotter(exp_dir)
    plotter.plot_similarity_fields(list(timesteps), results, cfg.name)
    plotter.plot_memory_ratio(list(timesteps), results['f_mem'], cfg.name)

def main():
    # 1. 初始化
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    
    exp_dir = get_experiment_dir(cfg)
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = setup_logger("DiffusionGeometryLab", os.path.join(exp_dir, "exp.log"))
    logger.info(f"Experiment started. Save dir: {exp_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. 执行阶段
    # 如果没有现成的 Checkpoint，则先训练
    # _, scheduler = run_training_phase(cfg, device, logger)
    
    # 直接进行训练，训练完成后会自动保存 Checkpoint
    run_training_phase(cfg, device, logger, exp_dir)
    
    # 3. 分析阶段
    scheduler = get_scheduler(cfg) # 重新实例化 scheduler
    run_analysis_phase(cfg, device, logger, exp_dir, scheduler)
    
    logger.info("Experiment finished.")

if __name__ == "__main__":
    main()
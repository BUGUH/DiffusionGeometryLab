"""
评估入口脚本
============
加载 model_gen.pt 和 model_mem.pt，运行全部指标分析。
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.utils.config import load_config, validate_config
from src.utils.seed import set_global_seed
from src.utils.device import DeviceManager
from src.data.dataset import build_dataloaders
from src.data.semantic_pairs import build_semantic_pairs
from src.models.score_network import ScoreNetwork
from src.schedulers import build_scheduler
from src.metrics.evaluator import UnifiedEvaluator
from src.logging.experiment_logger import ExperimentLogger
from src.logging.plot_utils import (
    plot_cosine_field_curves,
    plot_memory_ratio_curve,
    plot_reconstruction_gap,
    figure_to_tensor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="含 model_gen.pt, model_mem.pt 的目录")
    parser.add_argument("--overrides", nargs="*",                        default=[])
    args = parser.parse_args()

    # 1. 配置
    cfg = load_config(args.config, overrides=args.overrides)
    cfg = validate_config(cfg)
    set_global_seed(cfg.experiment.seed)
    dm = DeviceManager(cfg.device)

    # 2. 数据
    train_loader, test_loader, padder = build_dataloaders(cfg)
    train_data = train_loader.dataset.data
    test_data = test_loader.dataset.data

    # 3. 语义配对
    print("构建语义配对...")
    pairs = build_semantic_pairs(train_data, test_data, num_pairs=500)

    # 4. 调度器
    scheduler = build_scheduler(cfg)

    # 5. 加载两个模型
    ckpt_dir = Path(args.ckpt_dir)

    def load_model(tag: str) -> ScoreNetwork:
        model = ScoreNetwork(cfg)
        ckpt_path = ckpt_dir / f"model_{tag}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到 {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=dm.device)
        model.load_state_dict_custom(ckpt["model"])
        model.to(dm.device)
        model.eval()
        print(f"已加载 model_{tag} (step={ckpt.get('step', '?')})")
        return model

    model_gen = load_model("gen")
    model_mem = load_model("mem")

    # 6. 日志
    exp_logger = ExperimentLogger(cfg.logging, cfg.experiment)

    # 7. 统一评估器
    evaluator = UnifiedEvaluator(
        cfg=cfg,
        scheduler=scheduler,
        train_data=train_data,
        test_data=test_data,
        semantic_pairs=pairs,
    )

    # 8. 运行评估
    save_dir = Path(cfg.experiment.output_dir) / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 8a. 余弦场分析 ----
    print("\n" + "=" * 50)
    print("余弦相似度场分析")
    print("=" * 50)

    cosine_results = evaluator.evaluate_cosine_fields(model_gen, model_mem, dm.device)

    # 绘图
    fig = plot_cosine_field_curves(
        {
            "$S_{cos}(M_{gen}, M_{mem})$": cosine_results["gen_mem"],
            "$S_{cos}(M_{gen}, M_{emp})$": cosine_results["gen_empirical"],
            "$S_{cos}(M_{mem}, M_{emp})$": cosine_results["mem_empirical"],
        },
        title=f"Cosine Similarity Fields (d={cfg.data.target_dim})",
        save_path=save_dir / "cosine_fields.png",
    )
    exp_logger.log_image("eval/cosine_fields", figure_to_tensor(fig), step=0)

    if "train_test_gen" in cosine_results:
        fig2 = plot_cosine_field_curves(
            {
                "$S_{cos}(train, test)$ — Gen": cosine_results["train_test_gen"],
                "$S_{cos}(train, test)$ — Mem": cosine_results["train_test_mem"],
            },
            title=f"Train-Test Cosine Similarity (d={cfg.data.target_dim})",
            save_path=save_dir / "cosine_train_test.png",
        )
        exp_logger.log_image("eval/cosine_train_test", figure_to_tensor(fig2), step=0)

    # ---- 8b. 各模型独立评估 ----
    for tag, model in [("gen", model_gen), ("mem", model_mem)]:
        print(f"\n评估 M_{tag}...")
        metrics = evaluator.evaluate(model, scheduler, dm, step=0)
        prefixed = {f"eval_{tag}/{k}": v for k, v in metrics.items()}
        exp_logger.log_scalars(prefixed, step=0)
        print(f"  M_{tag} 指标: {metrics}")

    # ---- 8c. 重建差距 ----
    print("\n重建差距分析...")
    from src.metrics.reconstruction_gap import ReconstructionGapEvaluator
    recon_eval = ReconstructionGapEvaluator(
        scheduler=scheduler,
        timesteps=list(cfg.metrics.reconstruction_gap.timesteps),
    )
    for tag, model in [("gen", model_gen), ("mem", model_mem)]:
        recon = recon_eval.evaluate(model, train_loader, test_loader, dm.device)
        fig_recon = plot_reconstruction_gap(
            recon["timesteps"], recon["train_mse"], recon["test_mse"],
            save_path=save_dir / f"recon_gap_{tag}.png",
        )
        exp_logger.log_image(f"eval/recon_gap_{tag}", figure_to_tensor(fig_recon), step=0)
        print(f"  M_{tag} 最大 gap: {recon['gap'].max():.4f}")

    # ---- 8d. 汇总 ----
    print("\n" + "=" * 50)
    print(f"所有结果已保存至: {save_dir}")
    print("=" * 50)

    exp_logger.finish()


if __name__ == "__main__":
    main()
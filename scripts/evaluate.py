"""评估入口脚本。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
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
    plot_cosine_field_curves, plot_reconstruction_gap, figure_to_tensor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    cfg = validate_config(cfg)
    set_global_seed(cfg.experiment.seed)
    dm = DeviceManager(cfg.device)

    train_loader, test_loader, padder = build_dataloaders(cfg)
    train_data = train_loader.dataset.data
    test_data = test_loader.dataset.data

    print("构建语义配对...")
    pairs = build_semantic_pairs(train_data, test_data, num_pairs=500)

    scheduler = build_scheduler(cfg)
    ckpt_dir = Path(args.ckpt_dir)

    def load_model(tag):
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

    exp_logger = ExperimentLogger(cfg.logging, cfg.experiment)
    evaluator = UnifiedEvaluator(
        cfg=cfg, scheduler=scheduler,
        train_data=train_data, test_data=test_data,
        semantic_pairs=pairs,
    )

    save_dir = Path(cfg.experiment.output_dir) / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 余弦场
    print("\n余弦相似度场分析...")
    cosine_results = evaluator.evaluate_cosine_fields(model_gen, model_mem, dm.device)
    fig = plot_cosine_field_curves(
        {
            "$S_{cos}(M_{gen}, M_{mem})$": cosine_results["gen_mem"],
            "$S_{cos}(M_{gen}, M_{emp})$": cosine_results["gen_empirical"],
            "$S_{cos}(M_{mem}, M_{emp})$": cosine_results["mem_empirical"],
        },
        title=f"Cosine Similarity Fields (d={cfg.data.target_dim})",
        save_path=save_dir / "cosine_fields.png",
    )

    # 各模型评估
    for tag, model in [("gen", model_gen), ("mem", model_mem)]:
        print(f"\n评估 M_{tag}...")
        metrics = evaluator.evaluate(model, scheduler, dm, step=0)
        print(f"  M_{tag}: {metrics}")

    # 重建差距
    from src.metrics.reconstruction_gap import ReconstructionGapEvaluator
    recon_eval = ReconstructionGapEvaluator(
        scheduler=scheduler,
        timesteps=list(cfg.metrics.reconstruction_gap.timesteps),
    )
    for tag, model in [("gen", model_gen), ("mem", model_mem)]:
        recon = recon_eval.evaluate(model, train_loader, test_loader, dm.device)
        plot_reconstruction_gap(
            recon["timesteps"], recon["train_mse"], recon["test_mse"],
            save_path=save_dir / f"recon_gap_{tag}.png",
        )

    print(f"\n所有结果已保存至: {save_dir}")
    exp_logger.finish()


if __name__ == "__main__":
    main()
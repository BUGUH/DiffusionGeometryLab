"""训练入口脚本。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
from omegaconf import OmegaConf

from src.utils.config import load_config, merge_configs, validate_config, pretty_print_config
from src.utils.seed import set_global_seed
from src.utils.device import DeviceManager
from src.data.dataset import build_dataloaders
from src.models.score_network import ScoreNetwork
from src.schedulers import build_scheduler
from src.training.trainer import DiffusionTrainer
from src.logging.experiment_logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="DiffusionGeometryLab Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    cfg = validate_config(cfg)
    print(pretty_print_config(cfg))

    set_global_seed(cfg.experiment.seed)
    dm = DeviceManager(cfg.device)
    print(f"设备: {dm.device}, 混合精度: {dm.is_mixed_precision}")

    train_loader, test_loader, padder = build_dataloaders(cfg)

    model = ScoreNetwork(cfg)
    scheduler = build_scheduler(cfg)

    exp_logger = ExperimentLogger(cfg.logging, cfg.experiment)
    exp_logger.log_config(cfg)

    trainer = DiffusionTrainer(
        cfg=cfg, model=model, scheduler=scheduler,
        train_loader=train_loader, test_loader=test_loader,
        device_manager=dm, experiment_logger=exp_logger,
    )

    summary = trainer.train()
    print(f"\n训练摘要: {summary}")
    exp_logger.finish()


if __name__ == "__main__":
    main()
"""轨迹可视化脚本。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import torch

from src.utils.config import load_config, validate_config
from src.utils.seed import set_global_seed
from src.utils.device import DeviceManager
from src.data.dataset import build_dataloaders
from src.models.score_network import ScoreNetwork
from src.schedulers import build_scheduler
from src.sampling.reverse_sampler import ReverseSampler
from src.logging.plot_utils import plot_trajectory_snapshots, plot_fid_trajectory
from src.metrics.fid_evaluator import FIDEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    cfg = validate_config(cfg)
    set_global_seed(cfg.experiment.seed)
    dm = DeviceManager(cfg.device)

    train_loader, _, padder = build_dataloaders(cfg)
    train_data = train_loader.dataset.data

    model = ScoreNetwork(cfg)
    ckpt = torch.load(args.checkpoint, map_location=dm.device)
    model.load_state_dict_custom(ckpt["model"])
    model.to(dm.device)
    model.eval()

    scheduler = build_scheduler(cfg)
    sampler = ReverseSampler(
        model=model, scheduler=scheduler,
        method=cfg.sampling.get("method", "ddim"),
        num_steps=cfg.sampling.get("num_sampling_steps", 100),
        capture_timesteps=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
    )

    sample_shape = (args.num_samples,) + tuple(train_data.shape[1:])
    print(f"采样 {args.num_samples} 个样本, shape={sample_shape}...")
    generated, bundle = sampler.sample(sample_shape, dm.device)

    save_dir = Path(cfg.experiment.output_dir) / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectory_snapshots(
        bundle.x_ts, num_samples=min(8, args.num_samples),
        save_path=save_dir / "trajectory_snapshots.png",
    )
    print(f"保存: {save_dir / 'trajectory_snapshots.png'}")

    fid_eval = FIDEvaluator()
    traj_fid = fid_eval.compute_trajectory_fid(train_data, bundle, scheduler)
    if traj_fid:
        ts = np.array(sorted(traj_fid.keys()))
        fids = np.array([traj_fid[t] for t in ts])
        plot_fid_trajectory(ts, fids, save_path=save_dir / "trajectory_fid.png")
        print(f"保存: {save_dir / 'trajectory_fid.png'}")

    print(f"\n所有可视化保存至: {save_dir}")


if __name__ == "__main__":
    main()
"""PyTorch Dataset/DataLoader 封装 — 支持 CelebA / 合成图像 / Circle 向量。"""

from __future__ import annotations
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.zero_padding import ZeroPadder
from src.utils.seed import worker_init_fn

logger = logging.getLogger(__name__)


class DiffusionDataset(Dataset):
    """通用扩散数据集。"""
    def __init__(self, data: torch.Tensor, indices: Optional[np.ndarray] = None):
        self.data = data
        self.indices = indices if indices is not None else np.arange(len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], int(self.indices[idx])


# 向后兼容别名
CelebADiffusionDataset = DiffusionDataset


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, Optional[ZeroPadder]]:
    """
    统一数据加载入口。

    parquet_path 标记:
      - "__circle__"    → Circle 向量数据
      - "__synthetic__" → 合成图像数据（高斯斑点编码）
      - 其他            → 真实 Parquet 数据

    Returns
    -------
    (train_loader, test_loader, padder_or_None)
    """
    source = str(cfg.data.parquet_path).strip()
    logger.info(f"数据源: '{source}'")

    # ============================================================
    # Circle 向量数据
    # ============================================================
    if source == "__circle__":
        from src.data.synthetic_circle import build_circle_data
        tensor_data, raw_points = build_circle_data(cfg)

        out_dir = str(cfg.experiment.get("output_dir", "outputs"))
        os.makedirs(out_dir, exist_ok=True)
        torch.save(raw_points, os.path.join(out_dir, "raw_points.pt"))

        padder = None  # Circle 数据不需要零填充（已经在 D 维空间）
        padded_data = tensor_data

    # ============================================================
    # 合成图像数据
    # ============================================================
    elif source == "__synthetic__":
        from src.data.synthetic import build_synthetic_data
        images, points = build_synthetic_data(cfg)

        out_dir = str(cfg.experiment.get("output_dir", "outputs"))
        os.makedirs(out_dir, exist_ok=True)
        torch.save(points, os.path.join(out_dir, "raw_points.pt"))

        padder = ZeroPadder(
            source_size=cfg.data.grayscale_size,
            target_size=cfg.data.target_resolution,
            mode=cfg.data.get("padding_mode", "center"),
        )
        padded_data = padder.pad(images)

    # ============================================================
    # 真实 Parquet 数据
    # ============================================================
    else:
        from src.data.parquet_loader import load_parquet_images
        from src.data.preprocessing import preprocess_images

        raw_images = load_parquet_images(
            source, image_column=cfg.data.get("image_column", "image"),
        )
        processed = preprocess_images(
            raw_images,
            target_size=cfg.data.grayscale_size,
            normalize_range=tuple(cfg.data.normalize_range),
        )
        tensor_data = torch.from_numpy(processed)

        padder = ZeroPadder(
            source_size=cfg.data.grayscale_size,
            target_size=cfg.data.target_resolution,
            mode=cfg.data.get("padding_mode", "center"),
        )
        padded_data = padder.pad(tensor_data)

    # ============================================================
    # 通用：分割 + DataLoader
    # ============================================================
    N = len(padded_data)
    n_train = int(N * cfg.data.train_ratio)
    perm = torch.randperm(N)

    train_ds = DiffusionDataset(padded_data[perm[:n_train]], perm[:n_train].numpy())
    test_ds = DiffusionDataset(padded_data[perm[n_train:]], perm[n_train:].numpy())
    logger.info(f"数据: {padded_data.shape}, 训练={len(train_ds)}, 测试={len(test_ds)}")

    nw = cfg.data.get("num_workers", 0)
    kw = dict(
        num_workers=nw,
        pin_memory=cfg.data.get("pin_memory", False),
        worker_init_fn=worker_init_fn if nw > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, drop_last=True, **kw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size,
        shuffle=False, drop_last=False, **kw,
    )

    return train_loader, test_loader, padder
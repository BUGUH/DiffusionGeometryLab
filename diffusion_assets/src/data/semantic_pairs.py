"""构建语义相似的训练/测试样本对（用于 S_cos(train, test) 计算）。"""

from __future__ import annotations
import logging
from typing import List, Tuple

import torch
from src.utils.math_ops import pairwise_l2_distance_chunked

logger = logging.getLogger(__name__)


def build_semantic_pairs(
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    num_pairs: int = 500,
    chunk_size: int = 512,
) -> List[Tuple[int, int]]:
    """
    为每个测试样本在训练集中找像素空间最近邻。

    Returns
    -------
    list[(train_idx, test_idx)]
    """
    N_test = min(num_pairs, len(test_data))
    train_flat = train_data.view(len(train_data), -1).float()
    test_flat = test_data[:N_test].view(N_test, -1).float()

    dists = pairwise_l2_distance_chunked(test_flat, train_flat, chunk_size=chunk_size)
    nn_indices = dists.argmin(dim=1)  # (N_test,)

    pairs = [(int(nn_indices[i]), i) for i in range(N_test)]
    logger.info(f"构建 {len(pairs)} 个语义配对")
    return pairs
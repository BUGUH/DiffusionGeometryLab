"""Faiss 最近邻索引管理。"""

from __future__ import annotations
import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FaissNNIndex:
    """
    Faiss 高效最近邻搜索封装。

    支持:
    - L2 距离索引
    - 加噪版本索引（对训练集在指定 t 加噪后建索引）
    - GPU 加速（可选）
    """

    def __init__(self, dim: int, use_gpu: bool = False, nprobe: int = 32):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("需要 faiss: pip install faiss-cpu 或 faiss-gpu")

        self.dim = dim
        self.use_gpu = use_gpu
        self.nprobe = nprobe
        self._index = None
        self._data = None

    def build(self, data: torch.Tensor):
        """
        构建索引。

        Parameters
        ----------
        data : (N, D) 展平后的数据向量
        """
        arr = data.detach().cpu().numpy().astype(np.float32)
        N = arr.shape[0]
        self._data = arr

        if N < 4096:
            # 小数据集用暴力搜索
            self._index = self.faiss.IndexFlatL2(self.dim)
        else:
            # IVF 索引加速
            nlist = min(int(np.sqrt(N)), 256)
            quantizer = self.faiss.IndexFlatL2(self.dim)
            self._index = self.faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self._index.train(arr)
            self._index.nprobe = self.nprobe

        self._index.add(arr)
        logger.debug(f"Faiss 索引已构建: N={N}, dim={self.dim}")

    def build_noisy(self, data: torch.Tensor, t: float, scheduler):
        """对数据加噪后建索引。"""
        N = data.shape[0]
        t_tensor = torch.full((N,), t)
        noisy_data, _ = scheduler.add_noise(data, t_tensor)
        flat = noisy_data.view(N, -1)
        self.build(flat)

    def query(self, queries: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查询 k 近邻。

        Returns
        -------
        (distances, indices): 均为 (N_query, k)
        """
        q = queries.detach().cpu().numpy().astype(np.float32)
        distances, indices = self._index.search(q, k)
        return (
            torch.from_numpy(distances).float(),
            torch.from_numpy(indices).long(),
        )

    def query_knn_distances(self, queries: torch.Tensor, k: int = 2) -> torch.Tensor:
        """仅返回距离。"""
        dists, _ = self.query(queries, k)
        return dists
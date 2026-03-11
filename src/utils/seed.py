"""
随机种子管理模块
================
确保实验可复现性。统一管理 Python、NumPy、PyTorch、CUDA 的随机状态。

典型用法:
    set_global_seed(42)
    rng = get_rng(42)  # 独立随机生成器，不影响全局状态
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    设置全局随机种子，确保可复现性。

    Parameters
    ----------
    seed : int
        随机种子值。
    deterministic : bool, default True
        是否启用 PyTorch 确定性模式。
        注意：确定性模式可能降低训练速度 10-20%。

    Side Effects
    ------------
    - 设置 Python random, numpy, torch 的种子。
    - 设置 CUDA 种子（如果可用）。
    - 设置 PYTHONHASHSEED 环境变量。
    - 可选启用 torch 确定性模式。
    """
    # Python 内置随机
    random.seed(seed)

    # 环境变量（影响 dict、set 等的哈希随机化）
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU

    # 确定性模式
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # 旧版本 PyTorch 不支持 warn_only
                torch.use_deterministic_algorithms(True)
    else:
        # 非确定性模式：启用 cudnn benchmark 加速
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_rng(seed: int) -> torch.Generator:
    """
    创建独立的 PyTorch 随机生成器，不影响全局状态。

    适用场景:
    - 数据加载器的 worker 随机种子
    - 需要固定噪声注入的评估流程（如余弦场分析中的"相同噪声"）

    Parameters
    ----------
    seed : int

    Returns
    -------
    torch.Generator
        独立的 CPU 随机生成器。
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_numpy_rng(seed: int) -> np.random.Generator:
    """
    创建独立的 NumPy 随机生成器（新式 API）。

    Parameters
    ----------
    seed : int

    Returns
    -------
    np.random.Generator
    """
    return np.random.default_rng(seed)


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader 的 worker_init_fn 回调。

    确保每个 worker 拥有不同但可复现的随机种子。
    使用方法:
        DataLoader(..., worker_init_fn=worker_init_fn)

    Parameters
    ----------
    worker_id : int
        DataLoader 分配的 worker 编号。
    """
    # 获取当前 DataLoader 的基础种子
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
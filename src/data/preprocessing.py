"""图像预处理：灰度化 + resize + 归一化。"""

from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_images(
    images: List[np.ndarray],
    target_size: int = 32,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    批量预处理：RGB → 灰度 → resize → 归一化。

    Returns
    -------
    np.ndarray, shape (N, 1, target_size, target_size), dtype float32
    """
    processed = []
    lo, hi = normalize_range

    for img_np in images:
        img = Image.fromarray(img_np).convert("L")  # 灰度
        img = img.resize((target_size, target_size), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        arr = arr * (hi - lo) + lo  # 映射到目标范围
        processed.append(arr[np.newaxis])  # (1, H, W)

    result = np.stack(processed, axis=0)  # (N, 1, H, W)
    logger.info(f"预处理完成: shape={result.shape}, range=[{result.min():.2f}, {result.max():.2f}]")
    return result
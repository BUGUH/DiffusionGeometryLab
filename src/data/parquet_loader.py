"""从 Parquet 文件加载 CelebA 图像。"""

from __future__ import annotations
import io
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_parquet_images(
    parquet_path: str,
    image_column: str = "image",
    max_samples: Optional[int] = None,
) -> List[np.ndarray]:
    """
    读取 parquet 文件中的图像数据。

    支持两种存储格式:
    1. bytes 列（图像编码为 PNG/JPEG bytes）
    2. HuggingFace datasets 格式（嵌套 dict: {"bytes": ..., "path": ...}）

    Returns
    -------
    list[np.ndarray]
        每个元素为 (H, W, 3) uint8 RGB 图像。
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要 pandas: pip install pandas pyarrow")

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {path.resolve()}")

    logger.info(f"加载 Parquet: {path}")
    df = pd.read_parquet(path)

    if image_column not in df.columns:
        # 尝试常见列名
        candidates = ["image", "img", "pixel_values", df.columns[0]]
        image_column = next((c for c in candidates if c in df.columns), df.columns[0])
        logger.warning(f"未找到指定列，使用: '{image_column}'")

    if max_samples:
        df = df.head(max_samples)

    images = []
    for idx, row in enumerate(df[image_column]):
        try:
            img = _decode_image(row)
            if img is not None:
                images.append(img)
        except Exception as e:
            if idx < 3:
                logger.warning(f"跳过第 {idx} 张图像: {e}")

    logger.info(f"成功加载 {len(images)} 张图像")
    return images


def _decode_image(raw) -> Optional[np.ndarray]:
    """解码单张图像，支持多种格式。"""
    img_bytes = None

    if isinstance(raw, bytes):
        img_bytes = raw
    elif isinstance(raw, dict):
        img_bytes = raw.get("bytes", None)
    elif isinstance(raw, np.ndarray):
        if raw.ndim == 1:  # bytes array
            img_bytes = raw.tobytes()
        else:
            return raw if raw.ndim == 3 else None

    if img_bytes is not None:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(img)

    return None
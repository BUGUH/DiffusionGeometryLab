"""图像预处理：灰度化 + 中心裁剪(可选) + resize + 归一化。"""
from __future__ import annotations
import logging
from typing import List, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def preprocess_images(
    images: List[np.ndarray],
    target_size: int = 64,
    center_crop: bool = True,  # ★ 新增：默认开启中心裁剪，保护几何流形不被拉伸变形
    crop_size: int = 140,      # ★ 新增：CelebA 业内标准裁剪尺寸
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    批量预处理：RGB → 灰度 → (中心裁剪) → resize → 归一化。
    
    Returns
    -------
    np.ndarray, shape (N, 1, target_size, target_size), dtype float32
    """
    processed = []
    lo, hi = normalize_range
    for img_np in images:
        # 1. 灰度化
        img = Image.fromarray(img_np).convert("L")  
        
        # 2. ★ 新增逻辑：中心裁剪 (Center Crop) ★
        if center_crop:
            width, height = img.size
            # 计算裁剪框的左上角和右下角坐标
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = (width + crop_size) / 2
            bottom = (height + crop_size) / 2
            img = img.crop((left, top, right, bottom))
            
        # 3. 缩放到目标大小 (比如 64x64 或 128x128)
        img = img.resize((target_size, target_size), Image.BICUBIC)
        
        # 4. 数值归一化
        arr = np.array(img, dtype=np.float32) / 255.0  # 先映射到 [0, 1]
        arr = arr * (hi - lo) + lo                     # 再映射到目标范围 (默认 [-1, 1])
        
        processed.append(arr[np.newaxis])  # 增加通道维度变为 (1, H, W)
        
    result = np.stack(processed, axis=0)  # 拼接成批次 (N, 1, H, W)
    
    # 记录日志，加入了 crop 的状态显示
    logger.info(f"预处理完成: shape={result.shape}, crop={center_crop}, range=[{result.min():.2f}, {result.max():.2f}]")
    
    return result

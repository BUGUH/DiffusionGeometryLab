"""
设备管理模块
============
统一管理 GPU/CPU/MPS 设备选择与混合精度上下文。

典型用法:
    dm = DeviceManager(cfg.device)
    model = model.to(dm.device)
    with dm.autocast():
        loss = model(x)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """设备信息快照，用于日志记录。"""
    device_type: str           # "cuda", "cpu", "mps"
    device_name: str           # e.g. "NVIDIA A100-SXM4-80GB"
    device_count: int          # 可用 GPU 数量
    memory_total_gb: float     # 单卡显存 (GB)
    mixed_precision: str       # "fp16", "bf16", "none"
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None


class DeviceManager:
    """
    设备与混合精度管理器。

    Parameters
    ----------
    cfg : DictConfig
        包含 device.accelerator, device.gpu_ids, device.mixed_precision 的配置。

    Attributes
    ----------
    device : torch.device
        主计算设备。
    dtype : torch.dtype
        混合精度对应的数据类型。
    """

    def __init__(self, cfg: Union[DictConfig, dict]) -> None:
        if isinstance(cfg, dict):
            from omegaconf import OmegaConf
            cfg = OmegaConf.create(cfg)

        self._accelerator: str = cfg.get("accelerator", "auto")
        self._gpu_ids: List[int] = list(cfg.get("gpu_ids", [0]))
        self._mixed_precision: str = cfg.get("mixed_precision", "none")

        # 解析设备
        self._device = self._resolve_device()
        self._dtype = self._resolve_dtype()
        self._scaler = self._create_scaler()

        logger.info(f"DeviceManager 初始化完成: device={self._device}, "
                     f"mixed_precision={self._mixed_precision}")

    # ---- 公开属性 ----

    @property
    def device(self) -> torch.device:
        """主计算设备。"""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """混合精度数据类型。"""
        return self._dtype

    @property
    def scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """GradScaler 实例（仅 fp16 + CUDA 时非 None）。"""
        return self._scaler

    @property
    def is_cuda(self) -> bool:
        return self._device.type == "cuda"

    @property
    def is_mixed_precision(self) -> bool:
        return self._mixed_precision != "none"

    # ---- 核心方法 ----

    @contextmanager
    def autocast(self):
        """
        混合精度自动转型上下文管理器。

        用法:
            with dm.autocast():
                output = model(input)
                loss = criterion(output, target)
        """
        if self._mixed_precision == "none":
            yield
        else:
            device_type = "cuda" if self.is_cuda else "cpu"
            with torch.autocast(device_type=device_type, dtype=self._dtype):
                yield

    def to_device(self, x: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
        """将张量或模型移动到主设备。"""
        return x.to(self._device)

    def get_info(self) -> DeviceInfo:
        """获取当前设备信息快照。"""
        if self.is_cuda:
            props = torch.cuda.get_device_properties(self._device)
            return DeviceInfo(
                device_type="cuda",
                device_name=props.name,
                device_count=torch.cuda.device_count(),
                memory_total_gb=round(props.total_mem / (1024**3), 2),
                mixed_precision=self._mixed_precision,
                cuda_version=torch.version.cuda,
                cudnn_version=str(torch.backends.cudnn.version()),
            )
        else:
            return DeviceInfo(
                device_type=self._device.type,
                device_name=self._device.type.upper(),
                device_count=0,
                memory_total_gb=0.0,
                mixed_precision=self._mixed_precision,
            )

    def empty_cache(self) -> None:
        """清空 GPU 缓存。"""
        if self.is_cuda:
            torch.cuda.empty_cache()

    def memory_summary(self) -> str:
        """返回 GPU 显存使用摘要。"""
        if self.is_cuda:
            allocated = torch.cuda.memory_allocated(self._device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self._device) / (1024**3)
            return (f"GPU Memory — Allocated: {allocated:.2f} GB, "
                    f"Reserved: {reserved:.2f} GB")
        return "非 CUDA 设备，无显存统计。"

    # ---- 私有方法 ----

    def _resolve_device(self) -> torch.device:
        """根据 accelerator 配置解析实际设备。"""
        acc = self._accelerator.lower()

        if acc == "auto":
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self._gpu_ids[0]}")
                logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(device)}")
                return device
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Auto-detected Apple MPS device.")
                return torch.device("mps")
            else:
                logger.info("No GPU found, falling back to CPU.")
                return torch.device("cpu")

        elif acc == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("配置要求 CUDA，但当前环境无可用 GPU。")
            return torch.device(f"cuda:{self._gpu_ids[0]}")

        elif acc == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("配置要求 MPS，但当前环境不支持。")
            return torch.device("mps")

        elif acc == "cpu":
            return torch.device("cpu")

        else:
            raise ValueError(f"未知的 accelerator: '{acc}'")

    def _resolve_dtype(self) -> torch.dtype:
        """解析混合精度数据类型。"""
        mp = self._mixed_precision.lower()
        if mp == "fp16":
            return torch.float16
        elif mp == "bf16":
            if self.is_cuda:
                # 检查 bf16 支持（Ampere+ 架构）
                if not torch.cuda.is_bf16_supported():
                    logger.warning("当前 GPU 不支持 bf16，回退到 fp16。")
                    self._mixed_precision = "fp16"
                    return torch.float16
            return torch.bfloat16
        elif mp == "none":
            return torch.float32
        else:
            raise ValueError(f"未知的 mixed_precision: '{mp}'")

    def _create_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """创建 GradScaler（仅 fp16 + CUDA）。"""
        if self._mixed_precision == "fp16" and self.is_cuda:
            return torch.cuda.amp.GradScaler()
        return None
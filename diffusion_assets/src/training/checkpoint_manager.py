"""Checkpoint 保存/加载/管理。"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    管理训练 Checkpoint 的存储与加载。

    存储结构:
        output_dir/
          checkpoints/
            step_10000.pt
            step_20000.pt
            model_gen.pt
            best_fid.pt
            latest.pt
    """

    def __init__(self, output_dir: str, max_keep: int = 5):
        self.ckpt_dir = Path(output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self._periodic_ckpts = []  # 追踪周期性 checkpoint 用于清理

    def save(
        self,
        tag: str,
        model_state: dict,
        optimizer_state: dict,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """保存 checkpoint。"""
        payload = {
            "model": model_state,
            "optimizer": optimizer_state,
            "step": step,
            "metadata": metadata or {},
        }
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(payload, path)
        logger.info(f"Checkpoint 已保存: {path} (step={step})")
        return path

    def save_periodic(self, model_state, optimizer_state, step, metadata=None) -> Path:
        """周期性保存，自动清理旧文件。"""
        path = self.save(f"step_{step}", model_state, optimizer_state, step, metadata)
        self._periodic_ckpts.append(path)
        # 清理
        while len(self._periodic_ckpts) > self.max_keep:
            old = self._periodic_ckpts.pop(0)
            if old.exists():
                old.unlink()
                logger.debug(f"清理旧 checkpoint: {old}")
        # 同时保存 latest
        self.save("latest", model_state, optimizer_state, step, metadata)
        return path

    def save_phase(self, phase: str, model_state, optimizer_state, step, metadata=None) -> Path:
        """保存相变 checkpoint: model_gen.pt 或 model_mem.pt"""
        tag = f"model_{phase}"
        return self.save(tag, model_state, optimizer_state, step, metadata)

    def load(self, tag: str, device: str = "cpu") -> Dict[str, Any]:
        """加载 checkpoint。"""
        path = self.ckpt_dir / f"{tag}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {path}")
        ckpt = torch.load(path, map_location=device)
        logger.info(f"Checkpoint 已加载: {path} (step={ckpt.get('step', '?')})")
        return ckpt

    def load_latest(self, device="cpu") -> Optional[Dict[str, Any]]:
        """加载最新 checkpoint，不存在返回 None。"""
        path = self.ckpt_dir / "latest.pt"
        if path.exists():
            return self.load("latest", device)
        return None

    def exists(self, tag: str) -> bool:
        return (self.ckpt_dir / f"{tag}.pt").exists()
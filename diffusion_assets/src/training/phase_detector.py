"""
相变自动检测器
==============
实时监控训练指标，自动判定 M_gen（泛化）和 M_mem（记忆）状态。
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PhaseType(Enum):
    NONE = auto()
    GEN_DETECTED = auto()
    MEM_DETECTED = auto()


@dataclass
class PhaseEvent:
    """相变事件。"""
    phase: PhaseType
    step: int
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def is_gen(self) -> bool:
        return self.phase == PhaseType.GEN_DETECTED

    @property
    def is_mem(self) -> bool:
        return self.phase == PhaseType.MEM_DETECTED

    @property
    def detected(self) -> bool:
        return self.phase != PhaseType.NONE


class PhaseDetector:
    """
    M_gen / M_mem 自动检测器。

    M_gen 判定:
    - 测试集 FID 达到全局最低（滑动窗口内无改善则锁定）
    - 训练/验证损失未显著发散

    M_mem 判定:
    - 训练步数 τ >> τ_gen
    - 重建误差差距 (test_mse - train_mse) / train_mse > 阈值
    - 生成样本与训练集 LPIPS 极低（过拟合）
    """

    def __init__(self, cfg):
        pd_cfg = cfg.training.phase_detection
        self.enabled = pd_cfg.get("enabled", True)

        # M_gen 参数
        self.fid_patience = pd_cfg.get("fid_patience", 3)
        self.fid_window_size = pd_cfg.get("fid_window_size", 5)

        # M_mem 参数
        self.recon_gap_threshold = pd_cfg.get("recon_gap_threshold", 0.3)
        self.lpips_threshold = pd_cfg.get("lpips_threshold", 0.05)
        self.mem_step_multiplier = pd_cfg.get("mem_step_multiplier", 3.0)

        # 内部状态
        self._fid_history: deque = deque(maxlen=self.fid_window_size)
        self._best_fid: float = float("inf")
        self._best_fid_step: int = 0
        self._no_improve_count: int = 0
        self._gen_step: Optional[int] = None  # τ_gen
        self._gen_locked: bool = False
        self._mem_locked: bool = False

    def update(self, step: int, metrics: Dict[str, float]) -> PhaseEvent:
        """
        输入当前评估指标，返回相变事件。

        Parameters
        ----------
        step : int
        metrics : dict，可能包含:
            - "fid": FID 分数
            - "train_mse": 训练集重建 MSE
            - "test_mse": 测试集重建 MSE
            - "lpips": 生成样本与训练集最近邻的平均 LPIPS
            - "train_loss": 训练损失
            - "val_loss": 验证损失

        Returns
        -------
        PhaseEvent
        """
        if not self.enabled:
            return PhaseEvent(PhaseType.NONE, step)

        # 检测 M_gen
        gen_event = self._check_gen(step, metrics)
        if gen_event.detected:
            return gen_event

        # 检测 M_mem（必须在 gen 之后）
        mem_event = self._check_mem(step, metrics)
        if mem_event.detected:
            return mem_event

        return PhaseEvent(PhaseType.NONE, step, metrics)

    def _check_gen(self, step: int, metrics: Dict[str, float]) -> PhaseEvent:
        """检测泛化状态 M_gen。"""
        if self._gen_locked:
            return PhaseEvent(PhaseType.NONE, step)

        fid = metrics.get("fid", None)
        if fid is None:
            return PhaseEvent(PhaseType.NONE, step)

        self._fid_history.append(fid)

        # 检查是否是全局最低
        if fid < self._best_fid:
            self._best_fid = fid
            self._best_fid_step = step
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        # 损失未发散检查
        train_loss = metrics.get("train_loss", 0)
        val_loss = metrics.get("val_loss", train_loss)
        loss_diverged = val_loss > train_loss * 2.0 if train_loss > 0 else False

        # FID 连续 patience 次未改善 → 锁定 gen
        if self._no_improve_count >= self.fid_patience and not loss_diverged:
            self._gen_locked = True
            self._gen_step = self._best_fid_step
            logger.info(
                f"🟢 M_gen 检测到 @ step={self._gen_step}, "
                f"best_fid={self._best_fid:.2f}"
            )
            return PhaseEvent(
                PhaseType.GEN_DETECTED,
                self._gen_step,
                {"fid": self._best_fid, "actual_step": step},
            )

        return PhaseEvent(PhaseType.NONE, step)

    def _check_mem(self, step: int, metrics: Dict[str, float]) -> PhaseEvent:
        """检测记忆状态 M_mem。"""
        if self._mem_locked or self._gen_step is None:
            return PhaseEvent(PhaseType.NONE, step)

        # 条件 1: τ >> τ_gen
        if step < self._gen_step * self.mem_step_multiplier:
            return PhaseEvent(PhaseType.NONE, step)

        # 条件 2: 重建误差差距
        train_mse = metrics.get("train_mse", None)
        test_mse = metrics.get("test_mse", None)
        gap_satisfied = False
        recon_gap = 0.0
        if train_mse is not None and test_mse is not None and train_mse > 1e-10:
            recon_gap = (test_mse - train_mse) / train_mse
            gap_satisfied = recon_gap > self.recon_gap_threshold

        # 条件 3: LPIPS 极低
        lpips = metrics.get("lpips", None)
        lpips_satisfied = lpips is not None and lpips < self.lpips_threshold

        # 满足任一过拟合指标即可
        if gap_satisfied or lpips_satisfied:
            self._mem_locked = True
            logger.info(
                f"🔴 M_mem 检测到 @ step={step}, "
                f"recon_gap={recon_gap:.4f}, lpips={lpips}"
            )
            return PhaseEvent(
                PhaseType.MEM_DETECTED, step,
                {"recon_gap": recon_gap, "lpips": lpips or -1,
                 "gen_step": self._gen_step},
            )

        return PhaseEvent(PhaseType.NONE, step)

    @property
    def gen_detected(self) -> bool:
        return self._gen_locked

    @property
    def mem_detected(self) -> bool:
        return self._mem_locked

    @property
    def gen_step(self) -> Optional[int]:
        return self._gen_step

    @property
    def both_detected(self) -> bool:
        return self._gen_locked and self._mem_locked

    def state_dict(self) -> dict:
        return {
            "best_fid": self._best_fid,
            "best_fid_step": self._best_fid_step,
            "gen_step": self._gen_step,
            "gen_locked": self._gen_locked,
            "mem_locked": self._mem_locked,
            "no_improve_count": self._no_improve_count,
        }

    def load_state_dict(self, state: dict):
        self._best_fid = state["best_fid"]
        self._best_fid_step = state["best_fid_step"]
        self._gen_step = state["gen_step"]
        self._gen_locked = state["gen_locked"]
        self._mem_locked = state["mem_locked"]
        self._no_improve_count = state["no_improve_count"]
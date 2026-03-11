"""
核心训练循环
============
集成调度器、相变检测、checkpoint 管理、日志记录。
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.models.score_network import ScoreNetwork
from src.schedulers.base_scheduler import BaseScheduler
from src.training.checkpoint_manager import CheckpointManager
from src.training.phase_detector import PhaseDetector, PhaseType
from src.utils.device import DeviceManager

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """
    扩散模型训练器。

    功能:
    1. 标准去噪得分匹配训练
    2. 周期性评估 + 相变检测
    3. 自动保存 model_gen.pt / model_mem.pt
    4. 支持断点续训
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: ScoreNetwork,
        scheduler: BaseScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device_manager: DeviceManager,
        experiment_logger=None,
        evaluator=None,
    ):
        self.cfg = cfg
        self.model = model.to(device_manager.device)
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dm = device_manager
        self.exp_logger = experiment_logger
        self.evaluator = evaluator  # 外部评估器（FID、重建等）

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.backbone.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.get("weight_decay", 0.0),
        )

        # Checkpoint & 相变检测
        self.ckpt_mgr = CheckpointManager(
            cfg.experiment.output_dir, max_keep=5,
        )
        self.phase_detector = PhaseDetector(cfg)

        # 训练状态
        self.global_step = 0
        self.epoch = 0

        # 尝试恢复
        if cfg.experiment.get("resume_from"):
            self._resume(cfg.experiment.resume_from)

    def train(self) -> Dict[str, any]:
        """
        主训练循环。

        Returns
        -------
        dict: 训练摘要（最终 loss、相变步数等）
        """
        cfg_t = self.cfg.training
        max_steps = cfg_t.max_steps
        grad_clip = cfg_t.get("grad_clip_norm", 1.0)

        logger.info(f"开始训练: max_steps={max_steps}, device={self.dm.device}")
        train_iter = self._infinite_loader()

        while self.global_step < max_steps:
            self.model.train()
            step_start = time.time()

            # ---- 取数据 ----
            x_0, indices = next(train_iter)
            x_0 = x_0.to(self.dm.device)
            B = x_0.shape[0]

            # ---- 采样时间步 & 加噪            
            t = self.scheduler.sample_timesteps(B, device=self.dm.device)
            noise = torch.randn_like(x_0)
            x_t, noise = self.scheduler.add_noise(x_0, t, noise)

            # ---- 前向 + 损失 ----
            with self.dm.autocast():
                loss = self._compute_loss(x_t, t, noise, x_0)

            # ---- 反向传播 ----
            self.optimizer.zero_grad()
            if self.dm.scaler is not None:
                self.dm.scaler.scale(loss).backward()
                self.dm.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.backbone.parameters(), grad_clip
                )
                self.dm.scaler.step(self.optimizer)
                self.dm.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.backbone.parameters(), grad_clip
                )
                self.optimizer.step()

            # ---- EMA 更新 ----
            if self.cfg.training.get("use_ema", False):
                self.model.update_ema()

            self.global_step += 1
            step_time = time.time() - step_start

            # ---- 日志 ----
            if self.global_step % cfg_t.log_every_steps == 0:
                self._log_training(loss.item(), grad_norm.item(), step_time)

            # ---- 评估 + 相变检测 ----
            if self.global_step % cfg_t.eval_every_steps == 0:
                self._evaluate_and_detect()

            # ---- 周期保存 ----
            if self.global_step % cfg_t.save_every_steps == 0:
                self.ckpt_mgr.save_periodic(
                    self.model.get_state_dict(),
                    self.optimizer.state_dict(),
                    self.global_step,
                )

            # ---- 双相变都检测到则可提前停止 ----
            if self.phase_detector.both_detected:
                logger.info(f"M_gen 和 M_mem 均已检测到，训练完成 @ step={self.global_step}")
                break

        # 最终保存
        self.ckpt_mgr.save(
            "final", self.model.get_state_dict(),
            self.optimizer.state_dict(), self.global_step,
        )

        return self._build_summary()

    def _compute_loss(
        self, x_t: torch.Tensor, t: torch.Tensor,
        noise: torch.Tensor, x_0: torch.Tensor,
    ) -> torch.Tensor:
        """根据 output_type 计算去噪损失。"""
        output_type = self.cfg.model.output_type
        sched_type = self.cfg.scheduler.type

        if output_type == "noise":
            pred = self.model(x_t, t)
            loss = ((pred - noise) ** 2).mean()

        elif output_type == "score":
            pred = self.model(x_t, t)
            sigma = self.scheduler.get_sigma(t)
            sigma = self.scheduler._broadcast(sigma, x_t)
            target_score = -noise / sigma.clamp(min=1e-8)
            loss = ((pred - target_score) ** 2 * sigma ** 2).mean()

        elif output_type == "velocity":
            pred = self.model(x_t, t)
            # Flow Matching: target = ε - x_0
            if sched_type == "of":
                from src.schedulers.of_scheduler import OFScheduler
                target = noise - x_0
            else:
                # 通用 v-prediction: v = α'(t)*x_0 + σ'(t)*ε（简化版）
                mean_c, std = self.scheduler.marginal_params(t)
                mean_c = self.scheduler._broadcast(mean_c, x_0)
                std = self.scheduler._broadcast(std, x_0)
                target = mean_c * noise - std * x_0
            loss = ((pred - target) ** 2).mean()
        else:
            raise ValueError(f"未知 output_type: {output_type}")

        return loss

    @torch.no_grad()
    def _evaluate_and_detect(self):
        """周期性评估 + 相变检测。"""
        self.model.eval()
        metrics = {}

        # 1. 计算训练/测试重建误差
        train_mse = self._compute_reconstruction_mse(self.train_loader, max_batches=5)
        test_mse = self._compute_reconstruction_mse(self.test_loader, max_batches=5)
        metrics["train_mse"] = train_mse
        metrics["test_mse"] = test_mse
        metrics["recon_gap"] = (test_mse - train_mse) / max(train_mse, 1e-10)

        # 2. 外部评估器（FID、LPIPS 等）
        if self.evaluator is not None:
            ext_metrics = self.evaluator.evaluate(
                self.model, self.scheduler, self.dm, self.global_step,
            )
            metrics.update(ext_metrics)

        # 3. 训练/验证损失
        val_loss = self._compute_val_loss(max_batches=5)
        metrics["val_loss"] = val_loss

        # 4. 日志
        prefixed = {f"eval/{k}": v for k, v in metrics.items()}
        if self.exp_logger:
            self.exp_logger.log_scalars(prefixed, self.global_step)

        logger.info(
            f"[Step {self.global_step}] "
            f"train_mse={train_mse:.6f}, test_mse={test_mse:.6f}, "
            f"gap={metrics['recon_gap']:.4f}, "
            f"fid={metrics.get('fid', 'N/A')}"
        )

        # 5. 相变检测
        event = self.phase_detector.update(self.global_step, metrics)

        if event.is_gen:
            self.ckpt_mgr.save_phase(
                "gen", self.model.get_state_dict(),
                self.optimizer.state_dict(),
                event.step, event.metrics,
            )
            if self.exp_logger:
                self.exp_logger.log_phase_event("gen_detected", event.step, event.metrics)

        elif event.is_mem:
            self.ckpt_mgr.save_phase(
                "mem", self.model.get_state_dict(),
                self.optimizer.state_dict(),
                event.step, event.metrics,
            )
            if self.exp_logger:
                self.exp_logger.log_phase_event("mem_detected", event.step, event.metrics)

        self.model.train()

    @torch.no_grad()
    def _compute_reconstruction_mse(self, loader: DataLoader, max_batches: int = 5) -> float:
        """单步去噪重建 MSE: 在固定 t 处加噪再预测 x_0。"""
        total_mse, count = 0.0, 0
        t_eval = 0.5  # 中等噪声水平

        for i, (x_0, _) in enumerate(loader):
            if i >= max_batches:
                break
            x_0 = x_0.to(self.dm.device)
            B = x_0.shape[0]
            t = torch.full((B,), t_eval, device=self.dm.device)
            noise = torch.randn_like(x_0)
            x_t, _ = self.scheduler.add_noise(x_0, t, noise)

            # 单步预测 x_0
            pred_noise = self.model(x_t, t)
            mean_c, std = self.scheduler.marginal_params(t)
            mean_c = self.scheduler._broadcast(mean_c, x_0)
            std = self.scheduler._broadcast(std, x_0)
            x_0_pred = (x_t - std * pred_noise) / mean_c.clamp(min=1e-8)

            mse = ((x_0_pred - x_0) ** 2).mean().item()
            total_mse += mse * B
            count += B

        return total_mse / max(count, 1)

    @torch.no_grad()
    def _compute_val_loss(self, max_batches: int = 5) -> float:
        """在测试集上计算验证损失。"""
        total_loss, count = 0.0, 0

        for i, (x_0, _) in enumerate(self.test_loader):
            if i >= max_batches:
                break
            x_0 = x_0.to(self.dm.device)
            B = x_0.shape[0]
            t = self.scheduler.sample_timesteps(B, device=self.dm.device)
            noise = torch.randn_like(x_0)
            x_t, _ = self.scheduler.add_noise(x_0, t, noise)

            with self.dm.autocast():
                loss = self._compute_loss(x_t, t, noise, x_0)

            total_loss += loss.item() * B
            count += B

        return total_loss / max(count, 1)

    def _log_training(self, loss: float, grad_norm: float, step_time: float):
        """记录训练步日志。"""
        lr = self.optimizer.param_groups[0]["lr"]
        msg = (f"[Step {self.global_step}] loss={loss:.6f}, "
               f"grad_norm={grad_norm:.4f}, lr={lr:.2e}, "
               f"time={step_time:.3f}s")
        logger.info(msg)

        if self.exp_logger:
            self.exp_logger.log_training_step(
                self.global_step, loss, lr, grad_norm,
                extra={"train/step_time": step_time},
            )

    def _infinite_loader(self):
        """无限数据迭代器。"""
        while True:
            for batch in self.train_loader:
                yield batch
            self.epoch += 1

    def _resume(self, path: str):
        """从 checkpoint 恢复训练。"""
        logger.info(f"恢复训练: {path}")
        ckpt = torch.load(path, map_location=self.dm.device)
        self.model.load_state_dict_custom(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["step"]
        if "phase_detector" in ckpt.get("metadata", {}):
            self.phase_detector.load_state_dict(ckpt["metadata"]["phase_detector"])
        logger.info(f"恢复成功，从 step={self.global_step} 继续")

    def _build_summary(self) -> Dict[str, any]:
        """构建训练摘要。"""
        return {
            "final_step": self.global_step,
            "epochs": self.epoch,
            "gen_step": self.phase_detector.gen_step,
            "gen_detected": self.phase_detector.gen_detected,
            "mem_detected": self.phase_detector.mem_detected,
            "device": str(self.dm.device),
        }
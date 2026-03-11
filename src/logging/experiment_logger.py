"""
实验日志抽象层
==============
统一封装 WandB、TensorBoard、控制台日志，提供一致的 API。

典型用法:
    logger = ExperimentLogger(cfg.logging, cfg.experiment)
    logger.log_scalars({"loss": 0.5, "fid": 42.0}, step=1000)
    logger.log_image("generated", image_tensor, step=1000)
    logger.finish()
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ================================================================== #
#  后端抽象基类
# ================================================================== #

class LoggerBackend(ABC):
    """日志后端抽象接口。"""

    @abstractmethod
    def log_scalars(self, metrics: Dict[str, float], step: int) -> None: ...

    @abstractmethod
    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None: ...

    @abstractmethod
    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None: ...

    @abstractmethod
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None: ...

    @abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str, name: str, artifact_type: str) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


# ================================================================== #
#  WandB 后端
# ================================================================== #

class WandbBackend(LoggerBackend):
    """Weights & Biases 日志后端。"""

    def __init__(self, cfg_logging: DictConfig, cfg_experiment: DictConfig) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError("请安装 wandb: pip install wandb")

        self._wandb = wandb

        # 解析配置
        wandb_cfg = cfg_logging.wandb
        experiment_name = OmegaConf.to_container(cfg_experiment, resolve=True).get("name", "unnamed")

        self._run = wandb.init(
            project=wandb_cfg.get("project", "DiffusionGeometryLab"),
            entity=wandb_cfg.get("entity", None),
            name=experiment_name,
            tags=list(wandb_cfg.get("tags", [])),
            reinit=True,
        )
        logger.info(f"WandB 初始化完成: run={self._run.name}")

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
                if image.dim() == 3:
                    # (C, H, W) -> (H, W, C)
                    img_np = image.detach().cpu().permute(1, 2, 0).numpy()
                elif image.dim() == 2:
                    # (H, W) -> (H, W, 1)
                    img_np = image.detach().cpu().unsqueeze(-1).numpy()
                else:
                    img_np = image.detach().cpu().numpy()

                # 归一化到 [0, 255]
                img_np = _normalize_image_for_display(img_np)
                self._wandb.log({tag: self._wandb.Image(img_np)}, step=step)

    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        """images: (B, C, H, W)"""
        img_list = []
        for i in range(min(images.shape[0], 64)):  # 最多 64 张
            img_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
            img_np = _normalize_image_for_display(img_np)
            img_list.append(self._wandb.Image(img_np))
        self._wandb.log({tag: img_list}, step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self._wandb.log(
            {tag: self._wandb.Histogram(values.detach().cpu().numpy())},
            step=step,
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        self._wandb.config.update(config, allow_val_change=True)

    def log_artifact(self, path: str, name: str, artifact_type: str) -> None:
        artifact = self._wandb.Artifact(name, type=artifact_type)
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._wandb.finish()
        logger.info("WandB run 已结束。")


# ================================================================== #
#  TensorBoard 后端
# ================================================================== #

class TensorBoardBackend(LoggerBackend):
    """TensorBoard 日志后端。"""

    def __init__(self, cfg_logging: DictConfig, cfg_experiment: DictConfig) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("请安装 tensorboard: pip install tensorboard")

        log_dir = OmegaConf.to_container(cfg_logging.tensorboard, resolve=True).get(
            "log_dir", "outputs/tb_logs"
        )
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard 初始化完成: log_dir={log_dir}")

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, global_step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        # SummaryWriter.add_image 期望 (C, H, W), 值域 [0, 1]
        img = image.detach().cpu().float()
        if img.dim() == 2:
            img = img.unsqueeze(0)
        img = _normalize_tensor_01(img)
        self._writer.add_image(tag, img, global_step=step)

    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        # (B, C, H, W)
        imgs = images.detach().cpu().float()
        imgs = _normalize_tensor_01(imgs)
        self._writer.add_images(tag, imgs, global_step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self._writer.add_histogram(tag, values.detach().cpu(), global_step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        # TensorBoard 通过 hparams 记录配置
        # 展平嵌套字典
        flat = _flatten_dict(config)
        # hparams 值必须是基本类型
        safe_flat = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                     for k, v in flat.items()}
        self._writer.add_hparams(safe_flat, metric_dict={})

    def log_artifact(self, path: str, name: str, artifact_type: str) -> None:
        # TensorBoard 不原生支持 artifact，记录文本日志
        self._writer.add_text(
            f"artifacts/{artifact_type}",
            f"Artifact '{name}': {path}",
        )

    def finish(self) -> None:
        self._writer.flush()
        self._writer.close()
        logger.info("TensorBoard writer 已关闭。")


# ================================================================== #
#  空日志后端（无操作）
# ================================================================== #

class NullBackend(LoggerBackend):
    """空操作后端，用于禁用日志或测试。"""

    def log_scalars(self, metrics, step): pass
    def log_image(self, tag, image, step): pass
    def log_images(self, tag, images, step): pass
    def log_histogram(self, tag, values, step): pass
    def log_config(self, config): pass
    def log_artifact(self, path, name, artifact_type): pass
    def finish(self): pass


# ================================================================== #
#  统一日志门面
# ================================================================== #

class ExperimentLogger:
    """
    实验日志门面类，聚合多个后端。

    Parameters
    ----------
    cfg_logging : DictConfig
        logging 配置块。
    cfg_experiment : DictConfig
        experiment 配置块。

    Examples
    --------
    >>> exp_logger = ExperimentLogger(cfg.logging, cfg.experiment)
    >>> exp_logger.log_scalars({"train/loss": 0.5}, step=100)
    >>> exp_logger.log_cosine_field_curve(t_values, similarities, step=5000)
    >>> exp_logger.finish()
    """

    def __init__(
        self,
        cfg_logging: DictConfig,
        cfg_experiment: DictConfig,
    ) -> None:
        self._backends: List[LoggerBackend] = []
        self._step_offset: int = 0

        backend_name = cfg_logging.get("backend", "none").lower()

        # 设置 Python 标准日志
        self._setup_python_logging(cfg_experiment)

        # 初始化后端
        if backend_name in ("wandb", "both"):
            try:
                self._backends.append(WandbBackend(cfg_logging, cfg_experiment))
            except Exception as e:
                logger.warning(f"WandB 初始化失败，跳过: {e}")

        if backend_name in ("tensorboard", "both"):
            try:
                self._backends.append(TensorBoardBackend(cfg_logging, cfg_experiment))
            except Exception as e:
                logger.warning(f"TensorBoard 初始化失败，跳过: {e}")

        if not self._backends:
            self._backends.append(NullBackend())
            if backend_name != "none":
                logger.warning("所有日志后端初始化失败，使用 NullBackend。")

        logger.info(f"ExperimentLogger 就绪，后端: "
                     f"{[type(b).__name__ for b in self._backends]}")

    # ---- 核心日志方法 ----

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """记录标量指标。"""
        for b in self._backends:
            b.log_scalars(metrics, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """记录单张图像。"""
        for b in self._backends:
            b.log_image(tag, image, step)

    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        """记录一批图像。"""
        for b in self._backends:
            b.log_images(tag, images, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """记录直方图。"""
        for b in self._backends:
            b.log_histogram(tag, values, step)

    def log_config(self, config: Union[DictConfig, Dict[str, Any]]) -> None:
        """记录实验配置。"""
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        for b in self._backends:
            b.log_config(config)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        """记录文件 artifact（如 checkpoint）。"""
        for b in self._backends:
            b.log_artifact(path, name, artifact_type)

    # ---- 实验专用高级日志方法 ----

    def log_cosine_field(
        self,
        field_name: str,
        t_values: np.ndarray,
        similarities: np.ndarray,
        step: int,
        dimension: Optional[int] = None,
    ) -> None:
        """
        记录余弦相似度场曲线。

        Parameters
        ----------
        field_name : str
            字段名，如 "S_cos_gen_mem", "S_cos_model_empirical", "S_cos_train_test"。
        t_values : np.ndarray, shape (T,)
            时间步数组。
        similarities : np.ndarray, shape (T,)
            对应的平均余弦相似度。
        step : int
            训练步数。
        dimension : int, optional
            环境维度 d，用于区分不同维度的曲线。
        """
        prefix = f"cosine_field/{field_name}"
        if dimension is not None:
            prefix = f"cosine_field/d{dimension}/{field_name}"

        for i, (t, s) in enumerate(zip(t_values, similarities)):
            self.log_scalars({f"{prefix}/t_{t:.3f}": float(s)}, step=step)

        # 记录统计摘要
        self.log_scalars({
            f"{prefix}/mean": float(similarities.mean()),
            f"{prefix}/std": float(similarities.std()),
            f"{prefix}/min": float(similarities.min()),
            f"{prefix}/max": float(similarities.max()),
        }, step=step)

    def log_memory_ratio(
        self,
        t_values: np.ndarray,
        f_mem_values: np.ndarray,
        step: int,
        dimension: Optional[int] = None,
    ) -> None:
        """
        记录动态记忆比例 f_mem(t) 曲线。

        Parameters
        ----------
        t_values : np.ndarray, shape (T,)
        f_mem_values : np.ndarray, shape (T,)
        step : int
        dimension : int, optional
        """
        prefix = "memory_ratio"
        if dimension is not None:
            prefix = f"memory_ratio/d{dimension}"

        for t, f in zip(t_values, f_mem_values):
            self.log_scalars({f"{prefix}/t_{t:.3f}": float(f)}, step=step)

        self.log_scalars({
            f"{prefix}/mean": float(f_mem_values.mean()),
            f"{prefix}/max": float(f_mem_values.max()),
        }, step=step)

    def log_phase_event(
        self,
        phase: str,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录相变检测事件。

        Parameters
        ----------
        phase : str
            "gen_detected" 或 "mem_detected"。
        step : int
        metadata : dict, optional
            附加元数据（如当时的 FID、gap 值）。
        """
        event_data = {"phase_event/type": phase, "phase_event/step": step}
        if metadata:
            for k, v in metadata.items():
                event_data[f"phase_event/{k}"] = v
        self.log_scalars(event_data, step=step)
        logger.info(f"🔔 相变事件: {phase} @ step {step}, metadata={metadata}")

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """训练步常规日志的便捷方法。"""
        metrics = {
            "train/loss": loss,
            "train/lr": lr,
            "train/step": step,
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
        if extra:
            metrics.update(extra)
        self.log_scalars(metrics, step=step)

    # ---- 生命周期 ----

    def finish(self) -> None:
        """结束所有日志后端（刷新缓冲区、上传数据）。"""
        for b in self._backends:
            b.finish()
        logger.info("ExperimentLogger 已完成所有后端的清理。")

    # ---- 私有方法 ----

    @staticmethod
    def _setup_python_logging(cfg_experiment: DictConfig) -> None:
        """配置 Python 标准日志格式和级别。"""
        output_dir = OmegaConf.to_container(cfg_experiment, resolve=True).get(
            "output_dir", "outputs"
        )
        log_file = Path(output_dir) / "experiment.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # 根 logger 配置
        root_logger = logging.getLogger()

        # 避免重复添加 handler
        if not root_logger.handlers:
            root_logger.setLevel(logging.INFO)

            formatter = logging.Formatter(
                fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # 控制台 handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

            # 文件 handler
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)


# ================================================================== #
#  辅助函数
# ================================================================== #
def _normalize_image_for_display(img: np.ndarray) -> np.ndarray:
    """
    将图像数组归一化到 [0, 255] uint8，用于 WandB 显示。

    Parameters
    ----------
    img : np.ndarray
        任意值域的图像数组。

    Returns
    -------
    np.ndarray, dtype=uint8
    """
    img = img.astype(np.float32)
    # 处理 [-1, 1] 或 [0, 1] 范围
    if img.min() < 0:
        # 假设 [-1, 1] 范围
        img = (img + 1.0) / 2.0
    # 裁剪到 [0, 1]
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _normalize_tensor_01(tensor: torch.Tensor) -> torch.Tensor:
    """
    将张量归一化到 [0, 1]，用于 TensorBoard 显示。

    Parameters
    ----------
    tensor : torch.Tensor
        任意值域的图像张量。

    Returns
    -------
    torch.Tensor, 值域 [0, 1]
    """
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0.0, 1.0)


def _flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "/",
) -> Dict[str, Any]:
    """
    递归展平嵌套字典。

    Parameters
    ----------
    d : dict
        嵌套字典。
    parent_key : str
        父级键前缀。
    sep : str
        键之间的分隔符。

    Returns
    -------
    dict
        展平后的单层字典。

    Examples
    --------
    >>> _flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
    {"a/b": 1, "a/c/d": 2}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, (list, tuple)):
            items[new_key] = str(v)
        else:
            items[new_key] = v
    return items
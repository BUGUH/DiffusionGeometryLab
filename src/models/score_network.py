"""
得分网络封装 — 支持 UNet (图像) 和 MLP (向量)
=============================================
通过 cfg.model.arch 切换: "unet" | "mlp"
"""

from __future__ import annotations
from typing import Optional
from copy import deepcopy

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ScoreNetwork(nn.Module):
    """统一得分函数接口，自动选择 UNet 或 MLP 后端。"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.output_type = cfg.model.output_type
        self.scheduler_type = cfg.scheduler.type
        self.arch = cfg.model.get("arch", "unet")

        # ---- Circle 数据适配：根据 arch 选择后端 ----
        if self.arch == "mlp":
            from src.models.mlp import MLPScoreNet
            D = cfg.data.circle.ambient_dim
            mlp_cfg = cfg.model.get("mlp", {})
            self.backbone = MLPScoreNet(
                data_dim=D,
                hidden_dims=list(mlp_cfg.get("hidden_dims", [512, 512, 512])),
                time_embed_dim=cfg.model.get("time_embed_dim", 128),
                dropout=cfg.model.get("dropout", 0.1),
            )
        else:
            from src.models.unet import UNet
            self.backbone = UNet(
                in_channels=cfg.model.in_channels,
                out_channels=cfg.model.out_channels,
                base_channels=cfg.model.base_channels,
                channel_multipliers=list(cfg.model.channel_multipliers),
                num_res_blocks=cfg.model.num_res_blocks,
                attention_resolutions=list(cfg.model.attention_resolutions),
                dropout=cfg.model.dropout,
                time_embed_dim=cfg.model.time_embed_dim,
                input_resolution=cfg.data.target_resolution,
            )

        # EMA
        self.ema_model: Optional[nn.Module] = None
        if cfg.training.get("use_ema", False):
            self.ema_decay = cfg.training.ema_decay
            self.ema_model = deepcopy(self.backbone)
            self.ema_model.requires_grad_(False)

        self._print_info()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.backbone(x_t, t)

    def predict_noise(self, x_t, t, use_ema=False):
        model = self._get_model(use_ema)
        raw = model(x_t, t)
        if self.output_type == "noise":
            return raw
        raise RuntimeError(f"output_type='{self.output_type}' 不支持直接 predict_noise")

    def predict_score(self, x_t, t, sigma_t=None, use_ema=False):
        model = self._get_model(use_ema)
        raw = model(x_t, t)
        if self.output_type == "score":
            return raw
        elif self.output_type == "noise":
            assert sigma_t is not None
            sigma_t = self._ensure_shape(sigma_t, x_t)
            return -raw / sigma_t.clamp(min=1e-8)
        raise RuntimeError(f"output_type='{self.output_type}' 不支持 predict_score")

    def predict_velocity(self, x_t, t, use_ema=False):
        model = self._get_model(use_ema)
        raw = model(x_t, t)
        if self.output_type == "velocity":
            return raw
        raise RuntimeError(f"output_type='{self.output_type}' 不支持 predict_velocity")

    def predict_denoising_direction(self, x_t, t, sigma_t=None, use_ema=False):
        score = self.predict_score(x_t, t, sigma_t=sigma_t, use_ema=use_ema)
        return -score

    @torch.no_grad()
    def update_ema(self):
        if self.ema_model is None:
            return
        for p_ema, p in zip(self.ema_model.parameters(), self.backbone.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    def copy_to_ema(self):
        if self.ema_model is not None:
            self.ema_model.load_state_dict(self.backbone.state_dict())

    def _get_model(self, use_ema):
        if use_ema and self.ema_model is not None:
            return self.ema_model
        return self.backbone

    @staticmethod
    def _ensure_shape(sigma, x):
        while sigma.dim() < x.dim():
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _print_info(self):
        n = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        print(f"ScoreNetwork: arch={self.arch}, output={self.output_type}, "
              f"params={n:.2f}M, EMA={'✓' if self.ema_model else '✗'}")

    def get_state_dict(self, include_ema=True):
        state = {"unet": self.backbone.state_dict()}
        if include_ema and self.ema_model is not None:
            state["ema"] = self.ema_model.state_dict()
        return state

    def load_state_dict_custom(self, state):
        self.backbone.load_state_dict(state["unet"])
        if "ema" in state and self.ema_model is not None:
            self.ema_model.load_state_dict(state["ema"])
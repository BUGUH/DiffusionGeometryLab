"""
自适应维度 UNet（修复版）
========================
严格追踪 skip connection 通道数，避免编码-解码通道不匹配。
"""

from __future__ import annotations
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeMLPEmbedding(nn.Module):
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


def _group_norm(channels: int) -> nn.GroupNorm:
    """安全创建 GroupNorm，确保 num_groups 能整除 channels。"""
    num_groups = min(32, channels)
    while channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = _group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-1, -2) @ k * self.scale).softmax(dim=-1)
        h = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# src/models/unet.py (完整修复版)

class UNet(nn.Module):
    """
    自适应维度 UNet (修复版)。
    严格匹配 Encoder 和 Decoder 的 Skip Connections。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = (16, 8),
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        input_resolution: int = 64,
        min_resolution: int = 4,
    ):
        super().__init__()

        # 1. 计算层级数
        max_levels = len(channel_multipliers)
        auto_levels = max(1, int(math.log2(max(input_resolution, min_resolution) / min_resolution)))
        num_levels = min(max_levels, auto_levels)
        mults = list(channel_multipliers[:num_levels])

        self.input_resolution = input_resolution
        self.num_levels = num_levels

        time_dim = base_channels * 4
        self.time_embed = TimeMLPEmbedding(time_embed_dim, time_dim)
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 2. 构建编码器 (Encoder)
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        # 用于记录 skip 连接的通道数，确保 Decoder 能对应上
        # 注意：这里只记录通道数用于初始化 Decoder，实际 tensor 在 forward 中动态生成
        skip_channels_template = [] 
        
        ch = base_channels
        current_res = input_resolution

        for level in range(num_levels):
            out_ch = base_channels * mults[level]
            blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                if current_res in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))
                ch = out_ch
                skip_channels_template.append(ch) # ResBlock 输出

            self.encoder.append(blocks)

            # 除最后一层外都下采样
            if level < num_levels - 1:
                self.downsamples.append(Downsample(ch))
                skip_channels_template.append(ch) # Downsample 输出
                current_res //= 2
            else:
                self.downsamples.append(None)

        # 3. 构建中间层 (Middle)
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        # 4. 构建解码器 (Decoder)
        # 关键：必须严格按照 Encoder push 的顺序反向 pop
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # 复制一份模板用于构建，避免修改原列表影响逻辑（虽然这里是初始化阶段）
        # 实际上我们需要的是反向遍历 template 来确定输入通道
        # 但为了简单，我们直接在循环里模拟 pop 逻辑来构建层
        
        # 重新计算当前通道 ch (应该是 encoder 最后的输出通道)
        # ch 已经是 mid 的输入通道了
        
        # 我们需要一个指针或者反向迭代 template 来获取 skip 通道
        # 为了清晰，我们直接在 build 时反向读取 template
        template_idx = len(skip_channels_template) - 1

        for level in reversed(range(num_levels)):
            out_ch = base_channels * mults[level]
            blocks = nn.ModuleList()

            # 确定这一层需要多少个 ResBlock
            # Encoder 每层有 num_res_blocks 个 ResBlock + (如果有下采样) 1 个 Downsample
            # Decoder 对应层需要消耗掉这些 skip
            blocks_count = num_res_blocks
            if level < num_levels - 1:
                blocks_count += 1 # 多一个块来消耗 downsample 的 skip
            
            for _ in range(blocks_count):
                if template_idx < 0:
                    raise RuntimeError("Skip channels mismatch: template exhausted too early")
                
                skip_ch = skip_channels_template[template_idx]
                template_idx -= 1
                
                blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                if current_res in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))
                ch = out_ch

            self.decoder.append(blocks)

            if level > 0:
                self.upsamples.append(Upsample(ch))
                current_res *= 2
            else:
                self.upsamples.append(None)
        
        if template_idx != -1:
            raise RuntimeError(f"Skip channels mismatch: {template_idx + 1} skips left unused")

        # 5. 输出层
        self.conv_out = nn.Sequential(
            _group_norm(ch), nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.conv_in(x)
        
        skips = [] # ✅ 正确的位置：初始化 skip 列表

        # --- 编码阶段 ---
        for level in range(self.num_levels):
            # 处理该层的 ResBlocks 和 Attention
            for block in self.encoder[level]:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                    skips.append(h)  # 记录 ResBlock 输出
                else:  # AttentionBlock
                    h = block(h)
                    # 注意：AttentionBlock 通常不单独作为 skip 连接点，除非架构特殊设计
                    # 标准 UNet 只在 ResBlock 后 skip。如果上面 __init__ 里记录了 Attention 后的状态，这里要调整。
                    # 根据上面的 __init__ 逻辑，我们只记录了 ResBlock 和 Downsample 的输出。
                    # 所以这里不需要 append attention 的输出。
            
            # 处理下采样
            if self.downsamples[level] is not None:
                h = self.downsamples[level](h)
                skips.append(h)  # 记录 Downsample 输出

        # --- 中间阶段 ---
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # --- 解码阶段 ---
        for blocks, up in zip(self.decoder, self.upsamples):
            for block in blocks:
                if isinstance(block, ResBlock):
                    if not skips:
                        raise IndexError("Pop from empty skips list! Architecture mismatch.")
                    skip_h = skips.pop()
                    # 确保尺寸一致 (理论上应该一致，以防万一做个检查)
                    if h.shape != skip_h.shape:
                        # 如果高度宽度不一致（极少见），进行插值
                        if h.shape[2:] != skip_h.shape[2:]:
                            skip_h = F.interpolate(skip_h, size=h.shape[2:], mode="bilinear", align_corners=False)
                        # 如果通道不一致，ResBlock 的第一个卷积会处理 (input_ch = ch + skip_ch)
                    
                    h = torch.cat([h, skip_h], dim=1)
                    h = block(h, t_emb)
                else:  # AttentionBlock
                    h = block(h)
            
            # 处理上采样
            if up is not None:
                h = up(h)

        return self.conv_out(h)
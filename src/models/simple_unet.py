"""UNet 模型架构。"""
import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim=128, dim_exp=512):
        super().__init__()
        self.dim = dim
        self.dim_exp = dim_exp
        self.time_mlp = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=self.dim_exp),
            nn.GELU(),
            nn.Linear(in_features=self.dim_exp, out_features=self.dim_exp),
        )
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        ts = time * 1000.0 
        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.time_mlp(emb)

class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=4, groups=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.mhsa = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)
        h, _ = self.mhsa(h, h, h)
        h = h.transpose(2, 1).view(B, C, H, W)
        return x + h

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout_rate=0.1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.conv(self.dropout(self.act(self.norm(x))))

class ResnetBlock(nn.Module):
    def __init__(self, *, dim, dim_out, dropout_rate=0.1, time_emb_dims=512, groups=8, apply_attention=False):
        super().__init__()
        self.act = nn.SiLU()
        self.block1 = Block(dim=dim, dim_out=dim_out, dropout_rate=0.0, groups=groups)
        self.block2 = Block(dim=dim_out, dim_out=dim_out, dropout_rate=dropout_rate, groups=groups)
        self.dense = nn.Linear(time_emb_dims, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1, 1) if dim != dim_out else nn.Identity()
        self.attention = Attention(dim=dim_out) if apply_attention else nn.Identity()
    def forward(self, x, t):
        h = self.block1(x)
        h += self.dense(self.act(t))[:, :, None, None]
        h = self.block2(h)
        h = h + self.res_conv(x)
        h = self.attention(h)
        return h

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(channels, channels, 3, 2, 1)
    def forward(self, x, *args): return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        )
    def forward(self, x, *args): return self.upsample(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_channels=64
        base_channels_multiples=(1, 2, 4, 8)
        apply_attention=(False, True, True, False)
        dropout_rate=0.1
        time_emb_dims_exp = base_channels * 4
        self.time_embeddings = SinusoidalPositionEmbeddings(dim=base_channels, dim_exp=time_emb_dims_exp)
        self.init_conv = nn.Conv2d(1, base_channels, 3, 1, 1)
        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        curr_channels = [base_channels]
        in_channels = base_channels
        for level in range(len(base_channels_multiples)):
            is_last = (level >= (len(base_channels_multiples) - 1))
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(2):
                block = ResnetBlock(dim=in_channels, dim_out=out_channels, dropout_rate=dropout_rate,
                                    time_emb_dims=time_emb_dims_exp, apply_attention=apply_attention[level])
                self.encoder_blocks.append(block)
                in_channels = out_channels
                curr_channels.append(in_channels)
            if not is_last:
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)
        self.bottleneck_blocks = nn.ModuleList([
            ResnetBlock(dim=in_channels, dim_out=in_channels, dropout_rate=dropout_rate,
                        time_emb_dims=time_emb_dims_exp, apply_attention=True),
            ResnetBlock(dim=in_channels, dim_out=in_channels, dropout_rate=dropout_rate,
                        time_emb_dims=time_emb_dims_exp, apply_attention=False)
        ])
        for level in reversed(range(len(base_channels_multiples))):
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(3):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(dim=encoder_in_channels + in_channels, dim_out=out_channels, dropout_rate=dropout_rate,
                                    time_emb_dims=time_emb_dims_exp, apply_attention=apply_attention[level])
                in_channels = out_channels
                self.decoder_blocks.append(block)
            if level != 0: self.decoder_blocks.append(UpSample(in_channels=in_channels))
        self.final_block = Block(in_channels, 1, dropout_rate=0.0)
    def forward(self, x, t):
        time_emb = self.time_embeddings(t)
        h = self.init_conv(x)
        outs = [h]
        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)
        for layer in self.bottleneck_blocks: h = layer(h, time_emb)
        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)
        return self.final_block(h)
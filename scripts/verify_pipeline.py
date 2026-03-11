"""端到端管线验证脚本（可用合成数据离线测试）。"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from omegaconf import OmegaConf

# 构造最小配置（不依赖真实 parquet）
minimal_cfg = OmegaConf.create({
    "data": {
        "grayscale_size": 32,
        "target_dim": 4096,
        "target_resolution": 64,
        "padding_mode": "center",
        "normalize_range": [-1.0, 1.0],
    },
    "model": {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 32,
        "channel_multipliers": [1, 2, 4],
        "num_res_blocks": 1,
        "attention_resolutions": [8],
        "dropout": 0.0,
        "time_embed_dim": 128,
        "output_type": "noise",
    },
    "scheduler": {"type": "vp"},
    "training": {"use_ema": True, "ema_decay": 0.999},
})


def test_zero_padding():
    from src.data.zero_padding import ZeroPadder

    padder = ZeroPadder(32, 64, mode="center")
    x = torch.randn(4, 1, 32, 32)
    padded = padder.pad(x)
    assert padded.shape == (4, 1, 64, 64), f"期望 (4,1,64,64), 得到 {padded.shape}"

    # 验证填充区域为零
    offset = (64 - 32) // 2
    assert padded[:, :, :offset, :].abs().sum() == 0, "上方应全零"
    assert padded[:, :, offset+32:, :].abs().sum() == 0, "下方应全零"

    # 验证 unpad 恢复
    recovered = padder.unpad(padded)
    assert torch.allclose(x, recovered), "unpad 应恢复原始数据"

    # 测试 128 维度
    padder128 = ZeroPadder(32, 128, mode="center")
    padded128 = padder128.pad(x)
    assert padded128.shape == (4, 1, 128, 128)
    assert torch.allclose(x, padder128.unpad(padded128))

    print("✅ ZeroPadder 测试通过")


def test_unet_adaptive():
    from src.models.unet import UNet

    for res in [32, 64, 128]:
        model = UNet(
            in_channels=1, out_channels=1,
            base_channels=32, channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=1, attention_resolutions=[8],
            dropout=0.0, time_embed_dim=128,
            input_resolution=res,
        )
        x = torch.randn(2, 1, res, res)
        t = torch.rand(2)
        out = model(x, t)
        assert out.shape == x.shape, f"res={res}: 期望 {x.shape}, 得到 {out.shape}"
        print(f"  UNet(res={res}): levels={model.num_levels}, output={out.shape} ✓")

    print("UNet 自适应分辨率测试通过")


def test_score_network():
    from src.models.score_network import ScoreNetwork

    net = ScoreNetwork(minimal_cfg)
    x = torch.randn(2, 1, 64, 64)
    t = torch.rand(2)

    # 前向
    raw = net(x, t)
    assert raw.shape == x.shape

    # predict_noise
    noise = net.predict_noise(x, t)
    assert noise.shape == x.shape

    # predict_score（noise模式需要sigma_t）
    sigma = torch.ones(2) * 0.5
    score = net.predict_score(x, t, sigma_t=sigma)
    assert score.shape == x.shape

    # predict_denoising_direction
    direction = net.predict_denoising_direction(x, t, sigma_t=sigma)
    assert direction.shape == x.shape

    # EMA 更新
    net.copy_to_ema()
    net.update_ema()
    noise_ema = net.predict_noise(x, t, use_ema=True)
    assert noise_ema.shape == x.shape

    # 状态保存/加载
    state = net.get_state_dict(include_ema=True)
    assert "unet" in state and "ema" in state

    print("ScoreNetwork 测试通过")


def test_preprocessing_synthetic():
    """用合成数据测试预处理流程。"""
    from src.data.preprocessing import preprocess_images

    fake_images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    result = preprocess_images(fake_images, target_size=32, normalize_range=(-1, 1))
    assert result.shape == (10, 1, 32, 32)
    assert result.min() >= -1.0 and result.max() <= 1.0
    print("预处理测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("DiffusionGeometryLab 管线验证")
    print("=" * 60)
    test_zero_padding()
    test_unet_adaptive()
    test_score_network()
    test_preprocessing_synthetic()
    print("\n 所有验证通过！")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. 核心功能函数：加载 Inception-V3
# ==========================================
def load_feature_extractor(device):
    """
    加载 ImageNet 预训练的 Inception-V3 
    (替换了原来的 LightCNN)
    """
    print(f"🔄 正在初始化 Inception-V3 (ImageNet Pretrained)...")
    
    # 自动下载或加载官方预训练权重
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights, transform_input=False).to(device)
    
    # 我们不需要最后的 1000 类分类层，我们只要倒数第二层的 2048 维池化特征
    model.fc = nn.Identity()
    
    # 冻结所有参数，防止误操作改变权重
    for param in model.parameters():
        param.requires_grad = False
        
    model.eval()
    print("✅ Inception-V3 特征提取器加载成功！")
    return model

# ==========================================
# 2. 核心功能函数：极速提取特征 (内置硬盘缓存机制)
# ==========================================
def get_image_features(data_input, model, device, batch_size=64, desc="提取特征", cache_path=None):
    """
    带自动缓存功能的特征提取接口。
    如果提供了 cache_path 且文件存在，则直接秒读硬盘！
    """
    # ==========================================
    # 缓存拦截逻辑：如果已经提取过，直接读取！
    # ==========================================
    if cache_path is not None and os.path.exists(cache_path):
        print(f"📦 [Cache Hit] 发现已缓存的特征，直接加载: {cache_path}")
        return np.load(cache_path)

    # 1. 统一转为 Tensor
    if isinstance(data_input, np.ndarray):
        data_input = torch.tensor(data_input, dtype=torch.float32)
    else:
        data_input = data_input.clone().detach().float()
        
    # 全局视野判断，防止 Batch 塌陷
    global_min = data_input.min().item()
    needs_mapping = (global_min < 0.0)
    print(f"[{desc}] 全局最小像素: {global_min:.2f} -> 触发转至[0,1]映射: {needs_mapping}")
    
    dataset = TensorDataset(data_input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_features = []
    model.eval() 
    
    # Inception-V3 所需的 ImageNet 标准均值和方差
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=False):
            x = batch[0].to(device)
            
            # (A) 保证通道维度正确 (N, C, H, W)
            if x.dim() == 3:
                x = x.unsqueeze(1)
                
            # (B) 如果是单通道灰度图，必须复制成 3 通道 (Inception 需要 RGB)
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
                
            # (C) 映射到 [0, 1] 范围
            if needs_mapping:
                x = (x + 1.0) / 2.0
                
            # (D) Resize 到 Inception 的标准输入尺寸 299x299
            x_resized = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
            # (E) ImageNet 标准归一化
            x_normalized = TF.normalize(x_resized, mean=imagenet_mean, std=imagenet_std)
            
            # 提取 2048 维特征
            features = model(x_normalized)
            
            all_features.append(features.cpu().numpy())
            
    final_features = np.concatenate(all_features, axis=0)
    
    # ==========================================
    # 缓存保存逻辑：如果传入了路径，计算完顺手存下来
    # ==========================================
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, final_features)
        print(f"💾 特征已缓存至: {cache_path}")
        
    return final_features
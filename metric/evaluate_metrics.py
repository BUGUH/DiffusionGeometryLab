import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import torch
from tqdm import tqdm

@torch.no_grad()
def precompute_test_radii(test_features, k_values=[1, 3, 5], chunk_size=2048):
    """
    预计算 Test 集的 K-NN 流形半径 (基于标准的欧氏距离 L2)
    """
    # ❌ 删掉了 F.normalize
    test_tensor = torch.tensor(test_features, dtype=torch.float32).cuda()
    N_test = test_tensor.size(0)
    
    max_k = max(k_values)
    radii_dict = {k: torch.zeros(N_test, device='cpu') for k in k_values}
    
    for i in tqdm(range(0, N_test, chunk_size), desc=f"计算 Test Radii (Max K={max_k})"):
        end = min(i + chunk_size, N_test)
        test_chunk = test_tensor[i:end]
        
        # ✅ 使用 torch.cdist 直接计算 L2 欧氏距离矩阵
        dist_chunk = torch.cdist(test_chunk, test_tensor, p=2.0)
        
        # 取 top (max_k + 1) 小的距离 (因为自己到自己的距离是第1小，为0)
        topk_dists, _ = torch.topk(dist_chunk, k=max_k + 1, largest=False, dim=1)
        
        for k in k_values:
            radii_dict[k][i:end] = topk_dists[:, k].cpu()
            
    return radii_dict

@torch.no_grad()
def calculate_precision_knn(gen_features, test_features, radii_k, chunk_size=2048):
    """
    计算 Precision (基于标准的欧氏距离 L2)
    """
    # ❌ 删掉了 F.normalize
    gen_tensor = torch.tensor(gen_features, dtype=torch.float32).cuda()
    test_tensor = torch.tensor(test_features, dtype=torch.float32).cuda()
    radii = radii_k.cuda()
    
    N_gen = gen_tensor.size(0)
    hits_count = 0
    
    for i in range(0, N_gen, chunk_size):
        end = min(i + chunk_size, N_gen)
        gen_chunk = gen_tensor[i:end]
        
        # ✅ 使用 torch.cdist 直接计算 L2 欧氏距离矩阵
        dist_chunk = torch.cdist(gen_chunk, test_tensor, p=2.0)
        
        # 如果距离 <= 该 test 样本的 k-NN 半径，说明落入了流形
        in_manifold = dist_chunk <= (radii.unsqueeze(0))
        
        # 只要落入任意一个 test 样本的流形，就算作 hit
        hits_chunk = in_manifold.any(dim=1)
        hits_count += hits_chunk.sum().item()
            
    return hits_count / N_gen

@torch.no_grad()
def calculate_memorization_ratio(gen_features, train_features, ratio_threshold=1/3, chunk_size=2048):
    """
    计算 Memorization (基于标准的欧氏距离 L2)
    """
    # ❌ 删掉了 F.normalize
    gen_tensor = torch.tensor(gen_features, dtype=torch.float32).cuda()
    train_tensor = torch.tensor(train_features, dtype=torch.float32).cuda()
    
    N_gen = gen_tensor.size(0)
    memorized_count = 0
    
    for i in range(0, N_gen, chunk_size):
        end = min(i + chunk_size, N_gen)
        gen_chunk = gen_tensor[i:end]
        
        # ✅ 使用 torch.cdist 直接计算 L2 欧氏距离矩阵
        dist_chunk = torch.cdist(gen_chunk, train_tensor, p=2.0)
        
        # 找离得最近的 2 张训练图
        top2_dists, _ = torch.topk(dist_chunk, k=2, largest=False, dim=1)
        
        d1 = top2_dists[:, 0]
        d2 = torch.clamp(top2_dists[:, 1], min=1e-8) 
        
        ratios = d1 / d2
        memorized_count += (ratios < ratio_threshold).sum().item()
        
    return memorized_count / N_gen
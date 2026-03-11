"""
纯理论数值模拟轨迹的动态记忆比例 f_mem(t) 计算。
脱离 PyTorch，基于纯 NumPy 和 Faiss，用于精确验证 ODE 轨迹。
"""
from __future__ import annotations
import logging
import numpy as np
import faiss
from typing import Dict
from src.sampling.numerical_sampling import TheoreticalSchedule

logger = logging.getLogger(__name__)

class NumericalMemoryEvaluator:
    """
    针对纯数值 ODE 轨迹的 f_mem(t) 评估器。
    
    约定时间轴 (与 TheoreticalSchedule 保持一致):
    t=0 为纯噪声, t=1 为干净数据。
    """
    def __init__(
        self,
        schedule_mode: str = "VP",
        threshold: float = 1.0 / 3.0,
    ):
        self.schedule_mode = schedule_mode.upper()
        self.threshold = threshold
        self.schedule = TheoreticalSchedule(mode=self.schedule_mode)

    def compute(
        self,
        train_data_np: np.ndarray,
        trajectories: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        计算轨迹演化过程中的记忆比例。

        Parameters
        ----------
        train_data_np : (N_train, C, H, W) 真实的训练集
        trajectories : (num_samples, steps, C, H, W) ODE 求解出的演化轨迹

        Returns
        -------
        Dict 包含:
            "t": 演化时间戳数组 (0 到 1)
            "f_mem": 每个时间步的记忆化比例
            "mean_ratio": R_t 的均值
        """
        num_samples, steps, C, H, W = trajectories.shape
        N_train = train_data_np.shape[0]
        dim = C * H * W
        
        # Faiss 严格要求输入为连续的 float32
        train_flat = np.ascontiguousarray(train_data_np.reshape(N_train, -1).astype(np.float32))
        
        t_ode_values = np.linspace(0.0, 1.0, steps)
        f_mem_values = []
        mean_ratios = []
        
        for step_idx, t_ode in enumerate(t_ode_values):
            # 1. 提取当前时刻生成样本
            x_t_samples = np.ascontiguousarray(
                trajectories[:, step_idx, ...].reshape(num_samples, -1).astype(np.float32)
            )
            
            # 2. 获取理论参数并对训练集加等量噪声
            # 注意：t_ode=0 时代表噪声，这里取得的 alpha 极小，sigma 极大
            alpha_t, sigma_t, _, _ = self.schedule.get_params(t_ode)
            noise = np.random.randn(N_train, dim).astype(np.float32)
            train_noisy = np.ascontiguousarray(alpha_t * train_flat + sigma_t * noise)
            
            # 3. 使用 Faiss 暴力检索 L2 距离
            index = faiss.IndexFlatL2(dim)
            index.add(train_noisy)
            
            # 4. 查询 2-NN
            dists_sq, _ = index.search(x_t_samples, k=2)
            
            # 5. 还原为欧式距离，并防止除以零
            d1 = np.sqrt(np.clip(dists_sq[:, 0], a_min=1e-10, a_max=None))
            d2 = np.sqrt(np.clip(dists_sq[:, 1], a_min=1e-10, a_max=None))
            
            # 6. 计算比例
            R_t = d1 / d2
            f_mem = np.mean((R_t < self.threshold).astype(float))
            
            f_mem_values.append(f_mem)
            mean_ratios.append(np.mean(R_t))
            
        return {
            "t": t_ode_values,
            "f_mem": np.array(f_mem_values),
            "mean_ratio": np.array(mean_ratios),
        }
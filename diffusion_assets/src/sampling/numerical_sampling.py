"""精确理论 ODE 求解器 (Numerical Sampling)
不使用神经网络，直接利用真实数据集进行核密度估计(KDE)，模拟真实的概率流形轨迹。
"""
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import solve_ivp

class TheoreticalSchedule:
    """
    严格按照理论公式的调度器。
    约定：t=0 为纯噪声，t=1 为干净数据。
    """
    def __init__(self, mode="VP", sigma_max=50.0, sigma_min=0.01):
        self.mode = mode.upper()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min # VE 必须要有一个 sigma_min 防止对数爆炸

    def get_params(self, t):
        """返回 alpha(t), sigma(t), d_alpha(t)/dt, d_sigma(t)/dt"""
        
        if self.mode == "VP":
            # VP: α = sin(πt/2), σ = cos(πt/2)
            alpha = np.sin(np.pi * t / 2)
            sigma = np.cos(np.pi * t / 2)
            d_alpha = (np.pi / 2) * np.cos(np.pi * t / 2)
            d_sigma = -(np.pi / 2) * np.sin(np.pi * t / 2)
            
        elif self.mode == "VE":
            # VE: 指数衰减 (与 Song Yang 论文及你的训练代码完全对齐)
            # t=0 时为 sigma_max, t=1 时为 sigma_min
            alpha = 1.0
            d_alpha = 0.0
            
            # σ_t = σ_max * (σ_min / σ_max)^t
            ratio = self.sigma_min / self.sigma_max
            sigma = self.sigma_max * (ratio ** t)
            
            # 导数: dσ/dt = σ_t * ln(σ_min / σ_max)
            d_sigma = sigma * np.log(ratio)
            
        elif self.mode in ["OT", "RF"]:
            # OT (Rectified Flow): 匀速直线运动
            alpha = t
            sigma = 1.0 - t
            d_alpha = 1.0
            d_sigma = -1.0
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        # 防止 sigma 为严格的 0 导致除法报错
        sigma = max(sigma, 1e-8)
        return alpha, sigma, d_alpha, d_sigma

def compute_hat_z_flat(x_flat, z_data_flat, alpha, sigma):
    """
    最纯粹的理论 \hat{z}(x_t)。
    没有任何 if 语句和最近邻作弊，完全依靠 KDE 核密度的自然演化！
    """
    # 广播计算差值和距离平方: ||x - \alpha z_i||^2
    diff = x_flat - alpha * z_data_flat
    dists_sq = np.sum(diff**2, axis=1)
    
    # 直接计算 logits。
    # 由于我们在 schedule 里限制了 sigma 最小为 1e-8，这里不会报除以0的错
    logits = -dists_sq / (2 * sigma**2)
    
    # 依靠 logsumexp 的内部数值稳定技巧，即使 logits 是 -10^10，也能正确算出权重
    # 最接近的那个点的 weight 会自然而然地趋近于 1.0，其他的趋近于 0.0
    weights = np.exp(logits - logsumexp(logits))
    
    # 依靠公式自己的加权求和，自然坍缩到目标点
    return np.sum(weights[:, None] * z_data_flat, axis=0)

def ode_func_flat(t, x_flat, schedule, z_data_flat):
    """ODE 向量场 v_t(x_t)，针对展平数组。"""
    alpha, sigma, d_alpha, d_sigma = schedule.get_params(t)
    
    if sigma < 1e-4: 
        return np.zeros_like(x_flat)
        
    hat_z_flat = compute_hat_z_flat(x_flat, z_data_flat, alpha, sigma)
    lambda_t = d_alpha - (d_sigma * alpha) / sigma
    
    # 返回导数 dx/dt
    return (d_sigma / sigma) * x_flat + lambda_t * hat_z_flat

def simulate_ode_trajectory(z_data, schedule_mode="VP", num_samples=4, steps=50, method='RK45'):
    """
    执行数值模拟，从 t=0 (噪声) 积分到 t=1 (数据)。
    
    Parameters:
    -----------
    z_data: np.ndarray, 真实的参考流形数据，支持任意维度，例如 (N, 1, 64, 64)
    schedule_mode: str, 'VP', 'VE' 或 'OT'
    num_samples: int, 要生成多少张图
    steps: int, 采样的离散步数（用于记录轨迹）
    method: str, scipy solve_ivp 的求解器，'RK45' (自适应步长) 或 'Euler' (需自写固定步长)
    
    Returns:
    --------
    trajectories: np.ndarray, 形状为 (num_samples, steps, *data_shape)
    """
    schedule = TheoreticalSchedule(mode=schedule_mode)
    
    # 获取数据原始维度并进行全局展平 (为了求解器速度)
    data_shape = z_data.shape[1:] 
    D = np.prod(data_shape)
    z_data_flat = z_data.reshape(-1, D)
    
    # 1. 在 t=0 初始化纯噪声
    _, initial_sigma, _, _ = schedule.get_params(0.0)
    x0_flat = np.random.randn(num_samples, D) * initial_sigma
    
    t_span = (0.0, 1.0)
    t_eval = np.linspace(0.0, 1.0, steps)
    
    trajectories = []
    
    for i in range(num_samples):
        print(f"Simulating trajectory {i+1}/{num_samples} (Mode: {schedule_mode})...")
        
        # 定义当前样本的 ivp 函数
        def ivp_wrapper(t, y):
            return ode_func_flat(t, y, schedule, z_data_flat)
        
        # 运行 ODE 求解器
        sol = solve_ivp(
            fun=ivp_wrapper,
            t_span=t_span,
            y0=x0_flat[i],
            t_eval=t_eval,
            method=method,
            vectorized=False
        )
        
        # 还原维度: sol.y 形状是 (D, steps) -> 转置为 (steps, D) -> 恢复图像维度
        traj_reshaped = sol.y.T.reshape(steps, *data_shape)
        trajectories.append(traj_reshaped)
        
    return np.array(trajectories)
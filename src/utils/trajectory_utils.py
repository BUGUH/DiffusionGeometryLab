# 文件路径: src/utils/trajectory_utils.py
import numpy as np

def extract_hat_x0_from_trajectory(trajectories, t_axis, mode):
    """
    【修正版】利用数值差分和概率流 ODE 提取 \hat{x}_0
    注意：此处的 t_axis 约定 t=0 为纯噪声，t=1 为干净数据！
    """
    # np.gradient 算出来的速度：v_t = dx / dt
    v_t = np.gradient(trajectories, t_axis, axis=1) 
    
    hat_x0_traj = np.zeros_like(trajectories)
    
    for j, t in enumerate(t_axis):
        x_t = trajectories[:, j, ...]
        v = v_t[:, j, ...]
        
        if mode == "OT":
            # 修正后的时间反转 OT 公式
            hat_x0 = x_t + (1.0 - t) * v
            
        elif mode == "VE":
            sigma_min, sigma_max = 0.01, 50.0
            beta = np.log(sigma_max / sigma_min)
            # 修正后的时间反转 VE 公式
            hat_x0 = x_t + (1.0 / beta) * v
            
        elif mode == "VP":
            pi_over_2 = np.pi / 2.0
            # 修正后的时间反转 VP 公式
            alpha_tau = np.sin(pi_over_2 * t)
            sigma_tau = np.cos(pi_over_2 * t)
            
            hat_x0 = alpha_tau * x_t + (2.0 / np.pi) * sigma_tau * v
        else:
            raise ValueError(f"Unknown mode: {mode}")
                
        hat_x0_traj[:, j, ...] = hat_x0
        
    return np.clip(hat_x0_traj, -1.0, 1.0)
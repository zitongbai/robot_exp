import numpy as np
import matplotlib.pyplot as plt

from control_law import AdaptiveControlLaw


def compute_true_M(q, params):
    """
    计算机器人的真实质量矩阵
    
    参数:
    q: 关节角度向量 (n,)
    params: 包含机器人参数的对象，需要有以下属性:
        - n: 关节数量
        - links: 连杆长度列表
        - m_links: 连杆质量列表  
        - I_links: 连杆惯性矩列表
    
    返回:
    M: 质量矩阵 (n, n)
    """
    n = params.n
    M = np.zeros((n, n))
    T = np.eye(4)
    z_axes = np.zeros((3, n))
    r_com = np.zeros((3, n))
    
    # 计算每个连杆的z轴方向和质心位置
    for i in range(n):
        # 绕z轴的旋转矩阵
        cos_q = np.cos(q[i])
        sin_q = np.sin(q[i])
        Rz = np.array([[cos_q, -sin_q, 0],
                       [sin_q,  cos_q, 0],
                       [0,      0,     1]])
        
        # 齐次变换矩阵
        T_i = np.array([[Rz[0,0], Rz[0,1], Rz[0,2], 0],
                        [Rz[1,0], Rz[1,1], Rz[1,2], 0],
                        [Rz[2,0], Rz[2,1], Rz[2,2], params.links[i]],
                        [0,       0,       0,       1]])
        
        T = T @ T_i
        z_axes[:, i] = T[:3, 2]  # z轴方向
        r_com[:, i] = T[:3, 3] - 0.5 * params.links[i] * T[:3, 2]  # 质心位置
    
    # 计算质量矩阵
    for i in range(n):
        m_i = params.m_links[i]
        Jv = np.zeros((3, n))  # 线速度雅可比
        Jw = np.zeros((3, n))  # 角速度雅可比
        
        for j in range(i + 1):  # j从0到i
            Jv[:, j] = np.cross(z_axes[:, j], r_com[:, i])
            Jw[:, j] = z_axes[:, j]
        
        I_i = np.eye(3) * params.I_links[i]
        M = M + m_i * (Jv.T @ Jv) + (Jw.T @ I_i @ Jw)
    
    # 确保矩阵对称并添加小的正则化项
    M = (M + M.T) / 2 + 1e-6 * np.eye(n)
    return M

if __name__ == "__main__":
    
    num_dof = 7
    
    
    


